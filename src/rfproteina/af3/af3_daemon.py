"""AF3 launcher daemon. Two subcommands:

- ``serve`` — one Unix-socket daemon per GPU. Spawns a warm AF3 worker
  (apptainer subprocess) at startup — or, under ``--lazy-warmup``, on the first
  ``prewarm``/``fold`` — and forwards ``fold`` requests to it.
- ``launch`` — slurm-side supervisor. Spawns N ``serve`` daemons, pings
  each until ready, daemonises (printing eval-able exports), shuts every
  daemon down on SIGTERM/SIGINT/SIGHUP.

Cluster paths come from ``.ipd/.env.<cluster>``:
    AF3_IMAGE, AF3_JAX_CACHE_DIR, AF3_MODEL_DIR.

Socket protocol (line-delimited JSON):
    {"cmd": "fold", "manifest": "...", "save_full_outputs": false}
        -> {"status": "ok"}
    {"cmd": "ping"}     -> {"status": "ok"}
    {"cmd": "prewarm"}  -> {"status": "ok"}, starts a lazy daemon's warmup.
    {"cmd": "shutdown"} -> {"status": "ok"}, daemon exits.
"""

import argparse
import contextlib
import json
import os
import signal
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

from af3_apptainer import (
    AF3_NUM_RECYCLES,
    build_apptainer_fold_command,
)

_REQUIRED_ENV_VARS = ("AF3_IMAGE", "AF3_JAX_CACHE_DIR", "AF3_MODEL_DIR")


# ----------------------------------------------------------------------------
# `serve` subcommand: one daemon, one socket.
# ----------------------------------------------------------------------------


def _spawn_warm_worker(batch_script: str, gpu_device: int) -> subprocess.Popen:
    """Spawn the persistent AF3 worker subprocess.

    The worker loads AF3 once at startup and then loops reading manifest paths on stdin.
    """
    cmd = build_apptainer_fold_command(
        manifest_path=None,
        gpu_device=gpu_device,
        save_full_outputs=False,
        batch_script=batch_script,
        serve=True,
    )

    return subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=None,  # inherits daemon stderr → per-rank log file
        bufsize=1,
        text=True,
    )


def _wait_for_worker_ready(worker: subprocess.Popen, timeout_s: float = 600.0) -> None:
    """Block until the warm worker prints its ``{"status": "ready"}`` line."""
    t0 = time.perf_counter()
    while True:
        if worker.poll() is not None:
            raise RuntimeError(f"warm AF3 worker exited during startup (rc={worker.returncode})")
        if time.perf_counter() - t0 > timeout_s:
            raise TimeoutError(f"warm AF3 worker did not signal ready within {timeout_s:.0f}s")
        line = worker.stdout.readline()
        if not line:
            time.sleep(0.1)
            continue
        try:
            msg = json.loads(line.strip())
        except json.JSONDecodeError:
            # Worker may emit non-JSON noise before ready; ignore.
            continue
        if msg.get("status") == "ready":
            return


def _handle_fold(req: dict, worker: subprocess.Popen) -> dict:
    """Send one manifest path to the warm worker and read its status line.

    The worker processes requests serially on a single GPU, so the daemon's
    own ``listen(1)`` is the natural place for queueing. Concurrent format
    threads on the model-driver side already queue at the daemon socket.
    """
    manifest = req.get("manifest")
    if not manifest:
        return {"status": "error", "message": "fold request missing 'manifest' key"}
    if worker.poll() is not None:
        return {"status": "error", "message": f"warm AF3 worker has exited (rc={worker.returncode})"}
    try:
        worker.stdin.write(f"{manifest}\n")
        worker.stdin.flush()
        line = worker.stdout.readline()
    except (BrokenPipeError, OSError) as exc:
        return {"status": "error", "message": f"warm AF3 worker pipe broken: {exc}"}
    if not line:
        return {"status": "error", "message": "warm AF3 worker closed stdout"}
    try:
        return json.loads(line.strip())
    except json.JSONDecodeError:
        return {"status": "error", "message": f"non-JSON from worker: {line!r}"}


def _safe_unlink(path: str) -> None:
    """Remove a socket file, raising a clear error if it belongs to another user."""
    try:
        os.unlink(path)
    except PermissionError:
        raise PermissionError(
            f"Stale AF3 daemon socket from another job: {path}\nRemove it manually: rm {path}"
        ) from None


def _serve(args: argparse.Namespace) -> int:
    if args.skip_if_no_env and not all(os.environ.get(v) for v in _REQUIRED_ENV_VARS):
        print("[af3-daemon] AF3 env vars not set; not serving.", file=sys.stderr)
        return 0

    sock_path = args.socket_path
    if os.path.exists(sock_path):
        _safe_unlink(sock_path)
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.bind(sock_path)
    sock.listen(1)
    os.chmod(sock_path, 0o600)
    print(f"[af3-daemon] listening on {sock_path} (pid={os.getpid()})", file=sys.stderr, flush=True)

    # Warmup, spawn included, runs in a background thread: the socket answers pings
    # immediately while AF3 loads, and any failure is reported to `fold` callers
    # rather than taking the daemon down.
    worker_ready = threading.Event()
    worker: subprocess.Popen | None = None
    worker_error: Exception | None = None
    warmup_started = False

    def warmup() -> None:
        nonlocal worker, worker_error
        try:
            print(
                f"[af3-daemon] starting warm AF3 worker (gpu={args.gpu_device}, num_recycles={AF3_NUM_RECYCLES})",
                file=sys.stderr,
                flush=True,
            )
            worker = _spawn_warm_worker(args.batch_script, args.gpu_device)
            _wait_for_worker_ready(worker)
            print(f"[af3-daemon] warm AF3 worker ready (pid={worker.pid})", file=sys.stderr, flush=True)
        except Exception as exc:
            print(f"[af3-daemon] warm worker failed: {exc}", file=sys.stderr, flush=True)
            worker_error = exc
        finally:
            worker_ready.set()

    def start_warmup_once() -> None:
        """Start the warm worker on the first caller. No lock: every call comes from
        this thread -- the eager path below, or the single-threaded accept loop."""
        nonlocal warmup_started
        if not warmup_started:
            warmup_started = True
            threading.Thread(target=warmup, daemon=True, name="af3-warmup").start()

    # Eager by default, so warmup overlaps whatever the caller does next. Under
    # --lazy-warmup, AF3 loads on the first prewarm/fold instead.
    if not args.lazy_warmup:
        start_warmup_once()

    try:
        while True:
            conn, _ = sock.accept()
            try:
                data = b""
                while not data.endswith(b"\n"):
                    chunk = conn.recv(65536)
                    if not chunk:
                        break
                    data += chunk
                req = json.loads(data.decode())
                cmd = req.get("cmd")
                if cmd == "ping":
                    # Respond immediately — even before warmup completes.
                    # The launcher's _ping_until_ready only checks that a
                    # response arrives, not its content.
                    resp = {"status": "ok"}
                elif cmd == "shutdown":
                    conn.sendall((json.dumps({"status": "ok"}) + "\n").encode())
                    conn.close()
                    break
                elif cmd == "prewarm":
                    # Sent by a client that knows it will fold later, so a lazy daemon
                    # starts loading AF3 now and the warmup still overlaps generation.
                    # A no-op once warmup has started, so it is safe to send repeatedly.
                    start_warmup_once()
                    resp = {"status": "ok"}
                elif cmd == "fold":
                    start_warmup_once()
                    # Block until the warm worker is ready (or failed).
                    if not worker_ready.wait(timeout=1200.0):
                        resp = {"status": "error", "message": "AF3 worker warmup timed out after 1200s"}
                    elif worker_error is not None:
                        resp = {"status": "error", "message": f"AF3 worker failed during warmup: {worker_error}"}
                    else:
                        # Neither timed out nor errored, so the spawn succeeded.
                        resp = _handle_fold(req, worker)
                else:
                    resp = {"status": "error", "message": f"unknown cmd: {cmd}"}
                conn.sendall((json.dumps(resp) + "\n").encode())
            except Exception as e:
                with contextlib.suppress(OSError):
                    conn.sendall((json.dumps({"status": "error", "message": str(e)}) + "\n").encode())
            finally:
                conn.close()
    finally:
        # Tell the warm worker to exit cleanly; if that fails, kill it. `worker` is None
        # when warmup never ran, the spawn failed, or a shutdown landed mid-spawn; in that
        # last case the worker still exits on its own, on stdin EOF, when this process does.
        if worker is not None:
            try:
                if worker.poll() is None:
                    worker.stdin.write("shutdown\n")
                    worker.stdin.flush()
                    worker.wait(timeout=10)
            except Exception:
                with contextlib.suppress(Exception):
                    worker.terminate()
                    worker.wait(timeout=5)

    sock.close()
    if os.path.exists(sock_path):
        os.unlink(sock_path)
    return 0


# ----------------------------------------------------------------------------
# `launch` subcommand: spawn N daemons, signal-handle cleanup.
# ----------------------------------------------------------------------------

# AF3 JIT compilation + model loading can take several minutes on slower GPUs.
# Keep retries generous so the launcher doesn't declare failure during warmup.
# Each retry: 2s socket timeout + _PING_INTERVAL_S sleep = ~7s -> 120 x 7 ~= 14 min.
_PING_RETRIES = 120
_PING_INTERVAL_S = 5.0


def _socket_template() -> str:
    """Return the socket template, scoped to the current SLURM job (or PID as fallback)."""
    return f"{_socket_dir()}/af3_daemon_rank{{rank}}.sock"


def _socket_path(rank: int) -> str:
    return _socket_template().format(rank=rank)


def _socket_dir() -> str:
    # AF3_SOCKET_DIR separates two daemon sets in one job: `_spawn_one` unlinks whatever
    # socket it finds, so a second launch would otherwise rebind the first set's paths.
    override = os.environ.get("AF3_SOCKET_DIR")
    if override:
        return override
    job_id = os.environ.get("SLURM_JOB_ID") or str(os.getpid())
    return f"/tmp/af3_{job_id}"


def _ping_until_ready(sock_path: str) -> bool:
    """Open a fresh AF_UNIX connection and send `ping`. Retry until ready."""
    for _ in range(_PING_RETRIES):
        try:
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
                s.settimeout(2)
                s.connect(sock_path)
                s.sendall(b'{"cmd":"ping"}\n')
                s.recv(1024)
            return True
        except (TimeoutError, FileNotFoundError, ConnectionRefusedError, OSError):
            time.sleep(_PING_INTERVAL_S)
    return False


def _send_shutdown(sock_path: str, timeout_s: float = 5.0) -> None:
    """Best-effort `shutdown` over the daemon's socket."""
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
            s.settimeout(timeout_s)
            s.connect(sock_path)
            s.sendall(b'{"cmd":"shutdown"}\n')
            s.recv(1024)
    except OSError:
        pass


def _spawn_one(rank: int, log_dir: Path, lazy_warmup: bool = False) -> subprocess.Popen:
    """Start one daemon (this script in `serve` mode) on rank's socket."""
    os.makedirs(_socket_dir(), exist_ok=True)
    sock = _socket_path(rank)
    if os.path.exists(sock):
        _safe_unlink(sock)
    # Suffix with hostname so each node's daemons get their own log file
    host = os.environ.get("SLURMD_NODENAME") or os.uname().nodename
    log_file = log_dir / f"af3_daemon_{host}_rank{rank}.log"

    # Hand the log file to Popen, then close the parent's fd
    with open(log_file, "wb") as log_fh:
        return subprocess.Popen(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "serve",
                "--socket-path",
                sock,
                "--gpu-device",
                str(rank),
                *(["--lazy-warmup"] if lazy_warmup else []),
            ],
            stdout=log_fh,
            stderr=subprocess.STDOUT,
        )


def _launch(args: argparse.Namespace) -> int:
    if args.skip_if_no_env and not all(os.environ.get(v) for v in _REQUIRED_ENV_VARS):
        print("[af3-daemons] AF3 env vars not set; skipping daemon launch.", file=sys.stderr)
        return 0

    log_dir = Path(args.log_dir).resolve()
    log_dir.mkdir(parents=True, exist_ok=True)

    procs: list[subprocess.Popen] = [_spawn_one(r, log_dir, lazy_warmup=args.lazy_warmup) for r in range(args.num_gpus)]

    for r in range(args.num_gpus):
        if not _ping_until_ready(_socket_path(r)):
            print(f"[af3-daemons] rank={r} failed to come up", file=sys.stderr, flush=True)
            for p in procs:
                p.terminate()
            return 1

    if args.background:
        # Daemonise: parent prints shell-evalable `export` lines and exits, so
        # the slurm script can `eval "$(... --background)"` and pick up both
        # the socket template and the supervisor PID for an EXIT trap. Child
        # becomes the supervisor that signal-pauses + cleans up.
        pid = os.fork()
        if pid > 0:
            print(f"export AF3_LAUNCHER_SOCKET_TEMPLATE={_socket_template()}", flush=True)
            print(f"export AF3_DAEMONS_PID={pid}", flush=True)
            return 0
        # In the child: detach from tty + close stdout so the parent's
        # process-substitution capture sees EOF and unblocks.
        os.setsid()
        os.close(1)
    else:
        print(f"AF3_LAUNCHER_SOCKET_TEMPLATE={_socket_template()}", flush=True)

    # Block until signalled. SIGTERM (slurm) / SIGINT (Ctrl-C) / SIGHUP all
    # trigger graceful shutdown of every daemon.
    received: list[int] = []

    def _handler(signum: int, _frame: object) -> None:
        received.append(signum)

    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(sig, _handler)

    try:
        while not received:
            signal.pause()
    finally:
        for r in range(args.num_gpus):
            _send_shutdown(_socket_path(r))
        for p in procs:
            try:
                p.wait(timeout=5)
            except subprocess.TimeoutExpired:
                p.kill()
    return 0


# ----------------------------------------------------------------------------
# CLI dispatch
# ----------------------------------------------------------------------------


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("serve", help="One daemon on one Unix socket.")
    s.add_argument("--socket-path", required=True)
    s.add_argument(
        "--batch-script",
        default=str(Path(__file__).resolve().parent / "run_alphafold_batch.py"),
        help="Path to run_alphafold_batch.py.",
    )
    s.add_argument("--gpu-device", type=int, default=0, help="Driver-level GPU index for the warm worker.")
    s.add_argument(
        "--skip-if-no-env",
        action="store_true",
        help=f"Exit 0 without binding the socket if any of {', '.join(_REQUIRED_ENV_VARS)} is unset.",
    )
    s.add_argument(
        "--lazy-warmup",
        action="store_true",
        help="Load AF3 on the first prewarm/fold request instead of at startup, so a daemon "
        "that is never asked to fold never loads AF3 or touches the GPU.",
    )

    l = sub.add_parser("launch", help="Spawn N serve daemons and block on signals.")
    l.add_argument("--num-gpus", type=int, required=True)
    l.add_argument("--log-dir", required=True, help="Directory for per-rank daemon logs.")
    l.add_argument(
        "--background",
        action="store_true",
        help="Daemonise; parent prints eval-able `export AF3_LAUNCHER_SOCKET_TEMPLATE=...` "
        "and `export AF3_DAEMONS_PID=...` then exits.",
    )
    l.add_argument(
        "--skip-if-no-env",
        action="store_true",
        help=f"Exit 0 if any of {', '.join(_REQUIRED_ENV_VARS)} is unset.",
    )
    l.add_argument(
        "--lazy-warmup",
        action="store_true",
        help="Pass --lazy-warmup to each spawned daemon.",
    )

    args = p.parse_args()
    if args.cmd == "serve":
        return _serve(args)
    return _launch(args)


if __name__ == "__main__":
    raise SystemExit(main())
