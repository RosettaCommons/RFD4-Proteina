"""AF3 launcher daemon. Two subcommands:

- ``serve`` — one Unix-socket daemon per GPU. Spawns a warm AF3 worker
  (apptainer subprocess) at startup and forwards ``fold`` requests to it.
- ``launch`` — slurm-side supervisor. Spawns N ``serve`` daemons, pings
  each until ready, daemonises (printing eval-able exports), shuts every
  daemon down on SIGTERM/SIGINT/SIGHUP.

Cluster paths come from ``.ipd/.env.<cluster>``:
    AF3_APPTAINER_PATH, AF3_SCRIPTS_DIR, AF3_JAX_CACHE_DIR, AF3_MODEL_DIR.

Socket protocol (line-delimited JSON):
    {"cmd": "fold", "manifest": "...", "save_full_outputs": false}
        -> {"status": "ok"}
    {"cmd": "ping"}     -> {"status": "ok"}
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
import time
from pathlib import Path

from af3_apptainer import (
    AF3_NUM_RECYCLES,
    build_apptainer_fold_command,
)

_REQUIRED_ENV_VARS = ("AF3_APPTAINER_PATH", "AF3_SCRIPTS_DIR", "AF3_JAX_CACHE_DIR", "AF3_MODEL_DIR")


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
            f"Stale AF3 daemon socket from another job: {path}\n" f"Remove it manually: rm {path}"
        ) from None


def _serve(args: argparse.Namespace) -> int:
    sock_path = args.socket_path
    if os.path.exists(sock_path):
        _safe_unlink(sock_path)
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.bind(sock_path)
    sock.listen(1)
    os.chmod(sock_path, 0o600)
    print(f"[af3-daemon] listening on {sock_path} (pid={os.getpid()})", file=sys.stderr, flush=True)

    # Spawn the warm worker before accepting connections; clients ping
    # until ready while the socket listens.
    print(
        f"[af3-daemon] starting warm AF3 worker (gpu={args.gpu_device}, num_recycles={AF3_NUM_RECYCLES})",
        file=sys.stderr,
        flush=True,
    )
    worker = _spawn_warm_worker(args.batch_script, args.gpu_device)
    try:
        _wait_for_worker_ready(worker)
        print(f"[af3-daemon] warm AF3 worker ready (pid={worker.pid})", file=sys.stderr, flush=True)
    except Exception as exc:
        print(f"[af3-daemon] warm worker failed: {exc}", file=sys.stderr, flush=True)
        worker.terminate()
        return 1

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
                    resp = {"status": "ok"}
                elif cmd == "shutdown":
                    conn.sendall((json.dumps({"status": "ok"}) + "\n").encode())
                    conn.close()
                    break
                elif cmd == "fold":
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
        # Tell the warm worker to exit cleanly; if that fails, kill it.
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

_SOCKET_TEMPLATE = "/tmp/af3_{job_id}/af3_daemon_rank{rank}.sock"
_PING_RETRIES = 30
_PING_INTERVAL_S = 0.5


def _socket_template() -> str:
    """Return the socket template, scoped to the current SLURM job (or PID as fallback)."""
    job_id = os.environ.get("SLURM_JOB_ID") or str(os.getpid())
    return _SOCKET_TEMPLATE.replace("{job_id}", job_id)


def _socket_path(rank: int) -> str:
    return _socket_template().format(rank=rank)


def _socket_dir() -> str:
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


def _spawn_one(rank: int, log_dir: Path) -> subprocess.Popen:
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

    procs: list[subprocess.Popen] = [_spawn_one(r, log_dir) for r in range(args.num_gpus)]

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

    args = p.parse_args()
    if args.cmd == "serve":
        return _serve(args)
    return _launch(args)


if __name__ == "__main__":
    raise SystemExit(main())
