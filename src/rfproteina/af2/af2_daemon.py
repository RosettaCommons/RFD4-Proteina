"""AF2 launcher daemon.

``serve`` runs one Unix-socket daemon per GPU and owns a warm AF2 worker.
``launch`` starts the requested daemons, reports their socket template, and
shuts them down when the supervisor receives a termination signal.

Socket protocol (line-delimited JSON):
    {"cmd": "fold", "manifest": "..."} -> {"status": "ok"}
    {"cmd": "ping"}     -> {"status": "ok"}
    {"cmd": "shutdown"} -> {"status": "ok"}, daemon exits.
"""

import argparse
import contextlib
import json
import math
import os
import selectors
import shlex
import signal
import socket
import subprocess
import sys
import threading
import time
from collections import deque
from pathlib import Path

if __package__:
    from .af2_apptainer import AF2_NUM_RECYCLES, build_container_fold_command
else:  # Direct script execution.
    from af2_apptainer import AF2_NUM_RECYCLES, build_container_fold_command

_REQUIRED_ENV_VARS = ("AF2_IMAGE", "AF2_JAX_CACHE_DIR", "AF2_MODEL_DIR")
_DEFAULT_FOLD_TIMEOUT_S = float(os.environ.get("AF2_FOLD_TIMEOUT_S", "3600"))


def _spawn_warm_worker(batch_script: str, gpu_device: int) -> subprocess.Popen:
    """Spawn the persistent AF2 worker subprocess."""
    cmd = build_container_fold_command(
        manifest_path=None,
        gpu_device=gpu_device,
        batch_script=batch_script,
        serve=True,
    )
    print(f"[af2-daemon] worker command: {shlex.join(cmd)}", file=sys.stderr, flush=True)
    return subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=1,
        text=True,
    )


def _wait_for_worker_ready(
    worker: subprocess.Popen,
    stderr_tail: deque[str],
    stderr_lock: threading.Lock,
    stderr_done: threading.Event,
    timeout_s: float = 600.0,
) -> None:
    """Block until the warm worker prints its ready status line."""
    t0 = time.perf_counter()
    with selectors.DefaultSelector() as selector:
        selector.register(worker.stdout, selectors.EVENT_READ)
        while True:
            if worker.poll() is not None:
                stderr_done.wait(timeout=1.0)
                with stderr_lock:
                    detail = "\n".join(stderr_tail).strip()
                message = f"warm AF2 worker exited during startup (rc={worker.returncode})"
                raise RuntimeError(f"{message}:\n{detail}" if detail else message)
            elapsed = time.perf_counter() - t0
            if elapsed > timeout_s:
                raise TimeoutError(f"warm AF2 worker did not signal ready within {timeout_s:.0f}s")
            if not selector.select(timeout=min(0.1, timeout_s - elapsed)):
                continue
            line = worker.stdout.readline()
            if not line:
                continue
            try:
                msg = json.loads(line.strip())
            except json.JSONDecodeError:
                continue
            if msg.get("status") == "ready":
                return


def _request_timeout_s(req: dict, now_s: float | None = None) -> float:
    """Return request time remaining, including any socket queue or warmup time."""
    if "deadline_unix_s" in req:
        try:
            deadline_s = float(req["deadline_unix_s"])
        except (TypeError, ValueError) as exc:
            raise ValueError("fold request deadline_unix_s must be finite seconds") from exc
        if not math.isfinite(deadline_s):
            raise ValueError("fold request deadline_unix_s must be finite seconds")
        remaining_s = deadline_s - (time.time() if now_s is None else now_s)
        if remaining_s <= 0:
            raise TimeoutError("AF2 fold request deadline expired")
        return remaining_s

    try:
        timeout_s = float(req.get("timeout_s", _DEFAULT_FOLD_TIMEOUT_S))
    except (TypeError, ValueError) as exc:
        raise ValueError("fold request timeout_s must be positive finite seconds") from exc
    if not math.isfinite(timeout_s) or timeout_s <= 0:
        raise ValueError("fold request timeout_s must be positive finite seconds")
    return timeout_s


def _handle_fold(req: dict, worker: subprocess.Popen) -> dict:
    """Send one manifest path to the warm worker and return its response."""
    manifest = req.get("manifest")
    if not manifest:
        return {"status": "error", "message": "fold request missing 'manifest' key"}
    timeout_s = _request_timeout_s(req)
    if worker.poll() is not None:
        return {"status": "error", "message": f"warm AF2 worker has exited (rc={worker.returncode})"}
    try:
        worker.stdin.write(f"{manifest}\n")
        worker.stdin.flush()
    except (BrokenPipeError, OSError) as exc:
        return {"status": "error", "message": f"warm AF2 worker pipe broken: {exc}"}

    deadline = time.monotonic() + timeout_s
    with selectors.DefaultSelector() as selector:
        selector.register(worker.stdout, selectors.EVENT_READ)
        while True:
            if worker.poll() is not None:
                return {"status": "error", "message": f"warm AF2 worker has exited (rc={worker.returncode})"}
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                try:
                    worker.terminate()
                    worker.wait(timeout=5)
                except ProcessLookupError:
                    pass
                except subprocess.TimeoutExpired:
                    worker.kill()
                    worker.wait(timeout=5)
                return {"status": "error", "message": f"warm AF2 worker timed out after {timeout_s:.0f}s"}
            if not selector.select(timeout=min(0.1, remaining)):
                continue
            line = worker.stdout.readline()
            if not line:
                return {"status": "error", "message": "warm AF2 worker closed stdout"}
            try:
                return json.loads(line.strip())
            except json.JSONDecodeError:
                return {"status": "error", "message": f"non-JSON from worker: {line!r}"}


def _safe_unlink(path: str) -> None:
    """Remove a socket file, with a clear error for another user's socket."""
    try:
        os.unlink(path)
    except PermissionError:
        raise PermissionError(
            f"Stale AF2 daemon socket from another job: {path}\nRemove it manually: rm {path}"
        ) from None


def _start_warm_worker(batch_script: str, gpu_device: int) -> tuple[subprocess.Popen, threading.Event, list[Exception]]:
    """Start one worker and its asynchronous stderr/warmup monitors."""
    print(
        f"[af2-daemon] starting warm AF2 worker (gpu={gpu_device}, num_recycles={AF2_NUM_RECYCLES})",
        file=sys.stderr,
        flush=True,
    )
    worker = _spawn_warm_worker(batch_script, gpu_device)
    stderr_tail: deque[str] = deque(maxlen=20)
    stderr_lock = threading.Lock()
    stderr_done = threading.Event()

    def _drain_stderr() -> None:
        try:
            for line in worker.stderr:
                message = line.rstrip()
                with stderr_lock:
                    stderr_tail.append(message)
                print(message, file=sys.stderr, flush=True)
        finally:
            stderr_done.set()

    threading.Thread(target=_drain_stderr, daemon=True, name="af2-stderr").start()

    worker_ready = threading.Event()
    worker_errors: list[Exception] = []

    def _warmup() -> None:
        try:
            _wait_for_worker_ready(worker, stderr_tail, stderr_lock, stderr_done)
            print(f"[af2-daemon] warm AF2 worker ready (pid={worker.pid})", file=sys.stderr, flush=True)
        except Exception as exc:
            print(f"[af2-daemon] warm worker failed: {exc}", file=sys.stderr, flush=True)
            worker_errors.append(exc)
        finally:
            worker_ready.set()

    threading.Thread(target=_warmup, daemon=True, name="af2-warmup").start()
    return worker, worker_ready, worker_errors


def _replace_exited_worker(
    worker: subprocess.Popen,
    batch_script: str,
    gpu_device: int,
) -> tuple[subprocess.Popen, threading.Event, list[Exception]] | None:
    """Return a fresh warm-worker session when ``worker`` has exited."""
    if worker.poll() is None:
        return None
    print("[af2-daemon] replacing exited warm worker", file=sys.stderr, flush=True)
    return _start_warm_worker(batch_script, gpu_device)


def _serve(args: argparse.Namespace) -> int:
    if args.skip_if_no_env and not all(os.environ.get(v) for v in _REQUIRED_ENV_VARS):
        print("[af2-daemon] AF2 env vars not set; not serving.", file=sys.stderr)
        return 0

    sock_path = args.socket_path
    if os.path.exists(sock_path):
        _safe_unlink(sock_path)
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.bind(sock_path)
    sock.listen(1)
    os.chmod(sock_path, 0o600)
    print(f"[af2-daemon] listening on {sock_path} (pid={os.getpid()})", file=sys.stderr, flush=True)

    worker, worker_ready, worker_errors = _start_warm_worker(args.batch_script, args.gpu_device)

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
                    if replacement := _replace_exited_worker(worker, args.batch_script, args.gpu_device):
                        worker, worker_ready, worker_errors = replacement
                    warmup_timeout_s = min(1200.0, _request_timeout_s(req))
                    if not worker_ready.wait(timeout=warmup_timeout_s):
                        resp = {
                            "status": "error",
                            "message": f"AF2 worker warmup timed out after {warmup_timeout_s:.0f}s",
                        }
                    elif worker_errors:
                        resp = {
                            "status": "error",
                            "message": f"AF2 worker failed during warmup: {worker_errors[0]}",
                        }
                    else:
                        resp = _handle_fold(req, worker)
                    if replacement := _replace_exited_worker(worker, args.batch_script, args.gpu_device):
                        worker, worker_ready, worker_errors = replacement
                else:
                    resp = {"status": "error", "message": f"unknown cmd: {cmd}"}
                conn.sendall((json.dumps(resp) + "\n").encode())
            except Exception as exc:
                with contextlib.suppress(OSError):
                    conn.sendall((json.dumps({"status": "error", "message": str(exc)}) + "\n").encode())
            finally:
                conn.close()
    finally:
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


_SOCKET_TEMPLATE = "/tmp/af2_{job_id}/af2_daemon_rank{rank}.sock"
_PING_RETRIES = 120
_PING_INTERVAL_S = 5.0


def _socket_template(pid: int | None = None) -> str:
    """Return the socket template scoped to the SLURM job or launcher PID."""
    job_id = os.environ.get("SLURM_JOB_ID") or str(pid if pid is not None else os.getpid())
    return _SOCKET_TEMPLATE.replace("{job_id}", job_id)


def _socket_path(rank: int) -> str:
    return _socket_template().format(rank=rank)


def _socket_dir() -> str:
    job_id = os.environ.get("SLURM_JOB_ID") or str(os.getpid())
    return f"/tmp/af2_{job_id}"


def _ping_until_ready(sock_path: str, proc: subprocess.Popen) -> bool:
    """Open a fresh Unix-socket connection and retry a ping until ready."""
    for _ in range(_PING_RETRIES):
        if proc.poll() is not None:
            return False
        try:
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
                client.settimeout(2)
                client.connect(sock_path)
                client.sendall(b'{"cmd":"ping"}\n')
                client.recv(1024)
            return True
        except (TimeoutError, FileNotFoundError, ConnectionRefusedError, OSError):
            time.sleep(_PING_INTERVAL_S)
    return False


def _send_shutdown(sock_path: str, timeout_s: float = 5.0) -> None:
    """Best-effort shutdown over a daemon socket."""
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
            client.settimeout(timeout_s)
            client.connect(sock_path)
            client.sendall(b'{"cmd":"shutdown"}\n')
            client.recv(1024)
    except OSError:
        pass


def _spawn_one(rank: int, log_dir: Path) -> subprocess.Popen:
    """Start one serve daemon on the socket for ``rank``."""
    os.makedirs(_socket_dir(), exist_ok=True)
    sock = _socket_path(rank)
    if os.path.exists(sock):
        _safe_unlink(sock)
    host = os.environ.get("SLURMD_NODENAME") or os.uname().nodename
    log_file = log_dir / f"af2_daemon_{host}_rank{rank}.log"

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
        print("[af2-daemons] AF2 env vars not set; skipping daemon launch.", file=sys.stderr)
        return 0

    log_dir = Path(args.log_dir).resolve()
    log_dir.mkdir(parents=True, exist_ok=True)

    ready_write: int | None = None
    if args.background:
        ready_read, ready_write = os.pipe()
        supervisor_pid = os.fork()
        if supervisor_pid > 0:
            os.close(ready_write)
            try:
                ready = os.read(ready_read, 1)
            finally:
                os.close(ready_read)
            if ready != b"1":
                with contextlib.suppress(ChildProcessError):
                    os.waitpid(supervisor_pid, 0)
                return 1
            print(f"export AF2_LAUNCHER_SOCKET_TEMPLATE={_socket_template(supervisor_pid)}", flush=True)
            print(f"export AF2_DAEMONS_PID={supervisor_pid}", flush=True)
            return 0
        os.close(ready_read)
        os.setsid()
        os.close(1)

    procs: list[subprocess.Popen] = [_spawn_one(rank, log_dir) for rank in range(args.num_gpus)]

    for rank in range(args.num_gpus):
        if not _ping_until_ready(_socket_path(rank), procs[rank]):
            print(f"[af2-daemons] rank={rank} failed to come up", file=sys.stderr, flush=True)
            host = os.environ.get("SLURMD_NODENAME") or os.uname().nodename
            log_file = log_dir / f"af2_daemon_{host}_rank{rank}.log"
            with contextlib.suppress(OSError):
                print(log_file.read_text(), file=sys.stderr, flush=True)
            for proc in procs:
                proc.terminate()
            if ready_write is not None:
                os.write(ready_write, b"0")
                os.close(ready_write)
            return 1

    if ready_write is not None:
        os.write(ready_write, b"1")
        os.close(ready_write)
    else:
        print(f"AF2_LAUNCHER_SOCKET_TEMPLATE={_socket_template()}", flush=True)

    received: list[int] = []

    def _handler(signum: int, _frame: object) -> None:
        received.append(signum)

    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(sig, _handler)

    try:
        while not received:
            signal.pause()
    finally:
        for rank in range(args.num_gpus):
            _send_shutdown(_socket_path(rank))
        for proc in procs:
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)

    serve = sub.add_parser("serve", help="Run one daemon on one Unix socket.")
    serve.add_argument("--socket-path", required=True)
    serve.add_argument(
        "--batch-script",
        default=str(Path(__file__).resolve().parent / "run_alphafold2_batch.py"),
        help="Path to run_alphafold2_batch.py.",
    )
    serve.add_argument("--gpu-device", type=int, default=0, help="Driver-level GPU index.")
    serve.add_argument(
        "--skip-if-no-env",
        action="store_true",
        help=f"Exit 0 if any of {', '.join(_REQUIRED_ENV_VARS)} is unset.",
    )

    launch = sub.add_parser("launch", help="Spawn N serve daemons and block on signals.")
    launch.add_argument("--num-gpus", type=int, required=True)
    launch.add_argument("--log-dir", required=True, help="Directory for per-rank daemon logs.")
    launch.add_argument(
        "--background",
        action="store_true",
        help="Daemonise; print eval-able AF2 launcher socket and supervisor exports.",
    )
    launch.add_argument(
        "--skip-if-no-env",
        action="store_true",
        help=f"Exit 0 if any of {', '.join(_REQUIRED_ENV_VARS)} is unset.",
    )

    args = parser.parse_args()
    if args.cmd == "serve":
        return _serve(args)
    return _launch(args)


if __name__ == "__main__":
    raise SystemExit(main())
