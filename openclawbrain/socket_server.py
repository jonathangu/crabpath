"""Unix socket wrapper for the OpenClawBrain daemon."""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import logging
import os
import signal
import sys
import time
import uuid
from collections import deque
from pathlib import Path


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="openclawbrain.socket_server")
    parser.add_argument("--state", required=True)
    parser.add_argument("--socket-path")
    parser.add_argument("--embed-model", default="text-embedding-3-small")
    parser.add_argument("--auto-save-interval", type=int, default=10)
    parser.add_argument("--force", action="store_true", help="Bypass state lock (expert use)")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    return parser.parse_args(argv)


def _default_socket_path(state_path: str) -> str:
    expanded = Path(state_path).expanduser().parent
    agent = expanded.name or "default"
    return str(Path.home() / ".openclawbrain" / agent / "daemon.sock")


def _default_pid_path(socket_path: str) -> str:
    return str(Path(socket_path).expanduser().parent / "daemon.pid")


async def _readline_with_timeout(stream: asyncio.StreamReader, timeout: float) -> bytes:
    return await asyncio.wait_for(stream.readline(), timeout=timeout)


class SocketDaemonServer:
    """Run an NDJSON daemon behind a Unix socket transport."""

    def __init__(
        self,
        state_path: str,
        socket_path: str | None,
        embed_model: str,
        auto_save_interval: int,
        force: bool,
        verbose: bool,
    ) -> None:
        self.state_path = Path(state_path).expanduser()
        self.socket_path = (
            Path(socket_path).expanduser()
            if socket_path
            else Path(_default_socket_path(str(self.state_path)))
        )
        self.embed_model = embed_model
        self.auto_save_interval = auto_save_interval
        self.force = force
        self.pid_path = Path(_default_pid_path(str(self.socket_path)))

        self._logger = logging.getLogger("openclawbrain.socket_server")
        level = logging.DEBUG if verbose else logging.INFO
        handler = logging.StreamHandler(sys.stderr)
        self._logger.setLevel(level)
        if not self._logger.handlers:
            self._logger.addHandler(handler)
        self._logger.propagate = False

        self._daemon_lock = asyncio.Lock()
        self._restart_history: deque[float] = deque()
        self._daemon = None
        self._server: asyncio.base_events.Server | None = None
        self._stopping = False
        self._watcher: asyncio.Task[None] | None = None

        self._stop_event: asyncio.Event | None = None

    def _write_pid(self) -> None:
        self.pid_path.write_text(str(os.getpid()), encoding="utf-8")
        self._logger.debug("pid file written: %s", self.pid_path)

    def _remove_pid(self) -> None:
        if self.pid_path.exists() and self.pid_path.read_text(encoding="utf-8") == str(os.getpid()):
            self.pid_path.unlink(missing_ok=True)
            self._logger.debug("pid file removed: %s", self.pid_path)

    def _can_restart(self) -> bool:
        now = time.monotonic()
        while self._restart_history and now - self._restart_history[0] > 60:
            self._restart_history.popleft()
        return len(self._restart_history) < 3

    async def _kill_daemon(self) -> None:
        if self._daemon is None or self._daemon.returncode is not None:
            return
        self._logger.info("terminating daemon process")
        self._daemon.terminate()
        try:
            await asyncio.wait_for(self._daemon.wait(), timeout=2)
        except TimeoutError:
            self._daemon.kill()
            await asyncio.wait_for(self._daemon.wait(), timeout=2)
        self._daemon = None

    async def _start_daemon(self) -> None:
        cmd = [
            sys.executable,
            "-m",
            "openclawbrain",
            "daemon",
            "--state",
            str(self.state_path),
            "--embed-model",
            self.embed_model,
            "--auto-save-interval",
            str(self.auto_save_interval),
        ]
        if self.force:
            cmd.append("--force")
        self._logger.info("starting daemon: %s", " ".join(cmd))
        self._daemon = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )

    async def _ensure_daemon(self) -> None:
        if self._daemon is not None and self._daemon.returncode is None:
            return
        await self._restart_daemon("daemon not running")

    async def _restart_daemon(self, reason: str) -> None:
        if self._stopping:
            return
        if not self._can_restart():
            self._logger.error("daemon restart limit exceeded in 60s")
            if not self._stopping:
                await self._shutdown_server()
            raise RuntimeError("daemon restart limit exceeded in 60s")
        self._logger.warning("restarting daemon: %s", reason)
        self._restart_history.append(time.monotonic())
        await self._kill_daemon()
        await self._start_daemon()

    def _format_response(self, req_id: object, payload: object | None = None, error: str | None = None) -> str:
        if error is None:
            return json.dumps({"id": req_id, "result": payload})
        return json.dumps({"id": req_id, "error": {"code": -1, "message": error}})

    async def _forward(self, request: dict[str, object], req_id: object) -> str:
        payload = json.dumps(request)
        if self._daemon is None or self._daemon.stdin is None or self._daemon.stdout is None:
            raise RuntimeError("daemon unavailable")
        self._daemon.stdin.write((payload + "\n").encode("utf-8"))
        await self._daemon.stdin.drain()
        raw = await _readline_with_timeout(self._daemon.stdout, timeout=30)
        if not raw:
            raise RuntimeError("daemon returned empty response")
        response = json.loads(raw.decode("utf-8"))
        if not isinstance(response, dict):
            raise RuntimeError("daemon returned non-object response")
        if response.get("id") != req_id:
            raise RuntimeError("daemon response id mismatch")
        return json.dumps(response)

    async def _process_request(self, request: dict[str, object]) -> str:
        req_id = request.get("id", str(uuid.uuid4()))
        if request.get("id") is None:
            request["id"] = req_id
        if not isinstance(request.get("method"), str):
            raise ValueError("method must be a string")
        params = request.get("params")
        if params is None:
            request["params"] = {}
        elif not isinstance(params, dict):
            raise ValueError("params must be an object")

        method = request.get("method")

        async with self._daemon_lock:
            await self._ensure_daemon()
            try:
                result = await self._forward(request, req_id)
            except (BrokenPipeError, OSError, json.JSONDecodeError, asyncio.TimeoutError) as exc:
                # best effort: restart and retry once
                self._logger.warning("daemon request failed: %s", exc)
                await self._restart_daemon("communication failure")
                await self._ensure_daemon()
                result = await self._forward(request, req_id)

        # If client requested shutdown, initiate server shutdown
        if method == "shutdown":
            asyncio.create_task(self._shutdown_server())

        return result

    async def _handle_client(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        peer = writer.get_extra_info("peername")
        self._logger.debug("client connected: %s", peer)
        try:
            while True:
                raw = await reader.readline()
                if not raw:
                    break
                line = raw.strip()
                if not line:
                    continue
                request: dict[str, object] | None = None
                try:
                    request = json.loads(line.decode("utf-8"))
                    if not isinstance(request, dict):
                        raise ValueError("request must be object")
                    if request.get("id") is None:
                        request["id"] = str(uuid.uuid4())
                    payload = await self._process_request(request)
                except (json.JSONDecodeError, ValueError, RuntimeError) as exc:
                    request_id = request.get("id") if request is not None else None
                    payload = self._format_response(request_id, error=str(exc))
                except Exception as exc:  # noqa: BLE001
                    request_id = request.get("id") if request is not None else None
                    self._logger.exception("unexpected request failure: %s", exc)
                    payload = self._format_response(request_id, error=f"unexpected error: {exc}")
                writer.write((payload + "\n").encode("utf-8"))
                await writer.drain()
        finally:
            writer.close()
            await writer.wait_closed()
            self._logger.debug("client disconnected: %s", peer)

    async def _shutdown_server(self) -> None:
        if self._stopping:
            return
        self._stopping = True

        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()

        try:
            if self._daemon is not None and self._daemon.returncode is None:
                try:
                    shutdown_req = {"id": "__shutdown__", "method": "shutdown", "params": {}}
                    request_data = json.dumps(shutdown_req)
                    self._daemon.stdin.write((request_data + "\n").encode("utf-8"))
                    await self._daemon.stdin.drain()
                    await _readline_with_timeout(self._daemon.stdout, timeout=5)
                except asyncio.TimeoutError:
                    self._logger.warning("daemon shutdown timed out")
                except Exception as exc:  # noqa: BLE001
                    self._logger.warning("daemon shutdown request failed: %s", exc)
        finally:
            await self._kill_daemon()

            if self.socket_path.exists():
                self.socket_path.unlink(missing_ok=True)
            self._remove_pid()

        if self._stop_event is not None and not self._stop_event.is_set():
            self._stop_event.set()

    async def serve_forever(self) -> None:
        self._write_pid()
        await self._start_daemon()

        # Stale socket cleanup: test if another daemon is alive
        if self.socket_path.exists():
            import socket as _socket
            test_sock = _socket.socket(_socket.AF_UNIX, _socket.SOCK_STREAM)
            try:
                test_sock.connect(str(self.socket_path))
                test_sock.close()
                self._logger.error("another daemon is already running on %s", self.socket_path)
                raise SystemExit(f"daemon already running on {self.socket_path}")
            except (ConnectionRefusedError, FileNotFoundError, OSError):
                # Stale socket — clean it up
                self._logger.info("removing stale socket %s", self.socket_path)
                self.socket_path.unlink(missing_ok=True)
            finally:
                try:
                    test_sock.close()
                except OSError:
                    pass
        self.socket_path.parent.mkdir(parents=True, exist_ok=True)

        self._server = await asyncio.start_unix_server(self._handle_client, path=str(self.socket_path))
        self._logger.info("listening on %s", self.socket_path)

        self._stop_event = asyncio.Event()
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            try:
                loop.add_signal_handler(sig, lambda: asyncio.create_task(self._shutdown_server()))
            except NotImplementedError:
                self._logger.debug("signal handler unsupported for %s", sig)

        self._watcher = asyncio.create_task(self._watch_daemon())

        await self._stop_event.wait()

        if self._watcher is not None:
            self._watcher.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._watcher

    async def _watch_daemon(self) -> None:
        while not self._stopping:
            if self._daemon is None or self._daemon.returncode is not None:
                if self._daemon is not None:
                    self._logger.warning("daemon exited with code %s", self._daemon.returncode)
                async with self._daemon_lock:
                    try:
                        await self._restart_daemon("daemon exited unexpectedly")
                    except RuntimeError as exc:
                        self._logger.error("daemon failed to restart: %s", exc)
                        return
                    except Exception as exc:  # noqa: BLE001
                        self._logger.error("watchdog restart failed: %s", exc)
                        await self._shutdown_server()
                        return
            await asyncio.sleep(0.5)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    server = SocketDaemonServer(
        state_path=args.state,
        socket_path=args.socket_path,
        embed_model=args.embed_model,
        auto_save_interval=args.auto_save_interval,
        force=args.force,
        verbose=args.verbose,
    )
    try:
        asyncio.run(server.serve_forever())
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
