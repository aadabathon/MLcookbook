"""Bounded loopback HTTP shared by the local inference connectors."""

from __future__ import annotations

import http.client
import io
import ipaddress
import json
import select
from contextlib import contextmanager
from time import monotonic
from types import SimpleNamespace
from urllib.parse import urlsplit

MAX_JSON = 2 * 1024 * 1024


class LocalHTTPError(ValueError):
    def __init__(self, status: int, message: str):
        self.status = status
        super().__init__(f"Local LLM returned HTTP {status}: {message}")


def local_url(value: str, *, api_path: str = "") -> str:
    """Accept only explicit loopback HTTP endpoints; no proxies or redirects."""
    if not isinstance(value, str) or len(value) > 2048:
        raise ValueError("Enter a local inference URL, for example http://127.0.0.1:8080.")
    try:
        parsed = urlsplit(value.strip())
        host = parsed.hostname
        port = parsed.port if parsed.port is not None else 80
        if host == "localhost":
            host = "127.0.0.1"
        if (parsed.scheme != "http" or not host or not ipaddress.ip_address(host).is_loopback
                or parsed.username or parsed.password or parsed.query or parsed.fragment
                or parsed.path not in ("", "/", api_path, api_path + "/") or not 1 <= port <= 65535):
            raise ValueError
    except ValueError:
        raise ValueError("Use a loopback HTTP URL such as http://127.0.0.1:8080; remote hosts and credentials are unsupported.") from None
    return f"http://{'[' + host + ']' if ':' in host else host}:{port}"


def _check(stop_requested, deadline=None):
    if stop_requested is not None and stop_requested():
        raise InterruptedError("Local LLM operation cancelled.")
    if deadline is not None and monotonic() >= deadline:
        raise TimeoutError("Local LLM request reached its time limit. The previous completed report is preserved.")


class _PollingReader(io.RawIOBase):
    """Interrupt idle reads without retrying a timed-out socket.makefile buffer."""

    def __init__(self, transport, stop_requested, deadline):
        self.transport, self.stop_requested, self.deadline = transport, stop_requested, deadline

    def readable(self):
        return True

    def close(self):
        self.transport.close()
        super().close()

    def readinto(self, buffer):
        while True:
            _check(self.stop_requested, self.deadline)
            if select.select([self.transport], [], [], 0.05)[0]:
                try:
                    return self.transport.recv_into(buffer)
                except BlockingIOError:
                    continue


@contextmanager
def _response(base_url, path, body, seconds, stop_requested):
    _check(stop_requested)
    parsed = urlsplit(base_url)
    connection = http.client.HTTPConnection(parsed.hostname, parsed.port, timeout=min(3.0, seconds))
    deadline = monotonic() + seconds
    response = None
    try:
        connection.connect()
        transport = connection.sock
        data = None if body is None else json.dumps(body, ensure_ascii=False, allow_nan=False).encode("utf-8")
        connection.request("GET" if body is None else "POST", path, body=data,
                           headers={"Content-Type": "application/json", "Connection": "close"})
        transport.setblocking(False)
        # Keep stdlib HTTP parsing (including chunked transfer); poll only socket reads.
        def response_class(sock, **kwargs):
            reader = io.BufferedReader(_PollingReader(sock.dup(), stop_requested, deadline))
            return http.client.HTTPResponse(SimpleNamespace(makefile=lambda mode: reader), **kwargs)
        connection.response_class = response_class
        response = connection.getresponse()
        if response.status != 200:
            message = response.read(8192).decode("utf-8", errors="replace")
            try:
                message = json.loads(message).get("error", message)
            except (ValueError, AttributeError):
                pass
            if isinstance(message, dict):
                message = message.get("message", str(message))
            raise LocalHTTPError(response.status, str(message)[:1000])
        yield response
        _check(stop_requested, deadline)
    except (OSError, http.client.HTTPException) as exc:
        _check(stop_requested, deadline)
        if isinstance(exc, ConnectionRefusedError) or getattr(exc, "winerror", None) == 10061:
            raise ConnectionError(f"No local inference server is listening at {base_url}. "
                                  "Start the selected server separately, then refresh. A GGUF file path does not start a server.") from exc
        raise ConnectionError(f"Cannot complete the local inference request at {base_url}: {exc}. "
                              "Check that the selected server is running and the address is correct.") from exc
    finally:
        if response is not None:
            response.close()
        connection.close()


def _object(raw):
    try:
        value = json.loads(raw.decode("utf-8"))
    except (ValueError, UnicodeError) as exc:
        raise ValueError("Local LLM returned invalid UTF-8 JSON.") from exc
    if not isinstance(value, dict):
        raise ValueError("Local LLM returned a JSON value that is not an object.")
    if value.get("error"):
        error = value["error"]
        raise ValueError(f"Local LLM: {str(error.get('message', error) if isinstance(error, dict) else error)[:1000]}")
    return value


def _json(base_url, path, body=None, *, seconds=5.0, stop_requested=None):
    with _response(base_url, path, body, seconds, stop_requested) as response:
        data = response.read(MAX_JSON + 1)
        if len(data) > MAX_JSON:
            raise ValueError("Local LLM metadata response exceeds the size limit.")
        return _object(data)


