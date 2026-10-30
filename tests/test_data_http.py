"""Exercise the public-data HTTP reader without depending on public services."""

import json
import socket
import threading
import time

import pytest

from mlcookbook import _data_http

URL = "https://earthquake.usgs.gov/fdsnws/event/1/query"


@pytest.fixture
def provider(monkeypatch):
    connections, threads = [], []

    def install(parts, *, delay=0, idle=False):
        class Connection:
            def __init__(self, hostname, timeout):
                self.sock, self.writer = socket.socketpair()
                self.hostname, self.timeout, self.headers = hostname, timeout, None
                connections.append(self)

            def request(self, method, path, headers):
                self.headers = headers

                def send():
                    try:
                        for part in parts:
                            if delay:
                                time.sleep(delay)
                            self.writer.sendall(part)
                        if idle:
                            time.sleep(.3)
                    except OSError:
                        pass
                    finally:
                        self.writer.close()

                thread = threading.Thread(target=send, daemon=True)
                threads.append(thread)
                thread.start()

            def close(self):
                self.sock.close()

        monkeypatch.setattr(_data_http.http.client, "HTTPSConnection", Connection)
        return connections

    yield install
    for connection in connections:
        connection.close()
        connection.writer.close()
    for thread in threads:
        thread.join(2)


def response(body, **headers):
    fields = {"Connection": "close", "Content-Length": str(len(body)), **headers}
    return b"HTTP/1.1 200 OK\r\n" + b"".join(f"{name}: {value}\r\n".encode() for name, value in fields.items()) + b"\r\n" + body


def test_connection_close_does_not_retire_socket_before_body_and_utf8_fragments(provider):
    body = json.dumps({"place": "Café 地球"}, ensure_ascii=False).encode()
    wire = response(body)
    connections = provider([wire[index:index + 3] for index in range(0, len(wire), 3)])
    assert _data_http.fetch_json(URL) == {"place": "Café 地球"}
    assert connections[0].headers["Accept-Encoding"] == "identity"
    assert connections[0].sock.fileno() == -1


def test_chunked_json_and_empty_204_are_handled(provider):
    provider([b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\nConnection: close\r\n\r\n",
              b"5\r\n{\"x\":\r\n", b"2\r\n1}\r\n", b"0\r\n\r\n"])
    assert _data_http.fetch_json(URL) == {"x": 1}
    provider([b"HTTP/1.1 204 No Content\r\nConnection: close\r\n\r\n"])
    assert _data_http.fetch_json(URL) is None


@pytest.mark.parametrize("body", [b'{"x":1,"x":2}', b'{"x":NaN}', b'bad json', b'{"x":"\xff"}'])
def test_malformed_duplicate_or_nonfinite_json_rejected(provider, body):
    provider([response(body)])
    with pytest.raises(ValueError):
        _data_http.fetch_json(URL)


@pytest.mark.parametrize("status", [301, 401, 403, 429, 503])
def test_http_errors_do_not_echo_credentials_or_follow_redirects(provider, status):
    provider([f"HTTP/1.1 {status} Error\r\nLocation: https://example.com/secret\r\nContent-Length: 0\r\n\r\n".encode()])
    with pytest.raises(ValueError) as caught:
        _data_http.fetch_json(URL + "?key=private-credential")
    assert str(status) in str(caught.value) and "private-credential" not in str(caught.value)


def test_nasa_forbidden_query_explains_public_access_and_filter_recovery(provider):
    provider([b"HTTP/1.1 403 Forbidden\r\nServer: cloudflare\r\nContent-Length: 0\r\n\r\n"])
    with pytest.raises(ValueError) as caught:
        _data_http.fetch_json("https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query=private-query")
    message = str(caught.value)
    assert "403" in message and "no API key is required" in message
    assert "gateway" in message and "All methods" in message
    assert "private-query" not in message


def test_size_limits_cover_declared_and_streamed_bodies(provider):
    provider([response(b'{"value":123456789}')])
    with pytest.raises(ValueError, match="size limit"):
        _data_http.fetch_json(URL, max_bytes=5)
    provider([b"HTTP/1.0 200 OK\r\n\r\n{\"value\":123456789}"])
    with pytest.raises(ValueError, match="size limit"):
        _data_http.fetch_json(URL, max_bytes=5)


def test_valid_json_in_incomplete_http_body_is_rejected(provider):
    provider([response(b'{"value":1}', **{"Content-Length": "100"})])
    with pytest.raises(ValueError, match="declared length"):
        _data_http.fetch_json(URL)


def test_idle_headers_and_body_respect_cancellation_and_deadline(provider):
    provider([b"HTTP/1.1 200 OK\r\nContent-Length: 10\r\n\r\n"], idle=True)
    started = time.monotonic()
    with pytest.raises(InterruptedError):
        _data_http.fetch_json(URL, cancelled=lambda: time.monotonic() - started > .05)
    provider([], idle=True)
    with pytest.raises(TimeoutError):
        _data_http.fetch_json(URL, timeout=.05)


@pytest.mark.parametrize("url", ["http://earthquake.usgs.gov/", "https://example.com/", "https://user:pass@api.census.gov/",
                                  "https://api.census.gov:444/", "https://api.census.gov/#fragment"])
def test_only_fixed_public_https_hosts_are_accepted(url):
    with pytest.raises(ValueError, match="approved HTTPS"):
        _data_http.fetch_json(url)


def test_saved_request_url_strips_keys_and_tokens():
    url = _data_http.safe_url("https://api.census.gov/data/2024?get=NAME&key=secret&api_key=secret2&token=secret3")
    assert "secret" not in url and "get=NAME" in url
