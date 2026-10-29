"""Bounded HTTPS reads for the five explicit Data playground sources.

This is intentionally independent of the local-inference loopback transport.
There are no arbitrary hosts, redirects, proxies or credential-bearing errors.
"""

from __future__ import annotations

import http.client
import io
import json
import select
import ssl
from time import monotonic
from types import SimpleNamespace
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

ALLOWED_HOSTS = frozenset({
    "earthquake.usgs.gov", "world.openfoodfacts.org", "exoplanetarchive.ipac.caltech.edu",
    "api.census.gov", "archive-api.open-meteo.com",
})
MAX_BYTES = 12 * 1024 * 1024


def safe_url(url: str) -> str:
    """Remove API credentials before retaining request provenance."""
    parsed = urlsplit(url)
    query = [(key, value) for key, value in parse_qsl(parsed.query, keep_blank_values=True)
             if key.lower() not in {"key", "api_key", "token", "access_token"}]
    return urlunsplit((parsed.scheme, parsed.netloc, parsed.path, urlencode(query), ""))


def check_cancelled(cancelled=None, deadline=None) -> None:
    if cancelled is not None and cancelled():
        raise InterruptedError("Data fetch cancelled; the previous dataset is preserved.")
    if deadline is not None and monotonic() >= deadline:
        raise TimeoutError("Data fetch reached its time limit; try a smaller query.")


class _TLSReader(io.RawIOBase):
    """Poll TLS reads so cancellation also works while a server is idle."""

    def __init__(self, transport, cancelled, deadline):
        self.transport, self.cancelled, self.deadline = transport, cancelled, deadline

    def readable(self):
        return True

    def readinto(self, buffer):
        want_write = False
        while True:
            check_cancelled(self.cancelled, self.deadline)
            try:
                return self.transport.recv_into(buffer)
            except (ssl.SSLWantReadError, BlockingIOError):
                want_write = False
            except ssl.SSLWantWriteError:
                want_write = True
            select.select([] if want_write else [self.transport], [self.transport] if want_write else [], [], 0.05)


def _unique_object(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("Data source returned duplicate JSON fields.")
        value[key] = item
    return value


def _nonfinite(_value):
    raise ValueError("Data source returned non-finite JSON numbers.")


def fetch_json(url: str, *, cancelled=None, timeout: float = 30.0, max_bytes: int = MAX_BYTES):
    """Fetch strict UTF-8 JSON from an allowlisted HTTPS source, without redirects.

    Connection/TLS setup is bounded to five seconds; the response, including
    headers and chunked reads, remains cancellable until its overall deadline.
    """
    parsed = urlsplit(url)
    try:
        valid = (parsed.scheme == "https" and parsed.hostname in ALLOWED_HOSTS and parsed.port in (None, 443)
                 and not parsed.username and not parsed.password and not parsed.fragment
                 and parsed.path.startswith("/"))
    except ValueError:
        valid = False
    if not valid:
        raise ValueError("Data playground requests must use an approved HTTPS data source.")
    if not 0 < timeout <= 120 or not 0 < max_bytes <= MAX_BYTES:
        raise ValueError("Invalid data request bounds.")
    check_cancelled(cancelled)
    deadline = monotonic() + timeout
    connection = http.client.HTTPSConnection(parsed.hostname, timeout=min(5.0, timeout))
    response = None
    try:
        path = parsed.path + ("?" + parsed.query if parsed.query else "")
        connection.request("GET", path, headers={
            "Accept": "application/json", "Accept-Encoding": "identity", "Connection": "close",
            "User-Agent": "MLcookbook/1.0 (https://github.com/aadabathon/MLcookbook)",
        })
        connection.sock.setblocking(False)
        reader = io.BufferedReader(_TLSReader(connection.sock, cancelled, deadline))
        # Keep ownership of the TLS socket until the body has been read.
        # HTTPConnection.getresponse() closes a Connection: close socket after
        # its headers, relying on socket.makefile() references. Our polling
        # reader is not such a reference, and SSLSocket cannot be duplicated.
        response = http.client.HTTPResponse(SimpleNamespace(makefile=lambda mode: reader))
        response.begin()
        if response.status == 204:
            return None
        if response.status != 200:
            if response.status in (401, 403):
                detail = "Access was refused; check source availability and any required API key."
            elif response.status == 429:
                detail = "The source rate limit was reached; wait before fetching again."
            elif 300 <= response.status < 400:
                detail = "The source redirected the request; redirects are not followed."
            else:
                detail = "Try again later or use a smaller query."
            raise ValueError(f"Data source returned HTTP {response.status}. {detail}")
        length = response.getheader("Content-Length")
        if length is not None:
            try:
                declared = int(length)
            except ValueError:
                raise ValueError("Data source returned an invalid response length.") from None
            if declared < 0 or declared > max_bytes:
                raise ValueError("Data source response exceeds the size limit; reduce the query.")
        encoding = response.getheader("Content-Encoding", "identity").lower()
        if encoding not in ("", "identity"):
            raise ValueError("Data source returned unsupported compressed data.")
        chunks, total = [], 0
        while True:
            check_cancelled(cancelled, deadline)
            block = response.read(min(65536, max_bytes + 1 - total))
            if not block:
                break
            chunks.append(block)
            total += len(block)
            if total > max_bytes:
                raise ValueError("Data source response exceeds the size limit; reduce the query.")
        if length is not None and total != declared:
            raise ValueError("Data source response ended before its declared length; fetch again.")
        check_cancelled(cancelled, deadline)
        try:
            return json.loads(b"".join(chunks).decode("utf-8-sig"), object_pairs_hook=_unique_object,
                              parse_constant=_nonfinite)
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError("Data source returned invalid UTF-8 JSON; try again later.") from exc
    except (OSError, http.client.HTTPException) as exc:
        check_cancelled(cancelled, deadline)
        # Do not interpolate provider exceptions: they may contain an API key.
        raise ConnectionError(f"Cannot read {parsed.hostname}; check the internet connection and try again.") from exc
    finally:
        if response is not None:
            response.close()
        connection.close()
