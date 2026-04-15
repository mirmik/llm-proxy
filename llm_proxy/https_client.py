"""Minimal HTTPS client with TOFU fingerprint pinning."""

import hashlib
import http.client
import json
import ssl
from urllib.parse import urlsplit


def cert_fingerprint(der_bytes):
    digest = hashlib.sha256(der_bytes).hexdigest()
    return ":".join(digest[i:i + 2] for i in range(0, len(digest), 2))


def post_json(url, payload, *, timeout=30, headers=None, expected_fingerprint=None):
    return _json_request("POST", url, payload=payload, timeout=timeout, headers=headers, expected_fingerprint=expected_fingerprint)


def get_json(url, *, timeout=30, headers=None, expected_fingerprint=None):
    return _json_request("GET", url, payload=None, timeout=timeout, headers=headers, expected_fingerprint=expected_fingerprint)


def _json_request(method, url, *, payload, timeout, headers, expected_fingerprint):
    parsed = urlsplit(url)
    if parsed.scheme not in ("http", "https"):
        raise SystemExit(f"unsupported URL scheme: {parsed.scheme}")

    path = parsed.path or "/"
    if parsed.query:
        path += "?" + parsed.query

    req_headers = {"Accept": "application/json"}
    if headers:
        req_headers.update(headers)
    body = None
    if payload is not None:
        body = json.dumps(payload).encode("utf-8")
        req_headers["Content-Type"] = "application/json"

    conn = _make_connection(parsed, timeout)
    try:
        conn.request(method, path, body=body, headers=req_headers)
        response = conn.getresponse()
        fingerprint = None
        if parsed.scheme == "https":
            cert = conn.sock.getpeercert(binary_form=True) if conn.sock else None
            if not cert:
                raise SystemExit("server did not present a certificate")
            fingerprint = cert_fingerprint(cert)
            if expected_fingerprint and fingerprint != expected_fingerprint:
                raise SystemExit(
                    f"server fingerprint mismatch: expected {expected_fingerprint}, got {fingerprint}"
                )
        raw = response.read().decode("utf-8")
    finally:
        conn.close()

    if response.status >= 400:
        raise SystemExit(f"{response.status} {response.reason}: {raw}")

    data = json.loads(raw) if raw else {}
    if parsed.scheme == "https":
        data["_server_fingerprint"] = fingerprint
    return data


def _make_connection(parsed, timeout):
    host = parsed.hostname or "localhost"
    if parsed.scheme == "https":
        context = ssl._create_unverified_context()
        return http.client.HTTPSConnection(host, parsed.port or 443, timeout=timeout, context=context)
    return http.client.HTTPConnection(host, parsed.port or 80, timeout=timeout)
