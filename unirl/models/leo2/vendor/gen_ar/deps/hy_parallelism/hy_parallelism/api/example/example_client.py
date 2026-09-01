"""
HTTP client for the demo server started by :mod:`hy_parallelism.api.example.example_run`.

Dependencies: none beyond the stdlib (uses :mod:`urllib`).

Usage (server must be running, default ``http://127.0.0.1:8000``)::

    python -m hy_parallelism.api.example.example_client
    python -m hy_parallelism.api.example.example_client --base-url http://127.0.0.1:8000 --payload '{"hello": "world"}'
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request


def _request_json(method: str, url: str, body: dict | None = None, *, timeout_s: float = 60.0) -> dict:
    data = None if body is None else json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        method=method,
        headers={"Content-Type": "application/json"} if body is not None else {},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            raw = resp.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace")
        raise SystemExit(f"HTTP {e.code} {e.reason}: {detail}") from e
    except urllib.error.URLError as e:
        raise SystemExit(f"request failed: {e.reason}") from e
    return json.loads(raw) if raw else {}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-url", default="http://127.0.0.1:8000", help="Ray Serve HTTP root (no trailing slash)")
    p.add_argument(
        "--payload",
        default='{"demo": true}',
        help='JSON object for POST /generate (default: {"demo": true})',
    )
    p.add_argument("--timeout", type=float, default=60.0, help="per-request timeout seconds")
    args = p.parse_args()
    base = args.base_url.rstrip("/")

    health = _request_json("GET", f"{base}/health", timeout_s=args.timeout)
    print("GET /health ->", json.dumps(health, indent=2))

    try:
        payload_obj = json.loads(args.payload)
    except json.JSONDecodeError as e:
        print("--payload must be valid JSON object", file=sys.stderr)
        raise SystemExit(2) from e
    if not isinstance(payload_obj, dict):
        raise SystemExit("--payload must be a JSON object")

    gen = _request_json(
        "POST",
        f"{base}/generate",
        {"payload": payload_obj},
        timeout_s=args.timeout,
    )
    print("POST /generate ->", json.dumps(gen, indent=2))


if __name__ == "__main__":
    main()
