#!/usr/bin/env python3
"""Read-only manifest server for the Leo2 acceleration Lab Hub page."""

from __future__ import annotations

import argparse
import json
import mimetypes
import re
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

RANGE_PATTERN = re.compile(r"^bytes=(\d*)-(\d*)$")


def _non_empty_text(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TypeError(f"expected non-empty string for {field}, got {type(value).__name__}: {value!r}")
    return value


def _safe_relative(value: Any, *, field: str) -> str:
    text = _non_empty_text(value, field=field)
    path = Path(text)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"expected safe relative path for {field}, got {text!r}")
    return path.as_posix()


class ReleaseStore:
    def __init__(self, manifest_path: Path):
        self.manifest_path = manifest_path.resolve()
        self.payload: dict[str, Any] = {}
        self.artifact_root = Path()
        self.allowed_files: set[str] = set()
        self._mtime_ns = -1
        self.refresh()

    def refresh(self) -> None:
        stat = self.manifest_path.stat()
        if stat.st_mtime_ns == self._mtime_ns:
            return
        payload = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or payload.get("schema_version") != 1:
            raise ValueError(
                f"expected schema_version=1 object in {self.manifest_path}, got {payload!r}"
            )
        _non_empty_text(payload.get("release_id"), field="release_id")
        artifact_root = Path(
            _non_empty_text(payload.get("artifact_root"), field="artifact_root")
        ).resolve()
        if not artifact_root.is_dir():
            raise FileNotFoundError(f"artifact root not found: {artifact_root}")
        for field in ("contract", "metrics", "prompts", "figures", "files"):
            expected = dict if field == "contract" else list
            if not isinstance(payload.get(field), expected):
                raise TypeError(
                    f"expected {expected.__name__} for manifest.{field}, "
                    f"got {type(payload.get(field)).__name__}"
                )
        allowed = {
            _safe_relative(item.get("path"), field=f"files[{index}].path")
            for index, item in enumerate(payload["files"])
            if isinstance(item, dict)
        }
        if len(allowed) != len(payload["files"]):
            raise ValueError("manifest files must be unique objects with safe relative paths")
        self.payload = payload
        self.artifact_root = artifact_root
        self.allowed_files = allowed
        self._mtime_ns = stat.st_mtime_ns

    def file_path(self, relative: str) -> Path:
        safe = _safe_relative(unquote(relative), field="media path")
        if safe not in self.allowed_files:
            raise FileNotFoundError(f"path is not published by the release manifest: {safe}")
        path = (self.artifact_root / safe).resolve()
        if self.artifact_root != path and self.artifact_root not in path.parents:
            raise PermissionError(f"media path escapes artifact root: {safe}")
        if not path.is_file():
            raise FileNotFoundError(f"published media is missing: {safe}")
        return path


class Handler(BaseHTTPRequestHandler):
    store: ReleaseStore
    static_dir: Path

    def _json(self, payload: Any, status: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _static(self, name: str) -> None:
        path = self.static_dir / name
        if not path.is_file():
            self._json({"error": f"static asset not found: {name}"}, HTTPStatus.NOT_FOUND)
            return
        body = path.read_bytes()
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "public, max-age=60")
        self.end_headers()
        self.wfile.write(body)

    def _media(self, relative: str) -> None:
        path = self.store.file_path(relative)
        size = path.stat().st_size
        start, end, status = 0, size - 1, HTTPStatus.OK
        raw_range = self.headers.get("Range")
        if raw_range:
            match = RANGE_PATTERN.fullmatch(raw_range.strip())
            if match is None:
                raise ValueError(f"unsupported Range header: {raw_range!r}")
            start_text, end_text = match.groups()
            if not start_text:
                suffix = int(end_text)
                if suffix <= 0:
                    raise ValueError(f"invalid suffix Range header: {raw_range!r}")
                start = max(size - suffix, 0)
            else:
                start = int(start_text)
            end = min(int(end_text), size - 1) if end_text and start_text else size - 1
            if start >= size or start > end:
                self.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                self.send_header("Content-Range", f"bytes */{size}")
                self.end_headers()
                return
            status = HTTPStatus.PARTIAL_CONTENT
        length = end - start + 1
        self.send_response(status)
        self.send_header("Content-Type", mimetypes.guess_type(path.name)[0] or "application/octet-stream")
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(length))
        if status == HTTPStatus.PARTIAL_CONTENT:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        with path.open("rb") as handle:
            handle.seek(start)
            remaining = length
            while remaining:
                chunk = handle.read(min(1024 * 1024, remaining))
                if not chunk:
                    raise OSError(f"media file ended before declared range: {path}")
                self.wfile.write(chunk)
                remaining -= len(chunk)

    @staticmethod
    def _single(params: dict[str, list[str]], name: str) -> str:
        values = params.get(name, [""])
        if len(values) != 1:
            raise ValueError(f"expected one value for {name}, got {values!r}")
        value = values[0].strip()
        if len(value) > 200:
            raise ValueError(f"{name} must be at most 200 characters")
        return value

    def _metrics(self, params: dict[str, list[str]]) -> list[dict[str, Any]]:
        allowed = {"steps", "guidance", "language", "method"}
        unknown = set(params) - allowed
        if unknown:
            raise ValueError(f"unsupported metrics filter(s): {', '.join(sorted(unknown))}")
        rows = self.store.payload["metrics"]
        filters = {name: self._single(params, name) for name in allowed}
        if filters["language"] not in {"", "en", "zh"}:
            raise ValueError(f"unsupported language: {filters['language']!r}")
        try:
            guidance_filter = (
                float(filters["guidance"]) if filters["guidance"] else None
            )
        except ValueError as exc:
            raise ValueError(
                f"guidance must be numeric, got {filters['guidance']!r}"
            ) from exc
        result = []
        for row in rows:
            if filters["steps"] and str(row.get("steps")) != filters["steps"]:
                continue
            if guidance_filter is not None and float(row.get("guidance")) != guidance_filter:
                continue
            if filters["language"] and row.get("language") != filters["language"]:
                continue
            if filters["method"] and row.get("method") != filters["method"]:
                continue
            result.append(row)
        return result

    def _prompts(self, params: dict[str, list[str]]) -> dict[str, Any]:
        allowed = {"language", "query", "page", "page_size", "sort"}
        unknown = set(params) - allowed
        if unknown:
            raise ValueError(f"unsupported prompt filter(s): {', '.join(sorted(unknown))}")
        language = self._single(params, "language")
        if language not in {"", "en", "zh"}:
            raise ValueError(f"unsupported language: {language!r}")
        query = self._single(params, "query").casefold()
        sort = self._single(params, "sort") or "index"
        if sort not in {"index", "worst"}:
            raise ValueError(f"unsupported prompt sort: {sort!r}")
        try:
            page = int(self._single(params, "page") or "1")
            page_size = int(self._single(params, "page_size") or "24")
        except ValueError as exc:
            raise ValueError("page and page_size must be integers") from exc
        if page < 1 or not 1 <= page_size <= 100:
            raise ValueError(f"expected page>=1 and page_size in [1,100], got {page}/{page_size}")
        rows = [
            row
            for row in self.store.payload["prompts"]
            if (not language or row.get("language") == language)
            and (
                not query
                or query in str(row.get("prompt", "")).casefold()
                or query in str(row.get("pair_id", "")).casefold()
            )
        ]
        rows.sort(
            key=(
                (lambda row: (-float(row.get("worst_score") or 0), int(row["index"])))
                if sort == "worst"
                else (lambda row: int(row["index"]))
            )
        )
        start = (page - 1) * page_size
        return {"total": len(rows), "page": page, "page_size": page_size, "prompts": rows[start : start + page_size]}

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        try:
            self.store.refresh()
            if parsed.path in {"/", "/index.html"}:
                self._static("index.html")
                return
            if parsed.path in {"/app.js", "/style.css"}:
                self._static(parsed.path.lstrip("/"))
                return
            if parsed.path == "/api/health":
                self._json({"status": "ok", "release_id": self.store.payload["release_id"]})
                return
            if parsed.path == "/api/release":
                self._json(
                    {
                        key: value
                        for key, value in self.store.payload.items()
                        if key not in {"prompts", "files"}
                    }
                )
                return
            params = parse_qs(parsed.query, keep_blank_values=True)
            if parsed.path == "/api/metrics":
                self._json({"metrics": self._metrics(params)})
                return
            if parsed.path == "/api/prompts":
                self._json(self._prompts(params))
                return
            if parsed.path.startswith("/api/prompt/"):
                index = int(parsed.path.rsplit("/", 1)[1])
                unknown = set(params) - {"language"}
                if unknown:
                    raise ValueError(
                        f"unsupported prompt parameter(s): {', '.join(sorted(unknown))}"
                    )
                language = self._single(params, "language")
                if language not in {"", "en", "zh"}:
                    raise ValueError(f"unsupported language: {language!r}")
                matches = [
                    row
                    for row in self.store.payload["prompts"]
                    if int(row.get("index", -1)) == index
                    and (not language or row.get("language") == language)
                ]
                if len(matches) != 1:
                    self._json({"error": f"expected one prompt for index={index}, language={language!r}"}, HTTPStatus.NOT_FOUND)
                    return
                self._json(matches[0])
                return
            if parsed.path.startswith("/media/"):
                self._media(parsed.path[len("/media/") :])
                return
            self._json({"error": f"route not found: {parsed.path}"}, HTTPStatus.NOT_FOUND)
        except PermissionError as exc:
            self._json({"error": str(exc)}, HTTPStatus.FORBIDDEN)
        except FileNotFoundError as exc:
            self._json({"error": str(exc)}, HTTPStatus.NOT_FOUND)
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            self._json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
        except OSError as exc:
            self._json(
                {"error": f"service unavailable: {type(exc).__name__}: {exc}"},
                HTTPStatus.SERVICE_UNAVAILABLE,
            )

    def log_message(self, format_string: str, *args: Any) -> None:
        print(f"{self.address_string()} - {format_string % args}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bind", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=16023)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        raise ValueError(f"expected port in [1,65535], got {args.port}")
    Handler.store = ReleaseStore(args.manifest)
    Handler.static_dir = Path(__file__).parent / "static"
    server = ThreadingHTTPServer((args.bind, args.port), Handler)
    print(
        f"Leo2 Acceleration Lab listening on {args.bind}:{args.port}; "
        f"release={Handler.store.payload['release_id']}",
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
