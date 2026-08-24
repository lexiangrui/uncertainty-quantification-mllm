#!/usr/bin/env python3
from __future__ import annotations

import argparse
import fcntl
import json
import mimetypes
import sys
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.human_alignment import load_annotations, save_annotations  # noqa: E402


HTML_PATH = Path(__file__).with_name("index.html")


def make_handler(workspace: Path):
    write_lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def _json(self, value, status: int = HTTPStatus.OK) -> None:
            payload = json.dumps(value, ensure_ascii=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def _queue(self) -> dict:
            return json.loads((workspace / "samples.json").read_text(encoding="utf-8"))

        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            if parsed.path == "/":
                payload = HTML_PATH.read_bytes()
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
                return
            if parsed.path == "/api/state":
                queue = self._queue()
                self._json({"schema_version": queue["schema_version"], "blind": True, "counts": queue["counts"], "samples": queue["samples"], "annotations": load_annotations(workspace)["annotations"]})
                return
            if parsed.path == "/api/image":
                key = parse_qs(parsed.query).get("key", [""])[0]
                row = next((item for item in self._queue()["samples"] if item["key"] == key), None)
                if row is None or not row.get("image"):
                    self.send_error(HTTPStatus.NOT_FOUND)
                    return
                root = workspace.resolve()
                target = (workspace / row["image"]).resolve()
                if root not in target.parents or not target.is_file():
                    self.send_error(HTTPStatus.NOT_FOUND)
                    return
                payload = target.read_bytes()
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", mimetypes.guess_type(target.name)[0] or "application/octet-stream")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
                return
            self.send_error(HTTPStatus.NOT_FOUND)

        def do_POST(self) -> None:  # noqa: N802
            if urlparse(self.path).path != "/api/save":
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length > 64 * 1024:
                    raise ValueError("request too large")
                value = json.loads(self.rfile.read(length))
                key = value.get("key")
                keys = {row["key"] for row in self._queue()["samples"]}
                if key not in keys:
                    raise ValueError("unknown sample key")
                with write_lock:
                    lock_path = workspace / ".annotations.lock"
                    with lock_path.open("a+") as lock_handle:
                        fcntl.flock(lock_handle, fcntl.LOCK_EX)
                        state = load_annotations(workspace)
                        state["annotations"][key] = value.get("annotation")
                        save_annotations(workspace, state["annotations"])
                        saved = load_annotations(workspace)["annotations"][key]
                        fcntl.flock(lock_handle, fcntl.LOCK_UN)
                self._json({"ok": True, "annotation": saved})
            except (ValueError, TypeError, json.JSONDecodeError) as error:
                self._json({"ok": False, "error": str(error)}, HTTPStatus.BAD_REQUEST)

        def log_message(self, format: str, *args) -> None:
            print(f"[{self.log_date_time_string()}] {format % args}")

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve the local blind human-adjudication UI.")
    parser.add_argument("--workspace", type=Path, default=PROJECT_ROOT / "results/human_alignment")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    if not (args.workspace / "samples.json").is_file():
        raise FileNotFoundError(f"run prepare.py first: {args.workspace / 'samples.json'}")
    queue = json.loads((args.workspace / "samples.json").read_text(encoding="utf-8"))
    unavailable = [
        row["key"]
        for row in queue.get("samples", [])
        if row.get("image_status") not in {"available", "not_applicable"}
    ]
    if unavailable:
        raise ValueError(
            f"alignment images are not ready for {len(unavailable)} samples; "
            "rerun prepare.py without --skip-images"
        )
    server = ThreadingHTTPServer((args.host, args.port), make_handler(args.workspace))
    print(f"Human alignment UI: http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
