from __future__ import annotations

import importlib.util
import json
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("leo2_acceleration_lab_server", ROOT / "server.py")
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"could not load acceleration Lab server from {ROOT / 'server.py'}")
server = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(server)


class AccelerationLabServerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        media = self.root / "videos/example.mp4"
        media.parent.mkdir()
        media.write_bytes(b"0123456789")
        self.manifest = self.root / "release.json"
        self.manifest.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "release_id": "test-release",
                    "artifact_root": str(self.root),
                    "contract": {
                        "width": 848,
                        "height": 464,
                        "duration_seconds": 8,
                        "num_frames": 193,
                        "fps": 24,
                        "shift": 9,
                        "cp": 8,
                        "fsdp": 8,
                    },
                    "filters": {"steps": [6], "guidance": [5], "methods": ["off"]},
                    "metrics": [
                        {
                            "method": "off",
                            "steps": 6,
                            "guidance": 5,
                            "language": "en",
                            "latency_seconds": 1,
                            "paired_speedup": 1,
                        }
                    ],
                    "prompts": [
                        {
                            "index": 1,
                            "pair_id": "pair",
                            "language": "en",
                            "seed": 9,
                            "prompt": "Eight second prompt",
                            "videos": [
                                {
                                    "method": "off",
                                    "exact": True,
                                    "steps": 6,
                                    "guidance": 5,
                                    "path": "videos/example.mp4",
                                }
                            ],
                        }
                    ],
                    "figures": [],
                    "files": [
                        {
                            "path": "videos/example.mp4",
                            "sha256": "unused-in-server",
                            "bytes": 10,
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        server.Handler.store = server.ReleaseStore(self.manifest)
        server.Handler.static_dir = ROOT / "static"
        self.httpd = server.ThreadingHTTPServer(("127.0.0.1", 0), server.Handler)
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()
        self.base = f"http://127.0.0.1:{self.httpd.server_port}"

    def tearDown(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()
        self.temp.cleanup()

    def get_json(self, path: str) -> dict:
        with urllib.request.urlopen(self.base + path) as response:
            return json.load(response)

    def test_health_release_metrics_and_prompt_apis(self) -> None:
        self.assertEqual(
            self.get_json("/api/health"),
            {"status": "ok", "release_id": "test-release"},
        )
        self.assertEqual(self.get_json("/api/release")["contract"]["num_frames"], 193)
        self.assertEqual(
            self.get_json("/api/metrics?steps=6&guidance=5")["metrics"][0]["method"],
            "off",
        )
        self.assertEqual(self.get_json("/api/prompts?language=en")["total"], 1)
        self.assertEqual(self.get_json("/api/prompt/1")["seed"], 9)

    def test_media_range_request(self) -> None:
        request = urllib.request.Request(
            self.base + "/media/videos/example.mp4",
            headers={"Range": "bytes=2-5"},
        )
        with urllib.request.urlopen(request) as response:
            self.assertEqual(response.status, 206)
            self.assertEqual(response.headers["Content-Range"], "bytes 2-5/10")
            self.assertEqual(response.read(), b"2345")

    def test_unpublished_and_traversal_media_are_rejected(self) -> None:
        for path, status in (
            ("/media/videos/missing.mp4", 404),
            ("/media/%2E%2E/secret", 400),
        ):
            with self.assertRaises(urllib.error.HTTPError) as caught:
                urllib.request.urlopen(self.base + path)
            self.assertEqual(caught.exception.code, status)

    def test_static_page_is_served(self) -> None:
        with urllib.request.urlopen(self.base + "/") as response:
            body = response.read().decode()
        self.assertIn("Leo2 Acceleration Lab", body)


if __name__ == "__main__":
    unittest.main()
