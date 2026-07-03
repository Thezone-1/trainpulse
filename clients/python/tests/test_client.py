import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer

from trainpulse import TrainPulseClient, TrainPulseConnectionError, Tuner


class FakeDaemon(BaseHTTPRequestHandler):
    requests = []

    def _respond(self, code, payload):
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        FakeDaemon.requests.append(
            {
                "path": self.path,
                "auth": self.headers.get("Authorization"),
                "body": json.loads(self.rfile.read(length) or b"{}"),
            }
        )
        self._respond(202, {"accepted": True})

    def do_GET(self):
        FakeDaemon.requests.append({"path": self.path, "auth": self.headers.get("Authorization")})
        if self.path == "/v1/snapshot":
            self._respond(200, {"health": 88.0, "status": "info"})
        elif self.path == "/v1/recommendations":
            self._respond(
                200,
                {
                    "recommendations": [
                        {
                            "id": "increase_dataloader_workers",
                            "parameter": "dataloader_workers",
                            "suggested": "8",
                            "auto_applicable": True,
                        },
                        {
                            "id": "grow_micro_batch",
                            "parameter": "micro_batch_size",
                            "suggested": "16",
                            "auto_applicable": False,
                        },
                    ]
                },
            )
        else:
            self._respond(200, {"version": "test"})

    def log_message(self, *args):
        pass


class ClientTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = HTTPServer(("127.0.0.1", 0), FakeDaemon)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.url = f"http://127.0.0.1:{cls.server.server_address[1]}"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()

    def setUp(self):
        FakeDaemon.requests = []
        self.client = TrainPulseClient(self.url, token="tok")

    def test_report_posts_metrics_with_auth(self):
        ok = self.client.report(step_time_ms=123.4, tokens_per_sec=50000, global_step=7)
        self.assertTrue(ok)
        req = FakeDaemon.requests[0]
        self.assertEqual(req["path"], "/v1/training")
        self.assertEqual(req["auth"], "Bearer tok")
        self.assertEqual(req["body"]["global_step"], 7)

    def test_step_context_manager_times_and_reports(self):
        with self.client.step(global_step=3, tokens=1000):
            pass
        body = FakeDaemon.requests[0]["body"]
        self.assertEqual(body["global_step"], 3)
        self.assertGreater(body["step_time_ms"], 0)
        self.assertGreater(body["tokens_per_sec"], 0)

    def test_health_reads_snapshot(self):
        self.assertEqual(self.client.health(), 88.0)

    def test_tuner_applies_only_auto_applicable(self):
        tuner = Tuner(self.client, refresh_seconds=0)
        self.assertEqual(tuner.suggested("dataloader_workers", current=2), 8)
        # micro_batch_size is not auto-applicable: current value must survive.
        self.assertEqual(tuner.suggested("micro_batch_size", current=4), 4)
        advisory_ids = [r["id"] for r in tuner.advisories()]
        self.assertEqual(advisory_ids, ["grow_micro_batch"])

    def test_recommendation_for_respects_auto_only(self):
        tuner = Tuner(self.client, refresh_seconds=0)
        rec = tuner.recommendation_for("dataloader_workers")
        self.assertEqual(rec["id"], "increase_dataloader_workers")
        self.assertIsNone(tuner.recommendation_for("micro_batch_size"))
        rec = tuner.recommendation_for("micro_batch_size", auto_only=False)
        self.assertEqual(rec["id"], "grow_micro_batch")

    def test_report_fail_silently_when_daemon_down(self):
        dead = TrainPulseClient("http://127.0.0.1:9", timeout=0.2)
        self.assertFalse(dead.report(step_time_ms=1))  # no exception

    def test_read_raises_when_daemon_down(self):
        dead = TrainPulseClient("http://127.0.0.1:9", timeout=0.2)
        with self.assertRaises(TrainPulseConnectionError):
            dead.snapshot()


if __name__ == "__main__":
    unittest.main()
