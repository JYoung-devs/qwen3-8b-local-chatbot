"""API contract tests: stub the model, so tests never need CUDA or model weights."""

import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
model_stub = types.ModuleType("run_qwen")
model_stub.load_chatbot = lambda: ("test-tokenizer", "test-model")


def unexpected_generation(*_args, **_kwargs):
    raise AssertionError("Model generation must be mocked explicitly in tests")


model_stub.generate_response = unexpected_generation
sys.modules["run_qwen"] = model_stub

from fastapi.testclient import TestClient  # noqa: E402
import server  # noqa: E402


class ChatApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(server.app)
        cls.client.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls.client.__exit__(None, None, None)

    def test_one_turn_returns_model_reply_and_system_prompt(self):
        body = {"messages": [{"role": "user", "content": "Hello"}]}
        with patch.object(server, "generate_response", return_value="Hi") as mock_generate:
            response = self.client.post("/chat", json=body)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"response": "Hi"})
        mock_generate.assert_called_once()
        tokenizer, model, messages = mock_generate.call_args.args
        self.assertEqual((tokenizer, model), ("test-tokenizer", "test-model"))
        self.assertEqual(messages[0], {"role": "system", "content": server.SYSTEM_PROMPT})
        self.assertEqual(messages[1:], body["messages"])

    def test_multi_turn_history_is_forwarded_in_order(self):
        history = [
            {"role": "user", "content": "First"},
            {"role": "assistant", "content": "Previous answer"},
            {"role": "user", "content": "Follow-up"},
        ]
        with patch.object(server, "generate_response", return_value="Next") as mock_generate:
            response = self.client.post("/chat", json={"messages": history})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(mock_generate.call_args.args[2][1:], history)

    def test_invalid_conversations_are_rejected_before_inference(self):
        cases = {
            "empty": {"messages": []},
            "blank": {"messages": [{"role": "user", "content": "   "}]},
            "assistant_first": {"messages": [{"role": "assistant", "content": "Hello"}]},
            "repeated_user": {"messages": [
                {"role": "user", "content": "One"},
                {"role": "user", "content": "Two"},
            ]},
            "assistant_last": {"messages": [
                {"role": "user", "content": "One"},
                {"role": "assistant", "content": "Two"},
            ]},
            "unknown_role": {"messages": [{"role": "system", "content": "Hello"}]},
        }
        with patch.object(server, "generate_response") as mock_generate:
            for name, payload in cases.items():
                with self.subTest(name=name):
                    response = self.client.post("/chat", json=payload)
                    self.assertEqual(response.status_code, 422)
            mock_generate.assert_not_called()

    def test_cors_only_allows_configured_local_frontend(self):
        headers = {
            "Origin": "http://127.0.0.1:5500",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type",
        }
        allowed = self.client.options("/chat", headers=headers)
        self.assertEqual(allowed.status_code, 200)
        self.assertEqual(allowed.headers.get("access-control-allow-origin"), headers["Origin"])

        blocked = self.client.options("/chat", headers={**headers, "Origin": "https://example.com"})
        self.assertNotEqual(blocked.status_code, 200)
        self.assertIsNone(blocked.headers.get("access-control-allow-origin"))


if __name__ == "__main__":
    unittest.main()
