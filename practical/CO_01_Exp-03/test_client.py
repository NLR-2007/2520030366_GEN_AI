"""
GenAI Lab - Week 03
Tests LLMClient with a fake provider and a fake logger, so no real API
call or MongoDB connection is made.
"""

import unittest

from llm_client import LLMClient, LLMResult, ProviderError


class FakeProvider:
    provider_name = "Test"
    model = "fake-model"

    def generate(self, prompt):
        return LLMResult(
            text="AI stands for Artificial Intelligence.",
            input_tokens=5,
            output_tokens=8,
        )


class FlakyProvider:
    """Fails with a retryable 503 once, then succeeds."""

    provider_name = "Test"
    model = "fake-model"

    def __init__(self):
        self.calls = 0

    def generate(self, prompt):
        self.calls += 1
        if self.calls == 1:
            raise ProviderError("temporarily unavailable", 503)
        return LLMResult(text="ok", input_tokens=1, output_tokens=2)


class BrokenProvider:
    """Fails with a permanent 401, which must not be retried."""

    provider_name = "Test"
    model = "fake-model"

    def __init__(self):
        self.calls = 0

    def generate(self, prompt):
        self.calls += 1
        raise ProviderError("invalid api key", 401)


class FakeLogger:
    def __init__(self):
        self.records = []

    def log_call(self, record):
        self.records.append(record)
        return True


class TestLLMClient(unittest.TestCase):
    def test_generate_and_log_usage(self):
        logger = FakeLogger()
        client = LLMClient(FakeProvider(), logger)

        answer = client.generate("Explain AI")

        self.assertEqual(answer, "AI stands for Artificial Intelligence.")
        self.assertEqual(len(logger.records), 1)

        record = logger.records[0]
        self.assertEqual(record["total_tokens"], 13)
        self.assertEqual(record["status"], "success")
        self.assertEqual(record["attempts"], 1)
        self.assertIsInstance(record["latency_ms"], float)

    def test_retries_temporary_failure(self):
        logger = FakeLogger()
        provider = FlakyProvider()
        client = LLMClient(provider, logger)

        answer = client.generate("hello")

        self.assertEqual(answer, "ok")
        self.assertEqual(provider.calls, 2)
        self.assertEqual(logger.records[0]["status"], "success")
        self.assertEqual(logger.records[0]["attempts"], 2)

    def test_permanent_failure_is_logged_and_raised(self):
        logger = FakeLogger()
        provider = BrokenProvider()
        client = LLMClient(provider, logger)

        with self.assertRaises(ProviderError):
            client.generate("hello")

        self.assertEqual(provider.calls, 1)          # no retry on 401
        self.assertEqual(logger.records[0]["status"], "failed")
        self.assertIsNone(logger.records[0]["total_tokens"])
        self.assertIn("invalid api key", logger.records[0]["error"])


if __name__ == "__main__":
    unittest.main()
