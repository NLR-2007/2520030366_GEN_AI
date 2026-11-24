"""
GenAI Lab - Week 03
Provider-agnostic LLM client with usage, latency and cost monitoring.

    GroqProvider   -> cloud  (https://api.groq.com, OpenAI-compatible API)
    OllamaProvider -> local  (http://localhost:11434)

LLMClient measures latency, estimates cost, logs every call to MongoDB,
and retries temporary failures up to three times.
"""

import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass

import requests

GROQ_BASE = "https://api.groq.com/openai/v1"
DEFAULT_GROQ_MODEL = "openai/gpt-oss-120b"
DEFAULT_OLLAMA_MODEL = "llama3.2"

RETRYABLE_STATUS = (408, 409, 429, 500, 502, 503, 504)


@dataclass
class LLMResult:
    text: str
    input_tokens: int
    output_tokens: int


class ProviderError(RuntimeError):
    """An API error that carries the HTTP status code, so the client
    can tell a temporary failure from a permanent one."""

    def __init__(self, message, status_code=None):
        super().__init__(message)
        self.status_code = status_code


class LLMProvider(ABC):
    @abstractmethod
    def generate(self, prompt):
        pass


# ------------------------------------------------------------ hosted (Groq)

class GroqProvider(LLMProvider):
    provider_name = "Groq"

    def __init__(self, api_key, model=None):
        if not api_key:
            raise ValueError("GROQ_API_KEY not found in .env")
        self.api_key = api_key
        self.model = model or os.getenv("GROQ_MODEL") or DEFAULT_GROQ_MODEL

    def generate(self, prompt):
        try:
            response = requests.post(
                f"{GROQ_BASE}/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                },
                timeout=(5, 60),   # 5 s to connect, 60 s to read the answer
            )
        except requests.RequestException as error:
            raise ProviderError(f"Could not reach Groq: {error}") from error

        try:
            data = response.json()
        except ValueError:
            raise ProviderError(
                f"Groq returned a non-JSON response ({response.status_code})",
                response.status_code,
            )

        if not response.ok:
            detail = data.get("error", data)
            message = detail.get("message") if isinstance(detail, dict) else str(detail)
            raise ProviderError(f"Groq error: {message}", response.status_code)

        usage = data.get("usage") or {}
        return LLMResult(
            text=data["choices"][0]["message"]["content"] or "",
            input_tokens=usage.get("prompt_tokens", 0) or 0,
            output_tokens=usage.get("completion_tokens", 0) or 0,
        )


# ------------------------------------------------------------ local (Ollama)

class OllamaProvider(LLMProvider):
    provider_name = "Ollama"

    def __init__(self, model=DEFAULT_OLLAMA_MODEL):
        self.model = model
        base = os.getenv("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
        self.url = f"{base}/api/generate"

    def generate(self, prompt):
        response = requests.post(
            self.url,
            json={"model": self.model, "prompt": prompt, "stream": False},
            timeout=(5, 60),
        )
        response.raise_for_status()
        data = response.json()
        return LLMResult(
            text=data.get("response", ""),
            input_tokens=data.get("prompt_eval_count", 0) or 0,
            output_tokens=data.get("eval_count", 0) or 0,
        )


# ------------------------------------------------------------- common client

class LLMClient:
    def __init__(self, provider, usage_logger, max_attempts=3):
        self.provider = provider
        self.usage_logger = usage_logger
        self.max_attempts = max_attempts

    def _estimate_cost(self, result):
        """Hosted token cost in USD. Local inference is recorded as 0."""
        if self.provider.provider_name != "Groq":
            return 0.0, "local_inference_cost_not_included"

        input_rate = float(os.getenv("GROQ_INPUT_PRICE_PER_MILLION", "0"))
        output_rate = float(os.getenv("GROQ_OUTPUT_PRICE_PER_MILLION", "0"))
        cost = (
            result.input_tokens * input_rate
            + result.output_tokens * output_rate
        ) / 1_000_000
        return cost, "estimated_hosted_api_cost"

    @staticmethod
    def _is_retryable(error):
        status = getattr(error, "status_code", None)
        if status is None:
            response = getattr(error, "response", None)
            status = getattr(response, "status_code", None)

        name = type(error).__name__.lower()
        return (
            isinstance(error, (requests.exceptions.Timeout,
                               requests.exceptions.ConnectionError))
            or status in RETRYABLE_STATUS
            or "timeout" in name
            or "unavailable" in name
        )

    def generate(self, prompt):
        start = time.perf_counter()

        for attempt in range(self.max_attempts):
            try:
                result = self.provider.generate(prompt)
            except Exception as error:
                last_attempt = attempt == self.max_attempts - 1
                if not self._is_retryable(error) or last_attempt:
                    latency_ms = round((time.perf_counter() - start) * 1000, 2)
                    self.usage_logger.log_call({
                        "provider": self.provider.provider_name,
                        "model": self.provider.model,
                        "input_tokens": None,
                        "output_tokens": None,
                        "total_tokens": None,
                        "latency_ms": latency_ms,
                        "estimated_cost_usd": None,
                        "status": "failed",
                        "attempts": attempt + 1,
                        "error": str(error),
                    })
                    raise

                delay = 2 ** attempt
                print(
                    f"Temporary error. Retry {attempt + 1} "
                    f"of {self.max_attempts - 1} in {delay} seconds."
                )
                time.sleep(delay)
                continue

            latency_ms = round((time.perf_counter() - start) * 1000, 2)
            cost, cost_type = self._estimate_cost(result)

            self.usage_logger.log_call({
                "provider": self.provider.provider_name,
                "model": self.provider.model,
                "input_tokens": result.input_tokens,
                "output_tokens": result.output_tokens,
                "total_tokens": result.input_tokens + result.output_tokens,
                "latency_ms": latency_ms,
                "estimated_cost_usd": round(cost, 8),
                "cost_type": cost_type,
                "status": "success",
                "attempts": attempt + 1,
            })

            print(f"Latency: {latency_ms} ms")
            print(f"Input tokens: {result.input_tokens}")
            print(f"Output tokens: {result.output_tokens}")
            print(f"Estimated cost (USD): {cost}")
            return result.text
