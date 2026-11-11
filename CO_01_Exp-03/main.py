"""
GenAI Lab - Week 03
Ask a question through Groq (cloud) or Ollama (local); every call is
logged to MongoDB with tokens, latency and estimated cost.

Run:  python main.py
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from llm_client import GroqProvider, OllamaProvider, LLMClient
from usage_logger import UsageLogger

# The Windows console defaults to cp1252 and crashes on characters the
# model may return (dashes, quotes, emoji). UTF-8 keeps the output safe.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, OSError):
    pass

BASE_DIR = Path(__file__).resolve().parent

# Lab/.env holds the shared API key; Week-03/.env holds MONGODB_URI and
# the price settings. The local file wins where both define a name.
load_dotenv(BASE_DIR.parent / ".env", override=True)
load_dotenv(BASE_DIR / ".env", override=True)


def main():
    usage_logger = UsageLogger()
    try:
        provider_name = input("Choose provider (groq/ollama): ").strip().lower()

        if provider_name == "groq":
            api_key = (
                os.getenv("GROQ_API_KEY") or os.getenv("GROK_API_KEY") or ""
            ).strip()
            if not api_key:
                raise ValueError("GROQ_API_KEY not found in .env")
            provider = GroqProvider(api_key)
        elif provider_name == "ollama":
            provider = OllamaProvider(os.getenv("OLLAMA_MODEL", "llama3.2"))
        else:
            raise ValueError("Invalid provider selected")

        client = LLMClient(provider, usage_logger)

        prompt = input("Enter your question: ")
        answer = client.generate(prompt)

        print("\nLLM Response:")
        print(answer)
    finally:
        usage_logger.close()


if __name__ == "__main__":
    main()
