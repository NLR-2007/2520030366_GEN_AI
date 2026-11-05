"""
GenAI Lab - Week 02
Terminal chat client with two interchangeable LLM providers:

    1. Ollama  -> local  (http://localhost:11434)
    2. Groq    -> cloud  (https://api.groq.com)

The user picks the provider and model at the terminal, then chats.
Conversation history is kept so follow-up questions have context.

Run:  python app.py
"""

import os
import sys
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR.parent / ".env")

GROQ_KEY = (os.getenv("GROK_API_KEY") or os.getenv("GROQ_API_KEY") or "").strip()
GROQ_BASE = "https://api.groq.com/openai/v1"
OLLAMA_BASE = os.getenv("OLLAMA_HOST", "http://localhost:11434").rstrip("/")

SYSTEM_PROMPT = "You are a concise, helpful assistant."
TIMEOUT = 120

# Groq hosts speech and safety models too; keep only the chat-capable ones.
NON_CHAT = ("whisper", "orpheus", "prompt-guard", "tts", "embed", "guard")


# ----------------------------------------------------------------- providers

def groq_models():
    if not GROQ_KEY:
        raise RuntimeError("No API key found. Add GROK_API_KEY to Lab/.env")
    res = requests.get(
        f"{GROQ_BASE}/models",
        headers={"Authorization": f"Bearer {GROQ_KEY}"},
        timeout=30,
    )
    if res.status_code == 401:
        raise RuntimeError("Groq rejected the API key (401). Check Lab/.env")
    res.raise_for_status()
    ids = [m["id"] for m in res.json().get("data", [])]
    return sorted(i for i in ids if not any(w in i.lower() for w in NON_CHAT))


def ollama_models():
    try:
        res = requests.get(f"{OLLAMA_BASE}/api/tags", timeout=10)
    except requests.RequestException:
        raise RuntimeError(
            f"Cannot reach Ollama at {OLLAMA_BASE}\n"
            "   Is it installed and running?  Start it with:  ollama serve"
        )
    res.raise_for_status()
    names = [m["name"] for m in res.json().get("models", [])]
    if not names:
        raise RuntimeError(
            "Ollama is running but has no models downloaded.\n"
            "   Pull one first, e.g.:  ollama pull llama3.2"
        )
    return sorted(names)


def groq_chat(model, messages):
    res = requests.post(
        f"{GROQ_BASE}/chat/completions",
        headers={"Authorization": f"Bearer {GROQ_KEY}"},
        json={"model": model, "messages": messages},
        timeout=TIMEOUT,
    )
    data = res.json()
    if not res.ok:
        err = data.get("error", data)
        raise RuntimeError(err.get("message") if isinstance(err, dict) else str(err))
    return data["choices"][0]["message"]["content"]


def ollama_chat(model, messages):
    res = requests.post(
        f"{OLLAMA_BASE}/api/chat",
        json={"model": model, "messages": messages, "stream": False},
        timeout=TIMEOUT,
    )
    data = res.json()
    if not res.ok:
        raise RuntimeError(data.get("error", f"Ollama returned {res.status_code}"))
    return data["message"]["content"]


PROVIDERS = {
    "1": {"name": "Ollama (local)", "list": ollama_models, "chat": ollama_chat},
    "2": {"name": "Groq (cloud)",   "list": groq_models,   "chat": groq_chat},
}


# --------------------------------------------------------------------- menus

def rule(char="-"):
    print(char * 58)


def ask(prompt):
    """Read a line; treat Ctrl+C / Ctrl+D as a clean quit."""
    try:
        return input(prompt).strip()
    except (EOFError, KeyboardInterrupt):
        print("\nBye.")
        sys.exit(0)


def choose_provider():
    while True:
        rule("=")
        print("  Choose a provider")
        rule("=")
        print("  1. Ollama  - runs locally on this machine")
        print("  2. Groq    - runs in the cloud")
        print("  q. Quit")
        choice = ask("\nEnter choice [1/2/q]: ").lower()

        if choice == "q":
            print("Bye.")
            sys.exit(0)
        if choice in PROVIDERS:
            return PROVIDERS[choice]
        print("\n  Please enter 1, 2 or q.\n")


def choose_model(provider):
    """Return a model name, or None to go back to the provider menu."""
    print(f"\nFetching models from {provider['name']} ...")
    try:
        models = provider["list"]()
    except Exception as exc:
        print(f"\n  {exc}\n")
        return None

    rule()
    print(f"  Models available on {provider['name']}")
    rule()
    for i, m in enumerate(models, 1):
        print(f"  {i:>2}. {m}")
    print("   b. Back")

    while True:
        choice = ask("\nPick a model by number: ").lower()
        if choice == "b":
            return None
        if choice.isdigit() and 1 <= int(choice) <= len(models):
            return models[int(choice) - 1]
        print(f"  Enter a number from 1 to {len(models)}, or b.")


# ---------------------------------------------------------------------- chat

def chat_loop(provider, model):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    rule("=")
    print(f"  {provider['name']}  |  {model}")
    rule("=")
    print("  Type your question and press Enter.")
    print("  /switch - change provider    /clear - reset history    /exit - quit")
    rule()

    while True:
        prompt = ask("\nYou > ")

        if not prompt:
            continue
        if prompt.lower() in ("/exit", "/quit"):
            print("Bye.")
            sys.exit(0)
        if prompt.lower() == "/switch":
            return
        if prompt.lower() == "/clear":
            messages = [{"role": "system", "content": SYSTEM_PROMPT}]
            print("  History cleared.")
            continue

        messages.append({"role": "user", "content": prompt})

        print("\n... thinking", end="\r")
        started = time.time()
        try:
            reply = provider["chat"](model, messages)
        except Exception as exc:
            print(" " * 20, end="\r")
            print(f"  Error: {exc}")
            messages.pop()          # drop the failed turn
            continue
        elapsed = time.time() - started

        print(" " * 20, end="\r")
        print(f"AI  > {reply}")
        print(f"      [{provider['name']} | {model} | {elapsed:.1f}s]")

        messages.append({"role": "assistant", "content": reply})


def main():
    print("\n  GenAI Lab - Week 02 : Local vs Cloud LLM\n")
    while True:
        provider = choose_provider()
        model = choose_model(provider)
        if model:
            chat_loop(provider, model)


if __name__ == "__main__":
    main()
