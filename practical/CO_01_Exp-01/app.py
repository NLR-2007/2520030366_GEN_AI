"""
GenAI Lab - Week 01
Tiny Flask backend that keeps the Groq API key on the server and proxies
chat requests from the browser to the Groq chat-completions API.

Run:  python app.py      ->  http://127.0.0.1:5000
"""

import os
from pathlib import Path

import requests
from dotenv import load_dotenv
from flask import Flask, jsonify, request, send_from_directory

# .env lives one level up, in practical/
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR.parent / ".env")

API_KEY = (os.getenv("GROK_API_KEY") or "").strip()
API_URL = "https://api.groq.com/openai/v1/chat/completions"
DEFAULT_MODEL = "openai/gpt-oss-120b"

app = Flask(__name__, static_folder=None)


@app.get("/")
def index():
    return send_from_directory(BASE_DIR, "index.html")


@app.post("/api/chat")
def chat():
    if not API_KEY:
        return jsonify(error="GROK_API_KEY is missing from practical/.env"), 500

    body = request.get_json(silent=True) or {}
    prompt = (body.get("prompt") or "").strip()
    if not prompt:
        return jsonify(error="Prompt is empty."), 400

    payload = {
        "model": body.get("model") or DEFAULT_MODEL,
        "messages": [
            {"role": "system", "content": "You are a concise, helpful assistant."},
            {"role": "user", "content": prompt},
        ],
    }

    try:
        res = requests.post(
            API_URL,
            headers={
                "Authorization": f"Bearer {API_KEY}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=60,
        )
        data = res.json()
    except requests.RequestException as exc:
        return jsonify(error=f"Could not reach the Groq API: {exc}"), 502
    except ValueError:
        return jsonify(error=f"Bad response from the API ({res.status_code})."), 502

    if not res.ok:
        err = data.get("error", data)
        msg = err.get("message") if isinstance(err, dict) else str(err)
        return jsonify(error=msg or f"API returned {res.status_code}"), res.status_code

    try:
        reply = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError):
        return jsonify(error="Unexpected response shape from the API."), 502

    return jsonify(reply=reply, usage=data.get("usage"))


if __name__ == "__main__":
    if not API_KEY:
        print("WARNING: GROK_API_KEY not found in practical/.env")
    app.run(debug=True, port=5000)
