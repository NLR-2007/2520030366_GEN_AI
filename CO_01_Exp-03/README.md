# Week-03 — LLM Monitoring with MongoDB

Extends the Week-02 provider-agnostic client. Every LLM call is timed,
token-counted, costed and written to MongoDB Atlas, with request timeouts
and limited retries for temporary failures.

**Providers:** Groq (cloud) and Ollama (local). Groq replaces Gemini from
the lab handout — Week-02 already used Groq, so this continues from it.

## Files

| File | Purpose |
|---|---|
| `usage_logger.py` | Writes one document per call to `llm_monitoring.api_usage` |
| `llm_client.py` | `GroqProvider`, `OllamaProvider`, and the shared `LLMClient` |
| `main.py` | Pick a provider, ask a question |
| `test_client.py` | Unit tests using fake provider + fake logger |
| `.env.example` | Template for `.env` (which is gitignored) |

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env     # then fill in MONGODB_URI
```

`.env` needs `MONGODB_URI`. The Groq API key is read from `Lab/.env`
(`GROK_API_KEY`) unless `GROQ_API_KEY` is set here.

## Run

```bash
python -m unittest -v test_client
python main.py
```

For Ollama: start the app and `ollama pull llama3.2` once.

## Logged fields

`provider`, `model`, `input_tokens`, `output_tokens`, `total_tokens`,
`latency_ms`, `estimated_cost_usd`, `cost_type`, `status`, `attempts`,
`error` (failures only), `timestamp` (UTC — IST is UTC+5:30).

## Behaviour

- **Timeouts** — 5 s connect / 60 s read for both providers; 5 s server
  selection and 10 s socket for MongoDB.
- **Retries** — up to 3 attempts on 408/409/429/5xx, timeouts and
  connection errors, waiting 1 s then 2 s. Permanent errors (401, 404)
  are not retried.
- **Failures** — logged with null tokens and cost, since an unsuccessful
  response reports no usage.
- **Cost** — `GROQ_INPUT_PRICE_PER_MILLION` / `GROQ_OUTPUT_PRICE_PER_MILLION`
  in `.env`, both 0 on the free tier. Ollama is always 0 (local compute,
  electricity and hardware not counted).

## Verify in Atlas

Clusters → **Browse Collections** → database `llm_monitoring`,
collection `api_usage`.
