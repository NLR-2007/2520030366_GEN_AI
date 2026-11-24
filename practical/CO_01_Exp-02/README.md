# Week 02 — Local vs Cloud LLM

Terminal chat client that talks to two interchangeable providers:

| # | Provider | Where it runs | Endpoint |
|---|----------|---------------|----------|
| 1 | Ollama | local machine | `http://localhost:11434` |
| 2 | Groq | cloud | `https://api.groq.com/openai/v1` |

The provider and model are picked at the terminal, conversation history is kept
so follow-up questions have context, and each reply is tagged with the provider,
model and response time for comparison.

## Run

```bash
pip install -r requirements.txt
python app.py
```

The Groq key is read from `practical/.env` (`GROK_API_KEY`); Ollama needs `ollama serve`
running with at least one model pulled.

In-chat commands: `/switch` change provider · `/clear` reset history · `/exit` quit.

## Screenshots

| # | Screenshot | Shows |
|---|------------|-------|
| 1 | [01-startup-provider-menu.jpg](ScreenShots/01-startup-provider-menu.jpg) | App start, provider menu |
| 2 | [02-ollama-model-list-and-first-chat.jpg](ScreenShots/02-ollama-model-list-and-first-chat.jpg) | Ollama model list, first local reply (8.6s) |
| 3 | [03-ollama-what-is-llm-answer.jpg](ScreenShots/03-ollama-what-is-llm-answer.jpg) | Longer local answer from `llama3.2:3b` (10.3s) |
| 4 | [04-groq-model-list-selected.jpg](ScreenShots/04-groq-model-list-selected.jpg) | Switch to Groq, cloud model list |
| 5 | [05-groq-first-reply.jpg](ScreenShots/05-groq-first-reply.jpg) | First cloud reply from `openai/gpt-oss-20b` (1.0s) |
| 6 | [06-groq-followup-with-history.jpg](ScreenShots/06-groq-followup-with-history.jpg) | Follow-up answered with history kept (1.3s) |

Local runs take ~8–10s on this machine, the cloud models ~1s.
