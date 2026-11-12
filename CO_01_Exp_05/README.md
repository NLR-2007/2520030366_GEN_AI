# CO_01_Exp_05 — Token Economics

One question, many models. Ask something like **"What is AI?"**, send that exact
question to every chat model on Groq, and measure how many **input** and
**output** tokens each model consumes — and what those tokens cost.

## Aim

Understand token economics by measurement rather than by estimation: show that
for an identical prompt, the input token count hardly varies between models,
while the output token count and the per-token price vary enough to change the
bill by an order of magnitude.

## How it works

1. You type a question (or pass `-q "..."`).
2. The same question goes to each model in the `MODELS` table as a single user
   message.
3. Token counts are read from the API response's `usage` field — these are the
   provider's own numbers, not a heuristic.
4. Cost is computed as `input_tokens / 1M × input_price + output_tokens / 1M × output_price`.
5. A comparison table and an analysis block are printed: verbosity, speed,
   cheapest vs. most expensive, the input/output split of the bill, and the
   same question projected to 1,000 and 1,000,000 requests.

## Models compared

Prices are USD per 1 million tokens.

| Model id | Name | Input | Output |
|---|---|---|---|
| `openai/gpt-oss-20b` | GPT-OSS 20B | 0.075 | 0.30 |
| `openai/gpt-oss-120b` | GPT-OSS 120B | 0.15 | 0.60 |
| `openai/gpt-oss-safeguard-20b` | Safeguard GPT-OSS 20B | 0.075 | 0.30 |
| `qwen/qwen3.8-27b` | Qwen 3.8 27B | 0.80 | 4.00 |
| `allam-2-7b` | Allam 2 7B | n/a | n/a |

`allam-2-7b` has no published price on file, so its cost shows as `n/a` — it is
still measured for tokens, latency and speed. Add a price to the `MODELS` dict
in the script to include it in the cost comparison.

## Files

| File | Purpose |
|---|---|
| `token_economics.py` | The whole experiment — asks, measures, compares |
| `output.txt` | Captured run output for "What is AI?" |
| `.env.example` | Template for `.env` (which is gitignored) |

No third-party packages are needed. The script calls the REST endpoint with the
standard library (`urllib`), so there is nothing to `pip install`.

## Setup

```powershell
copy .env.example .env     # then paste your Groq API key into it
```

`.env` needs one line:

```
GROQ_API_KEY=gsk_...
```

A key from [console.groq.com](https://console.groq.com) on the free tier is
enough for this lab. Without a key the script still runs and prints a clearly
labelled offline estimate instead of real answers.

## Run

```powershell
python token_economics.py                                  # prompts for a question
python token_economics.py -q "What is AI?"
python token_economics.py -q "Explain overfitting" --max-tokens 200
python token_economics.py -q "What is AI?" --no-answers     # table only
python token_economics.py --models openai/gpt-oss-20b,qwen/qwen3.8-27b
```

| Flag | Meaning |
|---|---|
| `-q`, `--question` | The question to ask every model |
| `--models` | Comma-separated model ids (default: all) |
| `--max-tokens` | Cap on answer length (default 1024) |
| `--temperature` | Sampling temperature (default 0.7) |
| `--no-answers` | Print only the comparison table |

Keep `--max-tokens` generous. If models hit the cap, every answer is the same
length and the verbosity comparison becomes meaningless — the script warns when
this happens.

## Observed output

"What is AI?", `--max-tokens 1024`, temperature 0.7 (full run in `output.txt`):

| Model | Input | Output | Total | Cost/request | Latency | Tok/s |
|---|---|---|---|---|---|---|
| `openai/gpt-oss-20b` | 75 | 910 | 985 | $0.000279 | 1.54s | 593 |
| `openai/gpt-oss-safeguard-20b` | 75 | 1,024 | 1,099 | $0.000313 | 1.53s | 670 |
| `openai/gpt-oss-120b` | 75 | 886 | 961 | $0.000543 | 2.57s | 345 |
| `qwen/qwen3.8-27b` | 16 | 515 | 531 | $0.002073 | 1.25s | 412 |
| `allam-2-7b` | 11 | 470 | 481 | n/a | 0.69s | 681 |

Same question at scale:

| Model | 1,000 requests | 1,000,000 requests |
|---|---|---|
| `openai/gpt-oss-20b` | $0.28 | $278.62 |
| `openai/gpt-oss-safeguard-20b` | $0.31 | $312.82 |
| `openai/gpt-oss-120b` | $0.54 | $542.85 |
| `qwen/qwen3.8-27b` | $2.07 | $2,072.80 |

### What this shows

- **Input tokens are nearly identical** — 11 to 75 tokens for the same three-word
  question. The spread is not the question; it is each model's chat template and
  system scaffolding. The GPT-OSS family adds ~64 tokens of overhead that Allam
  does not.
- **Output tokens are where the variation lives** — 470 to 1,024 tokens, a 2.2x
  gap, purely because some models answer a simple question at length and others
  briefly. Nothing in the request asked for a longer answer.
- **Output dominates the bill: 98–99% of every model's cost.** Output tokens are
  priced 4–5x higher than input tokens, and there are far more of them. Trimming
  a prompt saves very little; constraining answer length saves a lot.
- **Cost spread is 7.4x** between the cheapest and most expensive model for
  identical work. Qwen costs more both ways: ~10x the input price and ~13x the
  output price of GPT-OSS 20B.
- **Price does not track speed.** The cheapest model is not the slowest, and
  `gpt-oss-120b` is both the slowest (345 tok/s) and nearly 2x the price of
  `gpt-oss-20b`.
- **Capped answers distort the comparison.** `safeguard-20b` stopped at exactly
  1,024 tokens — it hit `max_tokens`, so its real natural length is unknown and
  its cost here is a floor, not a measurement.

## Implementation notes

Three things about this environment that the script works around:

- **No SDK.** The installed `openai` 1.51.0 is incompatible with this machine's
  `httpx` version (`Client.__init__() got an unexpected keyword argument 'proxies'`),
  so requests go to `https://api.groq.com/openai/v1/chat/completions` through
  `urllib` directly. This also means zero dependencies.
- **User-Agent.** Groq is behind Cloudflare, which rejects the default `urllib`
  User-Agent with HTTP 403 / error 1010. The script sends a plain
  `token-economics-lab/1.0` instead.
- **Console encoding.** Windows consoles default to `cp1252` and raise
  `UnicodeEncodeError` on model output containing non-breaking hyphens, arrows or
  emoji. `stdout` is reconfigured to UTF-8 with replacement so an encoding error
  can never hide the token numbers.

Failures are per-model: if one model errors, its message appears in the table and
the remaining models still report.

## Viva questions

1. What is a token? Why is billing done per token instead of per word or per character?
2. Why do input and output tokens have different prices?
3. For the same question, why do different models report different input token counts?
4. Which side — input or output — dominated the cost in this experiment, and why?
5. If you had to cut your API bill in half, would you shorten prompts or shorten answers? Justify with the numbers above.
6. What does `max_tokens` control, and why does setting it too low ruin this comparison?
7. What is the `usage` field in an API response, and why is it more trustworthy than a chars/4 estimate?
8. Why does the chars/4 heuristic exist at all, and when is it good enough?
9. A model is 2x cheaper per token but writes 3x longer answers. Which is cheaper overall?
10. What is the context window, and how is it different from `max_tokens`?
11. Why does the same model return a different number of output tokens on two runs of the same question?
12. How does temperature affect token consumption, if at all?
