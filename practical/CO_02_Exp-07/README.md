# CO_02_Exp-07 — Next-Token Probabilities, Logits and Temperature

The first experiment of Module II. Opens the model up one layer below the text
it produces: instead of reading a generated sentence, we read the **probability
distribution the model computes before a single token is chosen**.

Handout, skilling session 7 (CO2):

> Inspect next-token probabilities; relate logits, softmax and temperature to
> the text a model actually produces.

## Aim

Show that a language model does not "write". At every step it produces one
score (a **logit**) for each of its 50,257 vocabulary tokens, softmax turns
those scores into a probability distribution, and a decoding rule picks one
token from it. Temperature changes the shape of that distribution before the
pick.

Module I treated the model as a box that returns text. From here on it is a
box that returns **a distribution**, and every generation setting (temperature,
top-k, top-p) is a rule applied to it.

GPT-2 (124M) is used rather than a hosted API because an API returns sampled
text; it does not hand back the raw logits. Running locally is the only way to
see Stage 2.

## The three stages

| Stage | What happens | What you see in the output |
|---|---|---|
| 1 | Text to subword token IDs (BPE) | Position, ID and decoded string for each input token |
| 2 | Forward pass to one raw logit per vocabulary token | Min and max logit across all 50,257 |
| 3 | Logits / temperature, then softmax, then top-k | Ranked candidate table with probabilities |

Raw logits are unbounded real numbers and are **not** probabilities. They are
only meaningful relative to each other — which is exactly why softmax, which
depends only on differences between logits, is the right function to convert
them.

## Files

| File | Purpose |
|---|---|
| `next_token_prob.py` | The lab program — loads GPT-2 once, then analyses prompts in a loop |
| `requirements.txt` | `torch`, `transformers` |
| `output.txt` | Console transcript of the recorded run |
| `gpt2_next_token_results.csv` | Every candidate from every run, appended |

The CSV is written with `utf-8-sig` so Excel on Windows renders the leading
spaces in tokens like `' Paris'` instead of mangling them. Those leading spaces
matter — in GPT-2's BPE vocabulary `' Paris'` (ID 6342) and `'Paris'` are two
different tokens.

## Setup

```powershell
pip install torch transformers
```

The shared `Lab/venv` has `transformers` but **not** `torch`, so this runs on
the global Python, which does.

The first run downloads GPT-2 (~548 MB) from Hugging Face into the local cache
and takes 15–20 minutes. Every later run starts in a couple of seconds. The
`HF_TOKEN` warning is harmless — anonymous downloads work, just with lower
rate limits.

Cache location:

```powershell
python -c "from huggingface_hub import constants; print(constants.HF_HUB_CACHE)"
```

## Run

```powershell
python next_token_prob.py
```

Three inputs per round, each with a safe default:

| Input | Default | Range |
|---|---|---|
| Prompt | — | `exit` / `quit` / `q` to leave |
| Top-k | 6 | clamped to 1–50 |
| Temperature | 1.0 | must be > 0; invalid input falls back to 1.0 |

Every round **appends** to `gpt2_next_token_results.csv`, so runs accumulate
and stay comparable across sessions.

To capture a transcript to a file, set the output encoding first — the table
uses block characters, which Windows' default cp1252 codec cannot encode when
stdout is redirected:

```powershell
$env:PYTHONIOENCODING = "utf-8"
python next_token_prob.py > output.txt
```

This affects redirection only; printing to the console works without it.

## Results

GPT-2 124M, `top_k=8`, five recorded runs.

### A. Temperature sweep — "The capital of France is"

The same prompt and therefore the **same raw logits** every time. Only the
scaling changes.

| Rank | Token | Raw logit | T = 0.2 | T = 1.0 | T = 2.0 |
|---|---|---|---|---|---|
| 1 | `' the'` | -100.2498 | **88.24%** | 8.46% | 0.47% |
| 2 | `' now'` | -100.8175 | 5.16% | 4.79% | 0.36% |
| 3 | `' a'` | -100.8555 | 4.27% | 4.62% | 0.35% |
| 4 | `' France'` | -101.2102 | 0.72% | 3.24% | 0.29% |
| 5 | `' Paris'` | -101.2143 | 0.71% | 3.22% | 0.29% |
| 6 | `' in'` | -101.4052 | 0.27% | 2.66% | 0.27% |
| 7 | `' also'` | -101.4141 | 0.26% | 2.64% | 0.27% |
| 8 | `' not'` | -101.5168 | 0.16% | 2.38% | 0.25% |
| | **Top-8 total** | | **99.80%** | **32.02%** | **2.55%** |

Logit range across the full vocabulary: **-129.8683 to -100.2498**.

### What it shows

- **Temperature does not change what the model believes.** The raw logits are
  identical in all three columns and the ranking never changes. Only the gaps
  between probabilities change.
- **T = 0.2 sharpens.** Dividing by 0.2 multiplies every logit gap by 5. The
  0.57 gap between rank 1 and rank 2 becomes 2.84, and rank 1 takes 88% of the
  mass. The top 8 tokens hold 99.80% — the other 50,249 are effectively dead.
- **T = 2.0 flattens.** Gaps halve, rank 1 falls to 0.47%, and the top 8 hold
  just **2.55%**. 97.45% of the probability sits in the long tail, which is why
  high-temperature text wanders into nonsense: the model is being asked to pick
  from 50,000 near-equal options.
- **Temperature is a ratio knob, not a creativity knob.** It rescales
  differences. The word "creativity" describes the consequence, not the
  mechanism.
- **The famous answer is not the top token.** GPT-2 ranks `' Paris'` **fifth**
  at 3.22% — below `' the'`, `' now'` and `' a'`. The model is not answering a
  question; it is continuing a string, and "The capital of France is the
  largest…" is a perfectly ordinary English continuation. A base model has no
  instinct that a question deserves an answer. That instinct is what
  instruction tuning adds, and it is Module IV's subject.

### B. Confidence varies by prompt — all at T = 1.0

| Prompt | Top token | Top prob | Top-8 mass |
|---|---|---|---|
| `The capital of France is` | `' the'` | 8.46% | 32.02% |
| `I woke up early and decided to` | `' go'` | 18.45% | 42.93% |
| `She opened the door and saw` | `' a'` | 16.61% | 47.38% |

Grammar constrains more sharply than meaning. After "decided to" only a verb
can follow, so the distribution concentrates (18.45% on `' go'`). After "is"
almost any part of speech is legal, so the mass spreads. **Top-8 mass is a
usable uncertainty signal**, and it is the same quantity top-p (nucleus
sampling) thresholds on.

### C. Why greedy decoding is not enough

At T = 1.0 the top-8 of the France prompt holds only 32% of the mass. Greedy
decoding always takes rank 1 and throws away the other 68% — which is why it
produces repetitive, locked-in text. Sampling from the distribution is what
makes varied text possible, and temperature/top-k/top-p are the rules that
decide how much of the tail stays eligible.

## Known limitations

| Limitation | Why |
|---|---|
| One token only, no generation loop | Shows the mechanism, not autoregression — the next experiment feeds the chosen token back in |
| Top-k display is not top-k *sampling* | Nothing is sampled here; the table ranks, it does not pick |
| Top-p (nucleus) not implemented | The cumulative-probability line is the quantity it would threshold on |
| GPT-2 124M is small and from 2019 | Big models are sharper, but the logit to softmax to sample mechanism is identical |
| Temperature applied to logits, not after softmax | Correct — applying it after softmax would not renormalise properly |

## Viva questions

1. What is a logit? Why is it not a probability?
2. Why can a logit be large and negative, and why does that not matter?
3. What does softmax do, and why does it depend only on *differences* between logits?
4. Where exactly is temperature applied — before or after softmax? Why does the order matter?
5. T = 0.2 and T = 2.0 give different probabilities for the same prompt. Did the model's knowledge change? Explain.
6. What does T approaching 0 give in the limit? What about T approaching infinity?
7. Why must temperature be strictly greater than 0?
8. GPT-2 puts `' Paris'` fifth for "The capital of France is". Is that a bug? What would change it?
9. Why are `' Paris'` and `'Paris'` different token IDs?
10. The top 8 tokens hold 32% of the mass at T = 1.0. What is the other 68%, and what does sampling from it look like?
11. What is the difference between top-k *display* and top-k *sampling*?
12. How does top-p (nucleus sampling) differ from top-k, and which number in this output would it use?
13. Why does greedy decoding produce repetitive text?
14. Why does "I woke up early and decided to" concentrate more probability than "The capital of France is"?
15. Why does this experiment use a local model instead of a hosted API?
