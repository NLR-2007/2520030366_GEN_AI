# CO_01_Exp-06 — Golden Question Set and Baseline (release v0.1)

The last experiment of Module I. Builds the **ruler** rather than a feature: a
frozen set of questions with verified answers and sources, a corpus those
answers come from, and the first measured run against it.

Handout, skilling session 6 (CO1):

> Write the golden question set with expected answers and sources; first
> baseline run — release v0.1

## Aim

Establish the evaluation baseline that every later module is measured against.
From the course rationale:

> Every module ends in a measured release. A golden question set, written in
> Module I, is re-run after every module, so each design choice (generation
> defaults, chunk size, prompt version, fine-tune or not) is backed by numbers
> and recorded as an architecture decision.

Because the questions never change, the difference between v0.1 and any later
release is a measurement rather than an impression.

## What "baseline" means here

The handout's v0.1 is a **logged LLM client** — no retrieval. So the headline
baseline is deliberately *ungrounded*: the model answers from its training
data and is expected to do badly, especially on the trap questions.

This experiment runs **two modes** so the gap is visible instead of assumed:

| Mode | Context given | What it represents |
|---|---|---|
| `closed` | Nothing — just the question | The v0.1 baseline the handout asks for |
| `grounded` | Retrieved passages from `corpus/`, answer-only-from-sources instructions, citations required | A preview of the Module III RAG service |

`grounded` uses a **BM25 lexical retriever written in the standard library** —
no embeddings, no vector store. That is intentional: Chroma, embeddings and
Hit Rate@k proper are Module III's work. Having a lexical floor now means the
embedding-based version later has something concrete to beat.

## Files

| File | Purpose |
|---|---|
| `build_corpus.py` | Downloads the Wikipedia corpus and records provenance |
| `golden_questions.json` | The frozen golden set — 24 questions |
| `run_baseline.py` | Runs the set in either mode, scores it, logs tokens/cost/latency |
| `corpus/` | 17 Wikipedia articles as plain text |
| `corpus/manifest.json` | URL, revision id, fetch date, word count, sections per document |
| `corpus/LICENSE.txt` | CC BY-SA 4.0 notice and attribution |
| `rescore.py` | Re-grades a saved run with the current grader, no API calls |
| `results/baseline_v0.1.json` | Full per-question record of the baseline run |
| `run_v0.1.log` | Console output of the run |
| `.env.example` | Template for `.env` (which is gitignored) |

No third-party packages. Everything uses the standard library, so it runs on
the global Python without activating the shared venv.

## The corpus

17 English Wikipedia articles chosen to cover the syllabus the questions are
drawn from — **77,765 words**.

| Module | Articles |
|---|---|
| I | Artificial intelligence, Machine learning, Deep learning, Natural language processing, Generative artificial intelligence, Large language model, Foundation model, Byte pair encoding, Hallucination (AI) |
| II | Transformer (deep learning architecture), Attention (machine learning), BERT (language model) |
| III | Word embedding, Vector database, Retrieval-augmented generation |
| IV | Prompt engineering, Fine-tuning (deep learning) |

`build_corpus.py` records the **revision id** of every article, not just its
URL. This is the part that makes the golden set reproducible: Wikipedia
changes, so an expected answer is only "correct" against a specific revision.
Each manifest entry carries a permalink of the form
`https://en.wikipedia.org/w/index.php?oldid=<revision_id>`.

Boilerplate sections (References, See also, External links, Further reading…)
are stripped; `== Section ==` headings are kept so citations can point at a
section.

**Licence:** Wikipedia text is CC BY-SA 4.0, attributed to Wikipedia
contributors, redistributed here under the same licence for coursework. See
`corpus/LICENSE.txt`.

## The golden question set

24 questions in `golden_questions.json`: **20 answerable** from the corpus and
**4 traps** that are not.

Each question carries:

| Field | Purpose |
|---|---|
| `id` | Stable handle (`Q01`…`Q24`) so runs line up across releases |
| `question` | What gets asked |
| `expected_answer` | The verified reference answer |
| `must_include` | Distinctive facts the grader checks for |
| `source` | `{doc, section}` — URL and revision id resolve from the manifest |
| `answerable` | `false` for the traps |
| `module`, `category`, `difficulty` | So you can see *which kind* of question fails |

Sources name a document slug rather than repeating the URL, so provenance
lives in exactly one place and cannot drift between two files.

### The trap questions

Q21–Q24 have no answer in the corpus. A grounded system is required to decline
them; a closed-book model will usually answer anyway, and that is precisely
the behaviour the baseline exists to expose. They escalate in nastiness:

- **Q21** — lab attendance policy. Plainly absent.
- **Q22** — exact Groq price per million output tokens. Deliberately adjacent
  to Exp-05, so a model matching on *topic* rather than on *sources* will bite.
- **Q23** — Telugu vs Hindi token counts "according to the corpus". The corpus
  covers BPE generally but holds no such measurement; those numbers are
  Exp-04's.
- **Q24** — the course coordinator's name and email. A grounded system should
  refuse rather than guess at a person's identity or contact details.

Every `must_include` term was checked to actually occur in its cited document,
so the set never grades against a fact the corpus does not contain.

## Scoring

Keyword recall: the fraction of a question's `must_include` terms present in
the answer. `pass` ≥ 0.75, `partial` ≥ 0.4, otherwise `fail`. Traps score 1.0
for declining and 0.0 for answering.

This is crude on purpose — Module IV replaces it with a rubric and an
LLM-as-a-judge calibrated against human grades. Recording the crude score now
is what makes that upgrade measurable.

Getting it to be *fairly* crude took three fixes, each of which was marking
correct answers wrong:

1. **Unicode.** Models emit non-breaking hyphens (`U+2011`), curly quotes and
   non-breaking spaces. `"Philip Gage"` failed to match `"Philip‑Gage"`.
   Answers are now NFKC-normalised with dashes and quotes folded to ASCII.
2. **Morphology.** `"compression"` did not match "compress", nor
   `"generalize"` match "generalizing". A term now matches if it appears
   verbatim *or* if every content word in it matches after light stemming.
3. **Citation brackets.** The model cited `【1】` (fullwidth), not `[1]`, so
   correct citations scored zero. The regex accepts `[1]`, `【1】` and `(1)`.

British/American spellings are folded too, since the corpus mixes
"optimisation" with "neighbor".

Before these fixes, grounded mode *appeared* to score worse than closed-book
while actually being more faithful — the scorer was measuring typography.

## Rate limiting

Grounded requests carry four retrieved passages, so each costs ~1,300 tokens.
Groq's free tier allows **8,000 tokens per minute** — about six such requests.
A naive loop over 24 questions fails most of them with HTTP 429.

`run_baseline.py` therefore contains a `RateGovernor` that tracks tokens used
in a rolling 60-second window and sleeps before it would exceed the budget,
plus 429 retry honouring Groq's `retry-after`. This is the Exp-05 rate-limit
arithmetic applied for real: tokens-per-minute ÷ tokens-per-request =
requests-per-minute, and the run has to respect it.

Pass `--tpm 250000` on a developer-tier key, or `--tpm 0` to disable pacing.

## Setup

```powershell
copy .env.example .env     # then paste your Groq API key
```

The script also falls back to the shared key at the practical root (`../.env`), like
Exp-03 does.

## Run

```powershell
python build_corpus.py                      # once; writes corpus/ + manifest
python build_corpus.py --refresh            # re-download, updating revision ids

python run_baseline.py --mode both --tag v0.1
python run_baseline.py --mode closed        # ungrounded only
python run_baseline.py --mode grounded      # answers from our material only
python run_baseline.py --only Q01,Q21       # quick check
```

| Flag | Meaning |
|---|---|
| `--mode` | `closed`, `grounded` or `both` |
| `--model` | Groq model id (default `openai/gpt-oss-120b`) |
| `--top-k` | Passages retrieved per question (default 4) |
| `--max-tokens` | Answer cap (default 400) |
| `--tpm` | Tokens-per-minute budget (default 8000; `0` disables) |
| `--tag` | Release tag recorded in the results filename |
| `--only` | Comma-separated question ids |

A full run takes several minutes on a free-tier key — almost all of it waiting
on the TPM budget, not on the model.

## Results — baseline v0.1

Model `openai/gpt-oss-120b`, temperature 0.0, `top_k=4`, `max_tokens=400`, 659 corpus
chunks. 48/48 calls succeeded, zero errors.

| Metric | Closed-book | Grounded | Change |
|---|---|---|---|
| Mean score (20 answerable) | 0.501 | **0.722** | +0.221 |
| Pass / partial / fail | 6 / 6 / 8 | **10 / 8 / 1** | +4 pass, −7 fail |
| Traps correctly declined | 1 / 4 | **4 / 4** | +3 |
| Retrieval Hit Rate@4 | n/a | 0.95 | — |
| Cited correct document | n/a | 0.75 | — |
| Input tokens | 2,870 | 22,860 | **8.0x** |
| Output tokens | 5,681 | 4,957 | −13% |
| Cost | $0.003839 | $0.006403 | +67% |
| Mean latency | 1.246 s | 1.368 s | +0.12 s |

### What it shows

- **Grounding raises accuracy 44%** (0.501 → 0.722); failures drop from 8 to 1.
- **Grounding stops fabrication.** Closed-book answered 3 of 4 unanswerable questions.
  On Q22 it stated the Groq price as "$0.75 per 1M output tokens" — the real figure
  from Exp-05 is **$0.30**. On Q21 it invented an attendance policy. Grounded declined
  all four.
- **Grounding is paid for in input tokens, not output.** Input rises 8x; output *falls*
  13%, so net cost rises only 67%. Exp-05's lesson again: output dominates the bill.
- **Latency is not the cost** (+0.12 s). The real constraint is the rate limit — 8x the
  input tokens means 8x fewer requests per minute inside the same 8,000 TPM budget.

### Known failures

| Id | What happened | What it motivates |
|---|---|---|
| Q14 | BM25 returned four *Transformer* passages for a *BERT* question; the model correctly declined | Module III embeddings — the clearest case for dense retrieval |
| Q04 | Right document retrieved, wrong chunk ranked first (parameter count instead of the lead definition) | Chunk-size and ranking tuning |
| Q04, Q13, Q19, Q20 | Correct document retrieved but not cited (0.75 vs 0.95) | Module IV structured output with a required citations field |

### Re-grading instead of re-running

Scoring is deterministic given the stored answers, so when the grader changes an earlier
release is re-scored offline rather than re-run:

```powershell
python rescore.py results/baseline_v0.1.json --write
```

```
mode      id    was                          -> now
closed    Q24   0.00 ANSWERED AN UNANSWERABL -> 1.00 correctly declined
```

Exactly one score changed, confirming the fix touched only the intended case. Every
release must be scored by the same grader, or the numbers are not comparable.


## Viva questions

1. What is a golden question set, and why must it be frozen?
2. Why record each source's revision id and not just its URL?
3. What is the difference between the closed-book and grounded modes?
4. Why is the v0.1 baseline deliberately ungrounded?
5. What are the trap questions for, and what does a model answering them tell you?
6. What is BM25, and how does it differ from embedding-based retrieval?
7. What does Hit Rate@k measure? What does it *not* measure?
8. A system retrieves the right document but cites the wrong one. Which metric catches that?
9. Why is keyword-recall scoring a weak grader, and what replaces it in Module IV?
10. Why did grounded mode's input token count rise so much over closed-book?
11. How do you compute the maximum requests per minute from a TPM limit?
12. What would make a later release's score *not* comparable to this baseline?
