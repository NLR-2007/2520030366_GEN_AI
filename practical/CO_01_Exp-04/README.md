# CO_01_Exp-04 — Tokenizer Lab

Comparison of **BPE**, **WordPiece** and **SentencePiece** tokenizers on English, Telugu and Hindi text.

## Aim

Implement and compare three subword tokenization approaches — Byte Pair Encoding (BPE), WordPiece and SentencePiece — and observe how each splits text into tokens and how many tokens each produces.

## Tokenizers used

| Tokenizer | Basic idea | Pretrained model |
|---|---|---|
| BPE (Byte Pair Encoding) | Repeatedly merges frequent adjacent symbol pairs. GPT-2 uses byte-level BPE. | `gpt2` |
| WordPiece | Learned subword vocabulary; a word is split into pieces when the whole word is not in the vocabulary. | `bert-base-multilingual-cased` |
| SentencePiece | Language-independent framework that learns subword units directly from raw text. XLM-R uses SentencePiece with a BPE model. | `xlm-roberta-base` |

> **Note:** This lab compares *real pretrained tokenizers*, not three algorithms trained on the same dataset with the same vocabulary size. GPT-2 is primarily English-trained, while mBERT and XLM-R are multilingual. Token-count differences therefore reflect both the tokenizer method and each model's vocabulary/training data.

## Files

| File | Purpose |
|---|---|
| `tokenizer_comparison.py` | Main program — loads the three tokenizers and compares them |
| `requirements.txt` | Dependencies (`transformers`, `sentencepiece`) |
| `output.txt` | Captured run output |
| `WEEK4_Tokenizer_Lab_BPE_WordPiece_SentencePiece.docx` | Lab manual |

## Setup

From this folder, activate the shared virtual environment and install dependencies:

```powershell
..\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Internet access is needed on the first run so the pretrained tokenizer files can be downloaded (they are then cached locally).

## Run

```powershell
python tokenizer_comparison.py
```

The script forces UTF-8 on stdout itself, so the Telugu and Hindi tokens display correctly even in a default `cp1252` Windows console.

### Run it with the venv, not the global Python

Make sure the venv is active (or that VS Code's selected interpreter is `Labenv`). If the global Python is used instead and it has PyTorch installed, `from transformers import AutoTokenizer` will pull in all of `torch` first and appear to hang for a long time before printing anything. This lab only needs the tokenizers, and the venv has no torch — so the import takes a few seconds.

## Observed token counts

Counts include each tokenizer's special tokens (`[CLS]`/`[SEP]` for mBERT, `<s>`/`</s>` for XLM-R; GPT-2 adds none).

| Language | BPE — GPT-2 | WordPiece — mBERT | SentencePiece — XLM-R |
|---|---|---|---|
| English | 8 | 10 | 10 |
| Telugu | 112 | 18 | 14 |
| Hindi | 68 | 17 | 14 |

### What this shows

- **English** — all three are close. GPT-2 is lowest (8) partly because it adds no special tokens; mBERT and XLM-R each add two.
- **Telugu and Hindi** — GPT-2 explodes to 112 and 68 tokens. Its byte-level BPE has no Indic subwords in its vocabulary, so it falls back to encoding the UTF-8 **bytes** of each character (visible as the `à°`/`à¤` fragments in the output). Each Indic character costs roughly three tokens.
- **mBERT and XLM-R** stay compact (14–18) because their vocabularies were trained on multilingual text and contain real Telugu/Hindi subwords such as `ప్రపంచ`, `దునియా`, `▁बदल`.
- XLM-R beats mBERT on both Indic languages — its larger SentencePiece vocabulary captures longer meaningful units (`స్తోంది` as one token vs. mBERT's `##స్` + `##తోంది`).

A lower token count is not automatically "better", but a tokenizer that shatters a language into bytes wastes context window and makes it far harder for the model to learn that language.

## Viva questions

1. What is tokenization, and why do language models need it?
2. What is the difference between a word and a subword token?
3. Explain the basic idea of Byte Pair Encoding (BPE).
4. How does WordPiece split an unknown or rare word?
5. What is SentencePiece? Does SentencePiece always mean the Unigram algorithm?
6. Why might GPT-2 split Telugu or Hindi text into more tokens than a multilingual tokenizer?
7. Does a lower token count always mean that a tokenizer is better? Give a reason.
8. What are token IDs, and where are they used?
9. Why should we compare tokenizers using the same input sentences?
10. What is the difference between token count and the number of characters in a sentence?
