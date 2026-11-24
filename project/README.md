# Kisaan Krushi V-01 GPT

**Project title:** Design and Development of Kisaan Krushi V-01 GPT: An Offline Android Language Model Focused on Factual Accuracy

Kisaan Krushi V-01 GPT is a small English language model being trained from scratch for offline use on an Android phone with 4 GB of RAM. The project prioritizes fewer confident factual errors, followed by fast generation on mobile hardware. The name does not imply that the current model is trained specifically for agriculture; its present corpus is general English and Wikipedia content.

## Project approach

- Curate a quality-controlled dataset of Wikipedia articles and licensed conversations.
- Pretrain a compact decoder-only Transformer on factual and general English text.
- Fine-tune separately on conversations to improve short replies.
- Evaluate factual answers on held-out questions and check basic conversation behavior.
- Investigate retrieval and abstention before claiming reliable factual assistance.
- Measure speed and memory on Android before calling the deployment complete.

## Dataset and model

| Item | Current design |
| --- | --- |
| Training tokens | 1,659,383,359 after planned repetition |
| Wikipedia articles | 207,454, capped at 1,200 words |
| Factual lead passages | 99,999 |
| Conversations | 229,707 |
| Tokenizer | SentencePiece BPE, 16,000 tokens |
| Parameters | 79,073,920 |
| Architecture | 16 layers, 640-dimensional model, grouped-query attention, SwiGLU, RoPE, RMSNorm |
| Context length | 1,024 tokens |
| Training hardware | Two NVIDIA Tesla T4 GPUs on Kaggle |

The dataset passed 51 structural and provenance audit checks. That audit does not establish factual correctness or model quality.

## Status (24 November 2026)

| Stage | Status |
| --- | --- |
| Dataset preparation and audit | Complete |
| Architecture and training code | Complete |
| Pretraining | In progress; final completion not yet verified |
| Conversational fine-tuning | Pending |
| Evaluation | Pending; 400 held-out factual questions prepared |
| Android deployment | Pending |

No final factual accuracy, hallucination rate, or Android speed result is available yet. Those numbers will be added only after evaluation. The training notebook and full dataset artifacts are maintained separately and are not part of this coursework repository.
