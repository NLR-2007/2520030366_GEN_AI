"""
Week 4 - Tokenizer Lab
Comparison of BPE, WordPiece and SentencePiece tokenizers
Languages: English, Telugu, Hindi
"""

import sys

# Windows consoles default to cp1252, which cannot print Telugu/Hindi.
# Force UTF-8 so the tokens display correctly however the script is launched.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from transformers import AutoTokenizer

TOKENIZERS = {
    "BPE - GPT-2": "gpt2",
    "WordPiece - Multilingual BERT": "bert-base-multilingual-cased",
    "SentencePiece - XLM-R": "xlm-roberta-base"
}

TEXTS = {
    "English": "Artificial intelligence is changing the world.",
    "Telugu": "\u0c15\u0c43\u0c24\u0c4d\u0c30\u0c3f\u0c2e \u0c2e\u0c47\u0c27\u0c38\u0c4d\u0c38\u0c41 \u0c2a\u0c4d\u0c30\u0c2a\u0c02\u0c1a\u0c3e\u0c28\u0c4d\u0c28\u0c3f \u0c2e\u0c3e\u0c30\u0c41\u0c38\u0c4d\u0c24\u0c4b\u0c02\u0c26\u0c3f.",
    "Hindi": "\u0915\u0943\u0924\u094d\u0930\u093f\u092e \u092c\u0941\u0926\u094d\u0927\u093f\u092e\u0924\u094d\u0924\u093e \u0926\u0941\u0928\u093f\u092f\u093e \u0915\u094b \u092c\u0926\u0932 \u0930\u0939\u0940 \u0939\u0948\u0964"
}


def load_tokenizers():
    loaded = {}
    for display_name, model_name in TOKENIZERS.items():
        print(f"Loading {display_name}...")
        loaded[display_name] = AutoTokenizer.from_pretrained(model_name)
    return loaded


def compare_tokenizers(loaded_tokenizers):
    for language, text in TEXTS.items():
        print("\n" + "=" * 75)
        print(f"LANGUAGE: {language}")
        print(f"INPUT TEXT: {text}")
        print("=" * 75)

        for tokenizer_name, tokenizer in loaded_tokenizers.items():
            encoded = tokenizer(
                text,
                add_special_tokens=True,
                truncation=False
            )
            token_ids = encoded["input_ids"]
            tokens = tokenizer.convert_ids_to_tokens(token_ids)

            print(f"\n{tokenizer_name}")
            print("Tokens     :", tokens)
            print("Token IDs  :", token_ids)
            print("Token count:", len(token_ids))


def main():
    tokenizers = load_tokenizers()
    compare_tokenizers(tokenizers)


if __name__ == "__main__":
    main()
