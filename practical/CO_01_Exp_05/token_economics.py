# Token Economics: how the SAME question costs different amounts on different models
#
# You type one question (e.g. "What is AI?"). The script sends it to every chat
# model available on Groq, shows each model's answer, the exact number of input
# and output tokens it consumed, and what those tokens cost.
#
# Run:
#     python token_economics.py
#     python token_economics.py -q "What is AI?"
#     python token_economics.py -q "Explain overfitting" --max-tokens 200
#
# Needs GROQ_API_KEY (read from the .env file next to this script, or from the
# environment). Without a key it falls back to an offline estimate.

import os
import sys
import json
import time
import argparse
import textwrap
import urllib.request
import urllib.error
from dataclasses import dataclass

API_URL = "https://api.groq.com/openai/v1/chat/completions"
# Groq sits behind Cloudflare, which rejects the default urllib User-Agent.
USER_AGENT = "token-economics-lab/1.0"


def _load_dotenv():
    """Minimal .env loader (no dependency). Populates os.environ for missing keys."""
    for name in (".env", ".env.local"):
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), name)
        if not os.path.exists(path):
            continue
        try:
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    k, _, v = line.partition("=")
                    k = k.strip()
                    v = v.strip().strip('"').strip("'")
                    if k and k not in os.environ:
                        os.environ[k] = v
        except OSError:
            pass


_load_dotenv()


def _make_stdout_unicode_safe():
    """Windows consoles default to cp1252 and crash on model output (dashes,
    arrows, emoji). Never let an encoding error hide the token numbers."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


_make_stdout_unicode_safe()

# ---------------------------------------------------------------- CONFIG ----
# price = USD per 1 million tokens.  None = not published / unknown.
MODELS = {
    "openai/gpt-oss-20b": {
        "name": "GPT-OSS 20B",
        "input_per_1m": 0.075,
        "output_per_1m": 0.30,
        "context_window": 131072,
    },
    "openai/gpt-oss-120b": {
        "name": "GPT-OSS 120B",
        "input_per_1m": 0.15,
        "output_per_1m": 0.60,
        "context_window": 131072,
    },
    "openai/gpt-oss-safeguard-20b": {
        "name": "Safeguard GPT-OSS 20B",
        "input_per_1m": 0.075,
        "output_per_1m": 0.30,
        "context_window": 131072,
    },
    "qwen/qwen3.8-27b": {
        "name": "Qwen 3.8 27B",
        "input_per_1m": 0.80,
        "output_per_1m": 4.00,
        "context_window": 131072,
    },
    "allam-2-7b": {
        "name": "Allam 2 7B",
        "input_per_1m": None,
        "output_per_1m": None,
        "context_window": 4096,
    },
}

DEFAULT_QUESTION = "What is AI?"
DEFAULT_MAX_TOKENS = 1024  # generous, so each model stops when it wants to
DEFAULT_TEMPERATURE = 0.7
# The measured per-question cost is projected onto these request counts.
SCALE_REQUESTS = [1_000, 1_000_000]


@dataclass
class Result:
    model: str
    ok: bool
    answer: str = ""
    error: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    input_cost: float = 0.0
    output_cost: float = 0.0
    total_cost: float = 0.0
    priced: bool = True
    latency_s: float = 0.0
    tps: float = 0.0
    answer_chars: int = 0
    finish_reason: str = ""


# ------------------------------------------------------------------ CORE ----
def ask_model(model, question, max_tokens=DEFAULT_MAX_TOKENS,
              temperature=DEFAULT_TEMPERATURE, timeout=90):
    """Send one question to one Groq model and measure its token usage."""
    key = os.environ.get("GROQ_API_KEY")
    if not key:
        return Result(model=model, ok=False, error="GROQ_API_KEY not set")

    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": question}],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }).encode()

    req = urllib.request.Request(
        API_URL,
        data=body,
        headers={
            "Authorization": "Bearer " + key,
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": USER_AGENT,
        },
    )

    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.load(resp)
    except urllib.error.HTTPError as ex:
        raw = ex.read().decode(errors="replace")
        msg = raw.strip()
        try:
            msg = json.loads(raw)["error"]["message"]
        except Exception:
            pass
        return Result(model=model, ok=False, error="HTTP %d: %s" % (ex.code, msg[:200]))
    except Exception as ex:
        return Result(model=model, ok=False, error="%s: %s" % (type(ex).__name__, ex))
    latency = time.perf_counter() - t0

    choice = (data.get("choices") or [{}])[0]
    answer = ((choice.get("message") or {}).get("content") or "").strip()
    usage = data.get("usage") or {}
    in_tok = int(usage.get("prompt_tokens", 0))
    out_tok = int(usage.get("completion_tokens", 0))
    total = int(usage.get("total_tokens", in_tok + out_tok))

    r = Result(
        model=model, ok=True, answer=answer,
        input_tokens=in_tok, output_tokens=out_tok, total_tokens=total,
        latency_s=latency, finish_reason=choice.get("finish_reason") or "",
        answer_chars=len(answer),
    )
    r.tps = (out_tok / latency) if latency > 0 else 0.0

    price_in = MODELS.get(model, {}).get("input_per_1m")
    price_out = MODELS.get(model, {}).get("output_per_1m")
    if price_in is None or price_out is None:
        r.priced = False
    else:
        r.input_cost = in_tok / 1_000_000 * price_in
        r.output_cost = out_tok / 1_000_000 * price_out
        r.total_cost = r.input_cost + r.output_cost
    return r


def estimate_tokens(text):
    """Rough offline heuristic: ~4 characters per token."""
    if not text:
        return 0
    return max(1, len(text) // 4)


# --------------------------------------------------------------- DISPLAY ----
def money(x, places=6):
    return "$%.*f" % (places, x)


def rule(char="=", width=78):
    print(char * width)


def show_answer(r, max_chars=600):
    m = MODELS.get(r.model, {})
    rule("-")
    print("%s  [%s]" % (m.get("name", r.model), r.model))
    rule("-")
    if not r.ok:
        print("  FAILED: " + r.error)
        print()
        return

    text = r.answer or "(empty response)"
    if len(text) > max_chars:
        text = text[:max_chars].rstrip() + " ...[truncated for display]"
    for para in text.split("\n"):
        if para.strip():
            print(textwrap.fill(para, width=74, initial_indent="  ",
                                subsequent_indent="  "))
        else:
            print()

    print()
    print("  Input tokens   : %6s   (the question + chat overhead)" % f"{r.input_tokens:,}")
    print("  Output tokens  : %6s   (the answer)" % f"{r.output_tokens:,}")
    print("  Total tokens   : %6s" % f"{r.total_tokens:,}")
    if r.priced:
        print("  Input cost     : %s  @ $%s/1M" % (money(r.input_cost), m.get("input_per_1m")))
        print("  Output cost    : %s  @ $%s/1M" % (money(r.output_cost), m.get("output_per_1m")))
        print("  Cost for this question: %s" % money(r.total_cost))
    else:
        print("  Cost           : n/a (pricing not published for this model)")
    print("  Latency        : %.2fs   |  %.0f output tok/s   |  answer %s chars"
          % (r.latency_s, r.tps, f"{r.answer_chars:,}"))
    if r.finish_reason == "length":
        print("  NOTE: the answer hit max_tokens, so output tokens are capped, "
              "not the model's natural length.")
    print()


def comparison_table(results):
    ok = [r for r in results if r.ok]
    if not ok:
        print("No successful calls, nothing to compare.")
        return

    rule()
    print("TOKEN + COST COMPARISON (same question, every model)")
    rule()
    print("%-30s%6s%7s%7s%12s%9s%8s"
          % ("Model", "In", "Out", "Total", "Cost/req", "Latency", "Tok/s"))
    print("-" * 79)
    for r in sorted(ok, key=lambda x: (not x.priced, x.total_cost)):
        cost = money(r.total_cost) if r.priced else "n/a"
        print("%-30s%6s%7s%7s%12s%8.2fs%8.0f"
              % (r.model, f"{r.input_tokens:,}", f"{r.output_tokens:,}",
                 f"{r.total_tokens:,}", cost, r.latency_s, r.tps))
    for r in results:
        if not r.ok:
            print("%-30s-- failed: %s" % (r.model, r.error[:35]))
    print()

    rule()
    print("WHAT THE NUMBERS SAY")
    rule()
    verbose = max(ok, key=lambda r: r.output_tokens)
    terse = min(ok, key=lambda r: r.output_tokens)
    fastest = max(ok, key=lambda r: r.tps)
    print("Most verbose   : %s -> %s output tokens" % (verbose.model, f"{verbose.output_tokens:,}"))
    print("Most concise   : %s -> %s output tokens" % (terse.model, f"{terse.output_tokens:,}"))
    if terse.output_tokens:
        print("Verbosity gap  : %.1fx more output tokens for the same question"
              % (verbose.output_tokens / terse.output_tokens))
    print("Fastest output : %s -> %.0f tokens/sec" % (fastest.model, fastest.tps))
    capped = [r for r in ok if r.finish_reason == "length"]
    if capped:
        print("Capped by max_tokens: %s" % ", ".join(r.model for r in capped))
        print("  -> raise --max-tokens to compare how long each model naturally answers.")

    priced = [r for r in ok if r.priced]
    if priced:
        cheap = min(priced, key=lambda r: r.total_cost)
        dear = max(priced, key=lambda r: r.total_cost)
        print("Cheapest       : %s -> %s" % (cheap.model, money(cheap.total_cost)))
        print("Most expensive : %s -> %s" % (dear.model, money(dear.total_cost)))
        if cheap.total_cost > 0:
            print("Cost spread    : %.1fx between cheapest and most expensive"
                  % (dear.total_cost / cheap.total_cost))

        print()
        print("Input vs output share of the bill:")
        for r in sorted(priced, key=lambda x: x.total_cost):
            if r.total_cost <= 0:
                continue
            ip = r.input_cost / r.total_cost * 100
            print("  %-30s input %5.1f%%   output %5.1f%%" % (r.model, ip, 100 - ip))

        print()
        print("The same question billed at scale (identical token counts):")
        print("%-30s%s" % ("Model", "".join("%14s" % f"{n:,} req" for n in SCALE_REQUESTS)))
        for r in sorted(priced, key=lambda x: x.total_cost):
            cells = "".join("%14s" % ("$" + format(r.total_cost * n, ",.2f"))
                            for n in SCALE_REQUESTS)
            print("%-30s%s" % (r.model, cells))
    print()


def offline_estimate(question, max_tokens):
    rule()
    print("OFFLINE ESTIMATE (no GROQ_API_KEY -> no real calls made)")
    rule()
    est_in = estimate_tokens(question)
    print("Question        : %r" % question)
    print("Estimated input : ~%s tokens (chars/4 heuristic)" % f"{est_in:,}")
    print("Assumed output  : %s tokens (max_tokens)" % f"{max_tokens:,}")
    print()
    print("%-30s%6s%7s%14s" % ("Model", "In", "Out", "Est cost/req"))
    print("-" * 57)
    for mid, m in MODELS.items():
        if m["input_per_1m"] is None:
            print("%-30s%6s%7s%14s" % (mid, f"{est_in:,}", f"{max_tokens:,}", "n/a"))
            continue
        c = (est_in / 1_000_000 * m["input_per_1m"]
             + max_tokens / 1_000_000 * m["output_per_1m"])
        print("%-30s%6s%7s%14s" % (mid, f"{est_in:,}", f"{max_tokens:,}", money(c)))
    print()
    print("Set GROQ_API_KEY in .env to get real answers and exact token counts.")
    print()


# ------------------------------------------------------------------ MAIN ----
def main():
    parser = argparse.ArgumentParser(
        description="Ask one question to many models and compare token use and cost.")
    parser.add_argument("-q", "--question", help="The question to ask every model")
    parser.add_argument("--models", help="Comma-separated model ids (default: all)")
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS,
                        help="Cap on the answer length (default %d)" % DEFAULT_MAX_TOKENS)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--no-answers", action="store_true",
                        help="Print only the comparison table, not the answers")
    args = parser.parse_args()

    question = args.question
    if not question:
        if sys.stdin.isatty():
            try:
                question = input("Enter your question [%s]: " % DEFAULT_QUESTION).strip()
            except EOFError:
                question = ""
        question = question or DEFAULT_QUESTION

    if args.models:
        model_ids = [m.strip() for m in args.models.split(",") if m.strip()]
        unknown = [m for m in model_ids if m not in MODELS]
        if unknown:
            print("NOTE: no pricing on file for %s -> cost shown as n/a."
                  % ", ".join(unknown))
    else:
        model_ids = list(MODELS)

    print()
    rule()
    print("TOKEN ECONOMICS - ONE QUESTION, MANY MODELS")
    rule()
    print("Question   : %s" % question)
    print("Models     : %d" % len(model_ids))
    print("max_tokens : %d   temperature: %s" % (args.max_tokens, args.temperature))
    print()

    if not os.environ.get("GROQ_API_KEY"):
        offline_estimate(question, args.max_tokens)
        return

    results = []
    for mid in model_ids:
        print("-> asking %s ..." % mid)
        results.append(ask_model(mid, question,
                                 max_tokens=args.max_tokens,
                                 temperature=args.temperature))
    print()

    if not args.no_answers:
        rule()
        print("ANSWERS")
        rule()
        for r in results:
            show_answer(r)

    comparison_table(results)

    print("Takeaway: the question is identical, so input tokens barely move;")
    print("what changes the bill is how many output tokens each model writes")
    print("and how much that model charges per output token.")
    print()


if __name__ == "__main__":
    main()
