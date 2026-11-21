"""
Exp-06 step 2 - run the golden question set and record the baseline.

Two modes, run the same 24 questions:

  --mode closed    No context at all. The model answers from its training
                   data. This is the v0.1 baseline the handout asks for:
                   deliberately ungrounded, so later modules have a "before"
                   number to beat.

  --mode grounded  Retrieves passages from corpus/ and instructs the model to
                   answer ONLY from them, with citations, and to say
                   NOT_IN_SOURCES when the passages do not contain the answer.
                   This is a preview of the Module III RAG service, using a
                   stdlib lexical retriever instead of Chroma embeddings.

Every call records input tokens, output tokens, cost, latency and a score, so
quality and economics are measured together.

Run:
    python run_baseline.py --mode closed
    python run_baseline.py --mode grounded
    python run_baseline.py --mode both --tag v0.1

Needs GROQ_API_KEY in .env (see .env.example).
"""

import os
import re
import sys
import json
import time
import math
import argparse
import unicodedata
import urllib.request
import urllib.error
from collections import Counter

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
CORPUS_DIR = os.path.join(HERE, "corpus")
RESULTS_DIR = os.path.join(HERE, "results")
GOLDEN = os.path.join(HERE, "golden_questions.json")

API_URL = "https://api.groq.com/openai/v1/chat/completions"
USER_AGENT = "genai-lab-baseline/1.0"

DEFAULT_MODEL = "openai/gpt-oss-120b"
# USD per 1M tokens, same table as Exp-05.
PRICES = {
    "openai/gpt-oss-20b": (0.075, 0.30),
    "openai/gpt-oss-120b": (0.15, 0.60),
    "openai/gpt-oss-safeguard-20b": (0.075, 0.30),
    "qwen/qwen3.8-27b": (0.80, 4.00),
    "allam-2-7b": (None, None),
}

DECLINE_TOKEN = "NOT_IN_SOURCES"

CLOSED_SYSTEM = (
    "You are a teaching assistant for a Generative AI course. "
    "Answer the question accurately and concisely, in at most 120 words."
)

GROUNDED_SYSTEM = (
    "You are a course assistant that answers ONLY from the provided sources.\n"
    "Rules:\n"
    "1. Use only the numbered SOURCES below. Do not use outside knowledge.\n"
    "2. Cite the sources you used as [1], [2] etc. inline.\n"
    "3. If the sources do not contain the answer, reply with exactly "
    + DECLINE_TOKEN + " and one short sentence saying what is missing. "
    "Do not guess, and do not fall back on general knowledge.\n"
    "4. Answer in at most 120 words."
)


def load_dotenv():
    for name in (".env", ".env.local"):
        path = os.path.join(HERE, name)
        if not os.path.exists(path):
            continue
        try:
            with open(path, encoding="utf-8") as f:
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
    # Fall back to the shared key at the Lab root, like Exp-03 does.
    if not os.environ.get("GROQ_API_KEY"):
        shared = os.path.join(os.path.dirname(HERE), ".env")
        if os.path.exists(shared):
            try:
                with open(shared, encoding="utf-8") as f:
                    for line in f:
                        if "=" in line:
                            k, _, v = line.partition("=")
                            if k.strip() in ("GROQ_API_KEY", "GROK_API_KEY"):
                                os.environ.setdefault("GROQ_API_KEY",
                                                      v.strip().strip('"').strip("'"))
            except OSError:
                pass


load_dotenv()


# ------------------------------------------------------------- retrieval ----
STOPWORDS = set("""a an and are as at be been being by for from has have how in into is it its
of on or that the their there these this to was were what when which who why will with you your
does do can could should would about between within than then""".split())


def tokenize(text):
    return [w for w in re.findall(r"[a-z0-9]+", text.lower())
            if w not in STOPWORDS and len(w) > 1]


def chunk_corpus(chunk_words=180, overlap=40):
    """Split every corpus document into overlapping word chunks, keeping the
    section heading each chunk came from so a citation can point at it."""
    manifest_path = os.path.join(CORPUS_DIR, "manifest.json")
    if not os.path.exists(manifest_path):
        print("ERROR: corpus/manifest.json not found. Run build_corpus.py first.")
        sys.exit(1)
    manifest = json.load(open(manifest_path, encoding="utf-8"))

    chunks = []
    for doc in manifest["documents"]:
        path = os.path.join(HERE, doc["file"])
        if not os.path.exists(path):
            continue
        text = open(path, encoding="utf-8").read()
        section = "lead"
        buf = []

        def flush(sec, words):
            if len(words) < 20:
                return
            chunks.append({
                "doc": doc["slug"],
                "title": doc.get("resolved_title") or doc["title"],
                "url": doc.get("url", ""),
                "revision_id": doc.get("revision_id"),
                "section": sec,
                "text": " ".join(words),
            })

        for line in text.split("\n"):
            m = re.match(r"^(={2,})\s*(.+?)\s*\1\s*$", line.strip())
            if m:
                # section boundary: emit what we have
                while len(buf) > chunk_words:
                    flush(section, buf[:chunk_words])
                    buf = buf[chunk_words - overlap:]
                flush(section, buf)
                buf = []
                section = m.group(2).strip()
                continue
            buf.extend(line.split())
            while len(buf) > chunk_words:
                flush(section, buf[:chunk_words])
                buf = buf[chunk_words - overlap:]
        flush(section, buf)

    # BM25-ish statistics
    df = Counter()
    for c in chunks:
        c["tokens"] = tokenize(c["text"])
        for w in set(c["tokens"]):
            df[w] += 1
    n = len(chunks)
    avg_len = sum(len(c["tokens"]) for c in chunks) / max(n, 1)
    idf = {w: math.log(1 + (n - d + 0.5) / (d + 0.5)) for w, d in df.items()}
    return chunks, idf, avg_len


def retrieve(query, chunks, idf, avg_len, k=4, k1=1.5, b=0.75):
    """Plain BM25 over the chunks. No embeddings, no vector store - that is
    Module III's job. This is the honest lexical baseline."""
    q = tokenize(query)
    scored = []
    for c in chunks:
        tf = Counter(c["tokens"])
        dl = len(c["tokens"]) or 1
        s = 0.0
        for w in q:
            if w not in tf:
                continue
            f = tf[w]
            s += idf.get(w, 0.0) * (f * (k1 + 1)) / (f + k1 * (1 - b + b * dl / avg_len))
        if s > 0:
            scored.append((s, c))
    scored.sort(key=lambda x: -x[0])
    return scored[:k]


# ------------------------------------------------------------------- llm ----
class RateGovernor:
    """Keeps the run inside a tokens-per-minute budget.

    Groq's free tier allows 8,000 TPM. A grounded request carries four
    retrieved passages, so it costs ~1,300 tokens - about six per minute.
    Without pacing, most of the golden set fails with HTTP 429. This is the
    Exp-05 rate-limit arithmetic applied for real: tokens/minute divided by
    tokens/request gives requests/minute, and the run has to respect it.

    Implemented as a token bucket rather than a 60-second window. A sliding
    window tempts you to "reset" it after a 429, but the server's own window
    has not reset - doing that produces a cascade of 429s. A bucket that only
    ever refills at tpm/60 tokens per second cannot outrun the server.
    """

    def __init__(self, tpm=8000, headroom=0.85):
        self.capacity = tpm * headroom
        self.rate = self.capacity / 60.0        # tokens per second
        # Start EMPTY, not full. A full bucket lets the first few calls burst
        # at several times the sustained rate, and the server counts that
        # burst in its own window - which is what trips the 429 cascade.
        self.tokens = 0.0
        self.last = time.monotonic()
        self.enabled = tpm > 0

    def _refill(self):
        now = time.monotonic()
        self.tokens = min(self.capacity, self.tokens + (now - self.last) * self.rate)
        self.last = now

    def wait_for(self, projected):
        if not self.enabled:
            return
        projected = min(projected, self.capacity)
        self._refill()
        if self.tokens < projected:
            sleep = (projected - self.tokens) / self.rate + 0.25
            print("      TPM budget: need %d, have %d, waiting %.0fs"
                  % (projected, self.tokens, sleep))
            time.sleep(sleep)
            self._refill()
        self.tokens -= projected

    def reconcile(self, projected, actual):
        """Charge the difference once the real usage is known."""
        if not self.enabled:
            return
        self.tokens = max(0.0, min(self.capacity, self.tokens + projected - actual))

    def drain(self):
        """A 429 means the server disagrees with our accounting. Empty the
        bucket so the next call waits a full refill instead of charging in."""
        if self.enabled:
            self._refill()
            self.tokens = 0.0


GOVERNOR = RateGovernor()


def call_groq(model, system, user, max_tokens=400, temperature=0.0, timeout=90,
              attempts=6):
    key = os.environ.get("GROQ_API_KEY")
    if not key:
        return {"ok": False, "error": "GROQ_API_KEY not set"}
    # Reserve budget before sending: prompt (chars/4) plus the worst-case answer.
    projected = (len(system) + len(user)) // 4 + max_tokens
    GOVERNOR.wait_for(projected)
    body = json.dumps({
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }).encode()
    req = urllib.request.Request(API_URL, data=body, headers={
        "Authorization": "Bearer " + key,
        "Content-Type": "application/json",
        "Accept": "application/json",
        "User-Agent": USER_AGENT,
    })
    data = None
    last_err = None
    for attempt in range(1, attempts + 1):
        t0 = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.load(resp)
            break
        except urllib.error.HTTPError as ex:
            raw = ex.read().decode(errors="replace")
            msg = raw.strip()
            try:
                msg = json.loads(raw)["error"]["message"]
            except Exception:
                pass
            last_err = "HTTP %d: %s" % (ex.code, msg[:200])
            if ex.code != 429 and ex.code < 500:
                return {"ok": False, "error": last_err}
            # Groq tells us how long to wait; prefer that over guessing.
            wait = None
            hdr = ex.headers.get("retry-after") if ex.headers else None
            if hdr:
                try:
                    wait = float(hdr)
                except ValueError:
                    wait = None
            if wait is None:
                m = re.search(r"try again in ([\d.]+)s", msg, re.I)
                wait = float(m.group(1)) + 0.5 if m else min(2.0 * attempt, 20.0)
            if attempt < attempts:
                print("      rate limited, waiting %.1fs (attempt %d/%d)"
                      % (wait, attempt, attempts))
                time.sleep(wait)
                GOVERNOR.drain()   # server's window has NOT reset; back right off
                continue
            return {"ok": False, "error": last_err}
        except Exception as ex:
            last_err = "%s: %s" % (type(ex).__name__, ex)
            if attempt >= attempts:
                return {"ok": False, "error": last_err}
            time.sleep(min(2.0 * attempt, 10.0))
    if data is None:
        return {"ok": False, "error": last_err or "request failed"}
    latency = time.perf_counter() - t0

    choice = (data.get("choices") or [{}])[0]
    usage = data.get("usage") or {}
    in_tok = int(usage.get("prompt_tokens", 0))
    out_tok = int(usage.get("completion_tokens", 0))
    GOVERNOR.reconcile(projected, int(usage.get("total_tokens", in_tok + out_tok)))
    p_in, p_out = PRICES.get(model, (None, None))
    cost = None
    if p_in is not None:
        cost = in_tok / 1_000_000 * p_in + out_tok / 1_000_000 * p_out
    return {
        "ok": True,
        "answer": ((choice.get("message") or {}).get("content") or "").strip(),
        "input_tokens": in_tok,
        "output_tokens": out_tok,
        "total_tokens": int(usage.get("total_tokens", in_tok + out_tok)),
        "cost_usd": cost,
        "latency_s": latency,
        "finish_reason": choice.get("finish_reason") or "",
    }


# ----------------------------------------------------------------- score ----
def looks_like_decline(answer):
    answer = normalize(answer)
    if DECLINE_TOKEN.lower() in answer:
        return True
    patterns = [
        r"\bnot\b.{0,25}\b(in|contained in|present in|found in)\b.{0,25}\bsources?\b",
        r"\bsources? (do|does) not\b",
        r"\bcannot (answer|be answered|provide)\b",
        r"\bno information\b.{0,30}\bsources?\b",
        # An honest "I don't know" counts as a decline even without the
        # NOT_IN_SOURCES token - closed-book mode is never told to use it.
        r"\b(do not|don't) have\b.{0,40}\b(information|data|details|access)\b",
        r"\b(do not|don't) have access\b",
        r"\b(am|'m) (not able|unable) to\b",
        r"\bunable to (provide|answer|confirm|verify)\b",
        r"\b(do not|don't) know\b",
        r"\bno (publicly |public )?(available )?information\b",
        r"\bnot (publicly )?(available|disclosed|published|specified)\b",
        r"\bi (do not|don't) have\b",
    ]
    return any(re.search(p, answer, re.I) for p in patterns)


def normalize(text):
    """Models emit typographic Unicode - non-breaking hyphens (U+2011), curly
    quotes, fullwidth brackets, non-breaking spaces. Compare on a flattened
    form or a correct answer scores zero for cosmetic reasons."""
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    for ch in "‐‑‒–—―−":
        text = text.replace(ch, "-")
    for ch in "‘’‛′":
        text = text.replace(ch, "'")
    for ch in "“”„″":
        text = text.replace(ch, '"')
    text = text.replace(" ", " ").replace(" ", " ").replace(" ", " ")
    text = re.sub(r"[*_`#]+", " ", text)          # markdown emphasis
    text = re.sub(r"\s+", " ", text)
    text = text.lower().strip()
    # The corpus mixes British and American spelling ("optimisation",
    # "neighbor"); a model may answer in either. Fold them together.
    for uk, us in (("neighbour", "neighbor"), ("behaviour", "behavior"),
                   ("optimisation", "optimization"), ("organisation", "organization"),
                   ("normalisation", "normalization"), ("tokenisation", "tokenization"),
                   ("recognise", "recognize"), ("modelling", "modeling"),
                   ("summarise", "summarize"), ("minimisation", "minimization")):
        text = text.replace(uk, us)
    # "117 M" / "1 k" -> "117m" / "1k": models space out magnitude suffixes.
    text = re.sub(r"(\d)\s+([kmbgt])\b", r"\1\2", text)
    # Spell numbers as digits on BOTH sides, so "twelve" meets "12".
    for word, digit in _NUMBER_WORDS:
        text = re.sub(r"\b" + word + r"\b", digit, text)
    return text


# Applied to answers and to must_include terms alike, so the comparison stays
# symmetric: whichever form the model picks, both sides canonicalise to digits.
_NUMBER_WORDS = [
    ("twenty", "20"), ("nineteen", "19"), ("eighteen", "18"), ("seventeen", "17"),
    ("sixteen", "16"), ("fifteen", "15"), ("fourteen", "14"), ("thirteen", "13"),
    ("twelve", "12"), ("eleven", "11"), ("ten", "10"), ("nine", "9"),
    ("eight", "8"), ("seven", "7"), ("six", "6"), ("five", "5"),
    ("four", "4"), ("three", "3"), ("two", "2"), ("one", "1"),
]


# Crude suffix stripper. Not linguistically principled - it only has to make
# "compression"/"compress" and "generalizing"/"generalize" compare equal.
# Longest suffixes first, and applied repeatedly: "generalizing" -> "general"
# -> "gener" has to meet "generalize" coming the other way.
_SUFFIXES = ("ization", "isation", "ational", "izing", "ising", "ation",
             "ingly", "izes", "ises", "ness", "ment", "able", "ible", "ical",
             "ing", "ize", "ise", "ers", "ied", "ies", "est", "ive", "ion",
             "al", "ly", "ed", "er", "es", "s")


def stem(word):
    w = word.lower()
    for _ in range(3):              # iterate: compression -> compress
        for suf in _SUFFIXES:
            if not w.endswith(suf) or len(w) - len(suf) < 4:
                continue
            # don't turn "compress" into "compres" or "class" into "clas"
            if suf in ("s", "es") and w.endswith(("ss", "us", "is")):
                continue
            w = w[: -len(suf)]
            break
        else:
            break
    return w


def stems_of(text):
    return {stem(w) for w in re.findall(r"[a-z0-9]+", normalize(text))}


def term_present(term, answer_norm, answer_stems):
    """A term counts as present if it appears verbatim, or if every content
    word in it appears in the answer after stemming."""
    t = normalize(term)
    if t and t in answer_norm:
        return True
    content = [w for w in re.findall(r"[a-z0-9]+", t)
               if w not in STOPWORDS and len(w) > 1]
    if not content:
        return False
    return all(stem(w) in answer_stems for w in content)


def score_answer(q, answer):
    """Keyword-recall score, plus the decline check for the trap questions.

    This is intentionally crude. Module IV replaces it with a rubric and an
    LLM-as-a-judge calibrated against human grades. Recording the crude score
    now is what makes that upgrade measurable.
    """
    a = normalize(answer)
    a_stems = stems_of(answer)
    if not q["answerable"]:
        declined = looks_like_decline(answer or "")
        return {
            "score": 1.0 if declined else 0.0,
            "declined": declined,
            "matched": [],
            "missed": [],
            "verdict": "correctly declined" if declined
                       else "ANSWERED AN UNANSWERABLE QUESTION (hallucination risk)",
        }

    terms = q["must_include"]
    matched = [t for t in terms if term_present(t, a, a_stems)]
    missed = [t for t in terms if not term_present(t, a, a_stems)]
    score = len(matched) / len(terms) if terms else 0.0
    declined = looks_like_decline(answer or "")
    if declined:
        return {"score": 0.0, "declined": True, "matched": matched, "missed": missed,
                "verdict": "declined a question the sources DO support"}
    verdict = "pass" if score >= 0.75 else ("partial" if score >= 0.4 else "fail")
    return {"score": score, "declined": False, "matched": matched,
            "missed": missed, "verdict": verdict}


def citation_check(q, answer, retrieved):
    """Did the grounded answer cite, and did it cite the right document?"""
    # Models cite as [1], but also as fullwidth brackets or parentheses.
    cited = sorted(set(int(n) for n in re.findall(
        r"[\[【［(](\d{1,2})[\]】］)]", answer or "")
        if 1 <= int(n) <= len(retrieved)))
    docs = [retrieved[i - 1]["doc"] for i in cited]
    expected = (q.get("source") or {}).get("doc") if q.get("source") else None
    return {
        "cited": cited,
        "cited_docs": docs,
        "expected_doc": expected,
        "cited_correct_doc": bool(expected and expected in docs),
        "retrieved_docs": [c["doc"] for c in retrieved],
        "expected_doc_retrieved": bool(expected and expected in [c["doc"] for c in retrieved]),
    }


# ------------------------------------------------------------------- run ----
def build_grounded_prompt(question, retrieved):
    blocks = []
    for i, c in enumerate(retrieved, 1):
        blocks.append("[%d] %s - section: %s\n%s" % (i, c["title"], c["section"], c["text"]))
    return "SOURCES\n=======\n" + "\n\n".join(blocks) + \
           "\n\nQUESTION\n========\n" + question


def run_mode(mode, questions, model, chunks, idf, avg_len, top_k, max_tokens):
    rows = []
    print()
    print("=" * 78)
    print("MODE: %s   model: %s" % (mode.upper(), model))
    print("=" * 78)

    for q in questions:
        if mode == "closed":
            retrieved = []
            res = call_groq(model, CLOSED_SYSTEM, q["question"], max_tokens=max_tokens)
        else:
            hits = retrieve(q["question"], chunks, idf, avg_len, k=top_k)
            retrieved = [c for _, c in hits]
            prompt = build_grounded_prompt(q["question"], retrieved)
            res = call_groq(model, GROUNDED_SYSTEM, prompt, max_tokens=max_tokens)

        if not res["ok"]:
            print("  %s  ERROR  %s" % (q["id"], res["error"]))
            rows.append({"id": q["id"], "question": q["question"], "mode": mode,
                         "ok": False, "error": res["error"]})
            continue

        sc = score_answer(q, res["answer"])
        row = {
            "id": q["id"], "question": q["question"], "mode": mode, "ok": True,
            "answerable": q["answerable"], "category": q["category"],
            "difficulty": q["difficulty"], "module": q["module"],
            "answer": res["answer"],
            "expected_answer": q["expected_answer"],
            "input_tokens": res["input_tokens"], "output_tokens": res["output_tokens"],
            "total_tokens": res["total_tokens"], "cost_usd": res["cost_usd"],
            "latency_s": round(res["latency_s"], 3),
            "finish_reason": res["finish_reason"],
            "score": round(sc["score"], 3), "verdict": sc["verdict"],
            "declined": sc["declined"],
            "matched_terms": sc["matched"], "missed_terms": sc["missed"],
        }
        if mode == "grounded":
            row["retrieval"] = citation_check(q, res["answer"], retrieved)
            row["sources_shown"] = [
                {"n": i, "doc": c["doc"], "section": c["section"],
                 "url": c["url"], "revision_id": c["revision_id"]}
                for i, c in enumerate(retrieved, 1)
            ]
        rows.append(row)

        flag = "" if q["answerable"] else "  [trap]"
        print("  %s  score %.2f  %-44s tok %4d/%4d%s"
              % (q["id"], sc["score"], sc["verdict"][:44],
                 res["input_tokens"], res["output_tokens"], flag))
    return rows


def summarise(rows):
    ok = [r for r in rows if r.get("ok")]
    ans = [r for r in ok if r.get("answerable")]
    traps = [r for r in ok if r.get("answerable") is False]
    costs = [r["cost_usd"] for r in ok if r.get("cost_usd") is not None]
    s = {
        "questions_run": len(rows),
        "calls_ok": len(ok),
        "calls_failed": len(rows) - len(ok),
        "answerable_count": len(ans),
        "answerable_mean_score": round(sum(r["score"] for r in ans) / len(ans), 3) if ans else None,
        "answerable_passed": sum(1 for r in ans if r["verdict"] == "pass"),
        "answerable_partial": sum(1 for r in ans if r["verdict"] == "partial"),
        "answerable_failed": sum(1 for r in ans if r["verdict"] == "fail"),
        "trap_count": len(traps),
        "traps_correctly_declined": sum(1 for r in traps if r["declined"]),
        "traps_answered_anyway": sum(1 for r in traps if not r["declined"]),
        "total_input_tokens": sum(r["input_tokens"] for r in ok),
        "total_output_tokens": sum(r["output_tokens"] for r in ok),
        "total_cost_usd": round(sum(costs), 6) if costs else None,
        "mean_latency_s": round(sum(r["latency_s"] for r in ok) / len(ok), 3) if ok else None,
    }
    if ok and ok[0]["mode"] == "grounded":
        g = [r for r in ok if r.get("answerable") and r.get("retrieval")]
        s["retrieval_hit_rate_at_k"] = round(
            sum(1 for r in g if r["retrieval"]["expected_doc_retrieved"]) / len(g), 3) if g else None
        s["cited_correct_doc_rate"] = round(
            sum(1 for r in g if r["retrieval"]["cited_correct_doc"]) / len(g), 3) if g else None
    return s


def print_summary(mode, s):
    print()
    print("-" * 78)
    print("SUMMARY - %s" % mode.upper())
    print("-" * 78)
    print("Answerable questions : %d" % s["answerable_count"])
    print("  mean score         : %s" % s["answerable_mean_score"])
    print("  pass / partial / fail : %d / %d / %d"
          % (s["answerable_passed"], s["answerable_partial"], s["answerable_failed"]))
    print("Trap questions       : %d" % s["trap_count"])
    print("  correctly declined : %d" % s["traps_correctly_declined"])
    print("  answered anyway    : %d   <- hallucination exposure"
          % s["traps_answered_anyway"])
    if "retrieval_hit_rate_at_k" in s:
        print("Retrieval Hit Rate@k : %s" % s["retrieval_hit_rate_at_k"])
        print("Cited correct doc    : %s" % s["cited_correct_doc_rate"])
    print("Tokens in / out      : %s / %s"
          % (f"{s['total_input_tokens']:,}", f"{s['total_output_tokens']:,}"))
    print("Total cost           : %s" % (("$%.6f" % s["total_cost_usd"])
                                         if s["total_cost_usd"] is not None else "n/a"))
    print("Mean latency         : %ss" % s["mean_latency_s"])
    if s["calls_failed"]:
        print("FAILED CALLS         : %d" % s["calls_failed"])


def main():
    parser = argparse.ArgumentParser(description="Run the golden question set.")
    parser.add_argument("--mode", choices=["closed", "grounded", "both"], default="both")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--top-k", type=int, default=4, help="Passages to retrieve (grounded)")
    parser.add_argument("--max-tokens", type=int, default=400)
    parser.add_argument("--tag", default="v0.1", help="Release tag recorded in the results file")
    parser.add_argument("--tpm", type=int, default=8000,
                        help="Tokens-per-minute budget to stay inside (free tier = 8000; 0 disables pacing)")
    parser.add_argument("--only", help="Comma-separated question ids, for a quick check")
    args = parser.parse_args()

    global GOVERNOR
    GOVERNOR = RateGovernor(tpm=args.tpm)

    golden = json.load(open(GOLDEN, encoding="utf-8"))
    questions = golden["questions"]
    if args.only:
        want = {x.strip().upper() for x in args.only.split(",")}
        questions = [q for q in questions if q["id"].upper() in want]

    if not os.environ.get("GROQ_API_KEY"):
        print("ERROR: GROQ_API_KEY not set. Copy .env.example to .env and add your key.")
        sys.exit(1)

    chunks, idf, avg_len = chunk_corpus()
    print("Corpus: %d chunks from corpus/manifest.json" % len(chunks))
    print("Golden set: %d questions (%d answerable, %d traps)"
          % (len(questions),
             sum(1 for q in questions if q["answerable"]),
             sum(1 for q in questions if not q["answerable"])))

    modes = ["closed", "grounded"] if args.mode == "both" else [args.mode]
    report = {
        "tag": args.tag,
        "run_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "model": args.model,
        "golden_set_version": golden["version"],
        "top_k": args.top_k,
        "max_tokens": args.max_tokens,
        "tpm_budget": args.tpm,
        "corpus_chunks": len(chunks),
        "modes": {},
    }

    for mode in modes:
        rows = run_mode(mode, questions, args.model, chunks, idf, avg_len,
                        args.top_k, args.max_tokens)
        s = summarise(rows)
        print_summary(mode, s)
        report["modes"][mode] = {"summary": s, "rows": rows}

    if len(modes) == 2:
        c = report["modes"]["closed"]["summary"]
        g = report["modes"]["grounded"]["summary"]
        print()
        print("=" * 78)
        print("CLOSED-BOOK vs GROUNDED")
        print("=" * 78)
        print("%-26s%14s%14s%14s" % ("Metric", "closed", "grounded", "change"))
        print("-" * 68)

        def line(label, a, b, fmt="%.3f"):
            if a is None or b is None:
                return
            delta = b - a
            print("%-26s%14s%14s%14s" % (label, fmt % a, fmt % b,
                                         ("%+" + fmt[1:]) % delta))

        line("Mean score (answerable)", c["answerable_mean_score"], g["answerable_mean_score"])
        line("Passed", c["answerable_passed"], g["answerable_passed"], "%d")
        line("Traps declined", c["traps_correctly_declined"], g["traps_correctly_declined"], "%d")
        line("Traps answered anyway", c["traps_answered_anyway"], g["traps_answered_anyway"], "%d")
        line("Input tokens", c["total_input_tokens"], g["total_input_tokens"], "%d")
        if c["total_cost_usd"] is not None and g["total_cost_usd"] is not None:
            line("Cost (USD)", c["total_cost_usd"], g["total_cost_usd"], "%.6f")
        line("Mean latency (s)", c["mean_latency_s"], g["mean_latency_s"])
        report["comparison"] = {"closed": c, "grounded": g}

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out = os.path.join(RESULTS_DIR, "baseline_%s.json" % args.tag)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print()
    print("Full per-question record written to: %s" % out)


if __name__ == "__main__":
    main()
