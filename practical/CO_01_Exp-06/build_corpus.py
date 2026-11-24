"""
Exp-06 step 1 — build the course corpus from Wikipedia.

Downloads the plain text of a fixed list of Wikipedia articles covering the
Module I-III syllabus and writes them to corpus/ as .txt files, plus a
manifest.json recording the exact URL, revision id and fetch date of each
article.

The revision id matters: Wikipedia changes. Recording it is what makes the
golden question set reproducible - an expected answer is only "correct"
against a specific revision of a specific article.

Run:
    python build_corpus.py
    python build_corpus.py --refresh        # re-download everything

Text is licensed CC BY-SA 4.0 by Wikipedia contributors. See corpus/LICENSE.txt.
"""

import os
import re
import sys
import json
import time
import argparse
import urllib.parse
import urllib.request
import urllib.error

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
CORPUS_DIR = os.path.join(HERE, "corpus")
MANIFEST = os.path.join(CORPUS_DIR, "manifest.json")

API = "https://en.wikipedia.org/w/api.php"
# Wikipedia asks automated clients to identify themselves. Generic name only -
# no personal contact details go out in a request.
USER_AGENT = "genai-lab-corpus/1.0 (educational coursework; contact via repository)"

# Articles chosen to cover the syllabus the golden question set is drawn from.
ARTICLES = [
    # Module I - the GenAI landscape, tokenization, models as dependencies
    ("Artificial intelligence", "module-1"),
    ("Machine learning", "module-1"),
    ("Deep learning", "module-1"),
    ("Natural language processing", "module-1"),
    ("Generative artificial intelligence", "module-1"),
    ("Large language model", "module-1"),
    ("Foundation model", "module-1"),
    ("Byte pair encoding", "module-1"),
    ("Hallucination (artificial intelligence)", "module-1"),
    # Module II - how generation actually works
    ("Transformer (deep learning architecture)", "module-2"),
    ("Attention (machine learning)", "module-2"),
    ("BERT (language model)", "module-2"),
    # Module III - embeddings, vector stores, RAG
    ("Word embedding", "module-3"),
    ("Vector database", "module-3"),
    ("Retrieval-augmented generation", "module-3"),
    # Module IV - prompting
    ("Prompt engineering", "module-4"),
    ("Fine-tuning (deep learning)", "module-4"),
]

# Wikipedia boilerplate sections that carry no answerable content.
DROP_SECTIONS = {
    "see also", "references", "further reading", "external links",
    "notes", "bibliography", "citations", "sources", "footnotes",
}

LICENSE_TEXT = """Corpus source and licence
=========================

Every .txt file in this folder is the plain-text extract of an English
Wikipedia article, downloaded by build_corpus.py.

Text from Wikipedia is licensed under the Creative Commons
Attribution-ShareAlike 4.0 License (CC BY-SA 4.0).
  https://creativecommons.org/licenses/by-sa/4.0/
  https://en.wikipedia.org/wiki/Wikipedia:Copyrights

Attribution: Wikipedia contributors. The exact article URL and the revision
id used for each file are recorded in manifest.json, so any extract can be
traced back to the precise version it came from.

This corpus is used for educational coursework (Generative AI and AI Platform
Engineering, 25CS2207E) and is redistributed under the same licence.
"""


def slugify(title):
    s = title.lower()
    s = re.sub(r"\(.*?\)", "", s)
    s = re.sub(r"[^a-z0-9]+", "_", s).strip("_")
    return s


def api_get(params, attempts=5):
    """GET the Wikipedia API, backing off on 429 / 5xx rather than giving up.
    Wikipedia rate-limits unauthenticated clients, so a plain loop over 17
    articles will otherwise fail halfway through."""
    params = dict(params)
    params["format"] = "json"
    params["formatversion"] = "2"
    url = API + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={
        "User-Agent": USER_AGENT,
        "Accept": "application/json",
    })

    wait = 2.0
    last = None
    for attempt in range(1, attempts + 1):
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                return json.load(resp)
        except urllib.error.HTTPError as ex:
            last = ex
            if ex.code != 429 and ex.code < 500:
                raise
            retry_after = ex.headers.get("Retry-After") if ex.headers else None
            delay = float(retry_after) if (retry_after or "").isdigit() else wait
        except urllib.error.URLError as ex:
            last = ex
            delay = wait
        if attempt < attempts:
            print("      rate-limited, waiting %.0fs (attempt %d/%d)"
                  % (delay, attempt, attempts))
            time.sleep(delay)
            wait = min(wait * 2, 30.0)
    raise last if last else RuntimeError("request failed")


def fetch_article(title):
    """Return (plain_text, meta) for one article, or raise."""
    data = api_get({
        "action": "query",
        "prop": "extracts|info",
        "titles": title,
        "explaintext": "1",
        "exsectionformat": "wiki",
        "inprop": "url",
        "redirects": "1",
    })
    pages = data.get("query", {}).get("pages", [])
    if not pages:
        raise RuntimeError("no pages returned")
    page = pages[0]
    if page.get("missing"):
        raise RuntimeError("article does not exist")
    text = page.get("extract") or ""
    if not text.strip():
        raise RuntimeError("empty extract")
    meta = {
        "resolved_title": page.get("title", title),
        "pageid": page.get("pageid"),
        "revision_id": page.get("lastrevid"),
        "url": page.get("fullurl") or
               "https://en.wikipedia.org/wiki/" + urllib.parse.quote(title.replace(" ", "_")),
    }
    return text, meta


def clean_text(text):
    """Drop boilerplate sections, normalise whitespace, keep == headings ==."""
    lines = text.split("\n")
    out = []
    skipping = False
    for line in lines:
        m = re.match(r"^(={2,})\s*(.+?)\s*\1\s*$", line.strip())
        if m:
            depth = len(m.group(1))
            name = m.group(2).strip().lower()
            if depth == 2:
                skipping = name in DROP_SECTIONS
            if skipping:
                continue
            out.append(line.strip())
            continue
        if skipping:
            continue
        out.append(line.rstrip())
    text = "\n".join(out)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text + "\n"


def list_sections(text):
    names = []
    for line in text.split("\n"):
        m = re.match(r"^(={2,})\s*(.+?)\s*\1\s*$", line.strip())
        if m and len(m.group(1)) == 2:
            names.append(m.group(2).strip())
    return names


def main():
    parser = argparse.ArgumentParser(description="Build the Wikipedia corpus for Exp-06.")
    parser.add_argument("--refresh", action="store_true",
                        help="Re-download articles even if the .txt already exists")
    parser.add_argument("--delay", type=float, default=1.5,
                        help="Seconds to wait between requests (be polite)")
    args = parser.parse_args()

    os.makedirs(CORPUS_DIR, exist_ok=True)
    with open(os.path.join(CORPUS_DIR, "LICENSE.txt"), "w", encoding="utf-8") as f:
        f.write(LICENSE_TEXT)

    entries = []
    failures = []
    print("Building corpus from English Wikipedia")
    print("=" * 72)

    for title, module in ARTICLES:
        slug = slugify(title)
        path = os.path.join(CORPUS_DIR, slug + ".txt")

        if os.path.exists(path) and not args.refresh:
            print("  skip (exists)   %s" % title)
            body = open(path, encoding="utf-8").read()
            entries.append({
                "slug": slug, "title": title, "module": module,
                "file": "corpus/%s.txt" % slug,
                "chars": len(body), "words": len(body.split()),
                "sections": list_sections(body),
                "note": "kept from previous run; re-run with --refresh to update",
            })
            continue

        try:
            raw, meta = fetch_article(title)
        except Exception as ex:
            print("  FAILED          %s  (%s: %s)" % (title, type(ex).__name__, ex))
            failures.append({"title": title, "error": "%s: %s" % (type(ex).__name__, ex)})
            continue

        body = clean_text(raw)
        with open(path, "w", encoding="utf-8") as f:
            f.write(body)

        entry = {
            "slug": slug,
            "title": title,
            "module": module,
            "file": "corpus/%s.txt" % slug,
            "resolved_title": meta["resolved_title"],
            "url": meta["url"],
            "pageid": meta["pageid"],
            "revision_id": meta["revision_id"],
            "permalink": "https://en.wikipedia.org/w/index.php?oldid=%s" % meta["revision_id"],
            "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "license": "CC BY-SA 4.0",
            "chars": len(body),
            "words": len(body.split()),
            "sections": list_sections(body),
        }
        entries.append(entry)
        print("  ok  %6s words  %s" % (entry["words"], title))
        time.sleep(args.delay)

    manifest = {
        "source": "English Wikipedia",
        "license": "CC BY-SA 4.0 (https://creativecommons.org/licenses/by-sa/4.0/)",
        "attribution": "Wikipedia contributors",
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "builder": "build_corpus.py",
        "document_count": len(entries),
        "total_words": sum(e.get("words", 0) for e in entries),
        "documents": entries,
        "failures": failures,
    }
    with open(MANIFEST, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print("=" * 72)
    print("Documents : %d" % len(entries))
    print("Words     : %s" % f"{manifest['total_words']:,}")
    print("Manifest  : %s" % MANIFEST)
    if failures:
        print("Failures  : %d (see manifest.json)" % len(failures))


if __name__ == "__main__":
    main()
