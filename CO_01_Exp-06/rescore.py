"""
Re-grade a saved baseline run with the CURRENT grader - no API calls.

Scoring is deterministic given the stored answers, so when the grader changes
(see the four bugs in the lab record) an earlier release can be re-scored
offline instead of re-run. This is what keeps releases comparable: every
release must be scored by the SAME grader.

Run:
    python rescore.py results/baseline_v0.1.json
    python rescore.py results/baseline_v0.1.json --write
"""

import os
import sys
import json
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_baseline as R        # noqa: E402  (reuse the live grader)

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


def rescore(path, write=False):
    report = json.load(open(path, encoding="utf-8"))
    golden = {q["id"]: q for q in
              json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                          "golden_questions.json"),
                             encoding="utf-8"))["questions"]}
    changes = []

    for mode, block in report["modes"].items():
        for row in block["rows"]:
            if not row.get("ok"):
                continue
            q = golden.get(row["id"])
            if q is None:
                continue
            new = R.score_answer(q, row["answer"])
            old_score, old_verdict = row["score"], row["verdict"]
            if (round(new["score"], 3) != old_score) or (new["verdict"] != old_verdict):
                changes.append((mode, row["id"], old_score, old_verdict,
                                round(new["score"], 3), new["verdict"]))
            row["score"] = round(new["score"], 3)
            row["verdict"] = new["verdict"]
            row["declined"] = new["declined"]
            row["matched_terms"] = new["matched"]
            row["missed_terms"] = new["missed"]
        block["summary"] = R.summarise(block["rows"])

    print("Re-scored %s with the current grader" % os.path.basename(path))
    print("=" * 78)
    if not changes:
        print("No score changed - the saved run already used this grader.")
    else:
        print("%-9s %-5s %-28s -> %s" % ("mode", "id", "was", "now"))
        print("-" * 78)
        for mode, qid, os_, ov, ns, nv in changes:
            print("%-9s %-5s %.2f %-23s -> %.2f %s" % (mode, qid, os_, ov[:23], ns, nv))
    print()

    for mode in report["modes"]:
        s = report["modes"][mode]["summary"]
        print("%-9s mean %.3f | pass/partial/fail %d/%d/%d | traps declined %d/%d"
              % (mode, s["answerable_mean_score"], s["answerable_passed"],
                 s["answerable_partial"], s["answerable_failed"],
                 s["traps_correctly_declined"], s["trap_count"]))

    if write:
        report["rescored_with_current_grader"] = True
        with open(path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print()
        print("Written back to %s" % path)
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Re-grade a saved run offline.")
    ap.add_argument("path")
    ap.add_argument("--write", action="store_true", help="Save the corrected scores")
    a = ap.parse_args()
    rescore(a.path, write=a.write)
