"""Score a MacBench review export: label accuracy with a 95% Wilson interval, overall and per target kind.

Accuracy = correct / (correct + wrong_target + bad_command); skips are left out. With --apply, writes
<bench>/review/verdicts.jsonl (one row per judged command) for the build to mark reviewed commands.

  python scripts/bench/macbench_review_merge.py ~/Downloads/macbench-review-v1.json --bench data/macbench/v1
"""
import argparse
import collections
import json
import math
from pathlib import Path


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"),) * 3
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0.0, c - h), min(1.0, c + h)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("export")
    ap.add_argument("--bench", help="benchmark folder, to check ids and (with --apply) write verdicts.jsonl")
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()
    v = json.load(open(args.export))["verdicts"]
    judged = {i: x for i, x in v.items() if x.get("verdict")}
    if args.bench:
        ids = {json.loads(l)["id"] for l in open(Path(args.bench) / "commands.jsonl")}
        unknown = set(judged) - ids
        if unknown:
            print(f"warning: {len(unknown)} ids not in {args.bench}/commands.jsonl")
    counts = collections.defaultdict(collections.Counter)
    for x in judged.values():
        for key in ("all", x.get("kind", "?")):
            counts[key][x["verdict"]] += 1
    print(f"{'kind':8s} {'n':>4s} {'acc':>6s}  95% CI          wrong_target bad_command skip")
    for key in ["all"] + sorted(k for k in counts if k != "all"):
        c = counts[key]
        n = c["correct"] + c["wrong_target"] + c["bad_command"]
        p, lo, hi = wilson(c["correct"], n)
        print(f"{key:8s} {n:4d} {p:6.1%}  [{lo:.1%}, {hi:.1%}]  {c['wrong_target']:12d} {c['bad_command']:11d} {c['skip']:4d}")
    notes = [(i, x["note"]) for i, x in judged.items() if x.get("note")]
    if notes:
        print(f"\n{len(notes)} notes:")
        for i, n in notes[:30]:
            print(f"  {i}: {n}")
    if args.apply and args.bench:
        out = Path(args.bench) / "review" / "verdicts.jsonl"
        with open(out, "w") as f:
            for i, x in judged.items():
                f.write(json.dumps({"id": i, "verdict": x["verdict"], "note": x.get("note", "")}) + "\n")
        print("wrote", out)


if __name__ == "__main__":
    main()
