"""Per-question scores for a kev.benchmark run on our records, which Kev's own report lumps together.

Element accuracy counts any acceptable target (records list several when they lead to the same place).
Reports accuracy, calibration (ECE), and how much the gate could do alone at 90% / 95% precision, per
question, plus element accuracy by source, style and the ambiguous / absent flags.

  python3 scripts/eval/score_kev.py --rows runs/<eval>/<set>/rows.json --records data/records/<set>.jsonl
  python3 scripts/eval/score_kev.py --runs runs/<eval> --records_dir data/records   (every set, one table)
"""
import argparse
import json
import math
from collections import defaultdict
from pathlib import Path


def softmax(z):
    m = max(z)
    e = [math.exp(x - m) for x in z]
    s = sum(e)
    return [x / s for x in e]


def ece(confs, hits, bins=10):
    n, err = len(confs), 0.0
    for b in range(bins):
        idx = [i for i, c in enumerate(confs) if b / bins < c <= (b + 1) / bins or (b == 0 and c == 0)]
        if idx:
            err += len(idx) / n * abs(sum(confs[i] for i in idx) / len(idx) - sum(hits[i] for i in idx) / len(idx))
    return err


def coverage_at(confs, hits, precision):
    order = sorted(range(len(confs)), key=lambda i: -confs[i])
    best = k = 0
    for n, i in enumerate(order, start=1):
        k += hits[i]
        if k / n >= precision:
            best = n
    return best / len(order) if order else 0.0


def score(rows, records):
    meta = {r["_meta"]["id"]: r for r in records}
    by_q = defaultdict(lambda: ([], []))
    groups = defaultdict(list)
    for row in rows:
        rec = meta.get(row["id"])
        if rec is None or row.get("variant", "clean") != "clean":
            continue
        q = rec["questions"][row["question"]]
        keys = row["keys"]
        p = softmax(row["logits"]) if row.get("logits") else row["p"]
        top = max(range(len(p)), key=p.__getitem__)
        ok = set(q.get("target") or [q["label"] if q["type"] == "choice" else str(q["label"]).lower()])
        hit = keys[top] in ok
        by_q[row["question"]][0].append(p[top])
        by_q[row["question"]][1].append(hit)
        if row["question"] == "element":
            m = rec["_meta"]
            groups[("source", m.get("source"))].append(hit)
            if m.get("style"):
                groups[("style", m["style"])].append(hit)
            groups[("label", "none" if q["label"] == "none" else ("several targets" if q.get("target") else "one target"))].append(hit)
            amb = rec["questions"].get("ambiguous")
            if amb is not None:
                groups[("ambiguous", str(amb["label"]))].append(hit)
    out = {"questions": {}, "element_by": {}}
    for qid, (confs, hits) in by_q.items():
        out["questions"][qid] = {"n": len(hits), "acc": round(sum(hits) / len(hits), 3), "ece": round(ece(confs, hits), 3),
                                 "act_alone_at_90": round(coverage_at(confs, hits, 0.9), 3),
                                 "act_alone_at_95": round(coverage_at(confs, hits, 0.95), 3)}
    for (k, v), hits in sorted(groups.items()):
        out["element_by"][f"{k}={v}"] = {"n": len(hits), "acc": round(sum(hits) / len(hits), 3)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows")
    ap.add_argument("--records")
    ap.add_argument("--runs", help="an eval directory with one subdirectory per record set")
    ap.add_argument("--records_dir")
    a = ap.parse_args()
    jobs = [(Path(a.rows), Path(a.records))] if a.rows else \
        [(d / "rows.json", Path(a.records_dir) / f"{d.name}.jsonl") for d in sorted(Path(a.runs).iterdir()) if (d / "rows.json").exists()]
    summary = {}
    for rows_path, rec_path in jobs:
        rows = json.load(open(rows_path))
        records = [json.loads(l) for l in open(rec_path)]
        summary[rec_path.stem] = score(rows, records)
    print(f"{'set':16} {'question':13} {'n':>6} {'acc':>6} {'ece':>6} {'alone@90':>9} {'alone@95':>9}")
    for s, r in summary.items():
        for qid, m in r["questions"].items():
            print(f"{s:16} {qid:13} {m['n']:6} {m['acc']:6.3f} {m['ece']:6.3f} {m['act_alone_at_90']:9.3f} {m['act_alone_at_95']:9.3f}")
    for s, r in summary.items():
        print(f"\n{s} element accuracy by:")
        for k, m in r["element_by"].items():
            print(f"  {k:28} n={m['n']:5}  acc={m['acc']:.3f}")
    if a.runs:
        (Path(a.runs) / "scores.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
