"""Untrained Kev baseline on the synthetic element-picking set.

Runs each example twice (original and reversed option order) to measure order
sensitivity, and reports accuracy, calibration, selective accuracy and latency.

Run inside Kev's environment:
  .vendor/kev/.venv/bin/python scripts/baseline/run_kev_baseline.py --run jaredpalmer/kev-4b@v1.0 --name kev-4b-4bit
"""
import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from rlca.candidates import NONE_OPTION, MenuPrefilter, build_candidates, load_elements  # noqa: E402
from rlca.kev_backend import LAYOUTS, build_prompt, load_kev, score_options  # noqa: E402


def ece(confs, correct, bins=10):
    total, err = len(confs), 0.0
    for b in range(bins):
        idx = [i for i, c in enumerate(confs) if b / bins < c <= (b + 1) / bins or (b == 0 and c == 0)]
        if idx:
            err += len(idx) / total * abs(statistics.mean(confs[i] for i in idx) - statistics.mean(correct[i] for i in idx))
    return err


def coverage_at(confs, correct, target_acc):
    """Largest share of examples the gate could act on alone while keeping accuracy >= target."""
    order = sorted(range(len(confs)), key=lambda i: -confs[i])
    best, hits = 0.0, 0
    for n, i in enumerate(order, start=1):
        hits += correct[i]
        if hits / n >= target_acc:
            best = n / len(order)
    return best


def summarize(rows):
    correct = [r["correct"] for r in rows]
    confs = [r["p_top"] for r in rows]
    lat = sorted(r["ms"] for r in rows)
    out = {
        "n": len(rows),
        "accuracy": statistics.mean(correct),
        "accuracy_order_avg": statistics.mean(r["correct_avg"] for r in rows),
        "order_flip_rate": statistics.mean(r["flip"] for r in rows),
        "prefilter_recall": statistics.mean(r["target_in_candidates"] for r in rows if r["target_id"]),
        "ece": ece(confs, correct),
        "brier_top": statistics.mean((c - k) ** 2 for c, k in zip(confs, correct)),
        "coverage_at_95": coverage_at(confs, correct, 0.95),
        "coverage_at_90": coverage_at(confs, correct, 0.90),
        "latency_ms_p50": lat[len(lat) // 2],
        "latency_ms_p90": lat[int(len(lat) * 0.9)],
        "tokens_mean": statistics.mean(r["tokens"] for r in rows),
        "options_mean": statistics.mean(r["n_options"] for r in rows),
    }
    for key in ("style", "app"):
        groups = defaultdict(list)
        for r in rows:
            groups[r[key]].append(r["correct"])
        out[f"accuracy_by_{key}"] = {g: round(statistics.mean(v), 3) for g, v in sorted(groups.items())}
    hard = [r["correct"] for r in rows if r.get("hard")]
    out["accuracy_hard"] = statistics.mean(hard) if hard else None
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--data", default=str(ROOT / "data/synthetic/baseline_v0.jsonl"))
    ap.add_argument("--bits", type=int, default=4)
    ap.add_argument("--menu_k", type=int, default=40)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--layout", default="state_title", choices=LAYOUTS)
    args = ap.parse_args()

    examples = [json.loads(line) for line in open(args.data)]
    if args.limit:
        examples = examples[: args.limit]
    snapshots = {}
    prefilter = MenuPrefilter()
    tok, model = load_kev(args.run, bits=args.bits)
    import mlx.core as mx

    rows = []
    for n, ex in enumerate(examples, start=1):
        if ex["snapshot"] not in snapshots:
            snapshots[ex["snapshot"]] = json.loads((ROOT / "data/snapshots" / ex["snapshot"]).read_text())
        snap = snapshots[ex["snapshot"]]
        cands = build_candidates(ex["command"], load_elements(snap), prefilter, menu_k=args.menu_k)
        ids = [e.id for e in cands] + [None]
        options = [e.describe() for e in cands] + [NONE_OPTION]
        state, instr = build_prompt(snap["app"], snap["window_title"], ex["command"], args.layout)

        probs, ms, tokens = score_options(tok, model, state, options, instr)
        rev, _, _ = score_options(tok, model, state, options[::-1], instr)
        rev = rev[::-1]
        avg = [(a + b) / 2 for a, b in zip(probs, rev)]
        top = max(range(len(probs)), key=probs.__getitem__)
        top_rev = max(range(len(rev)), key=rev.__getitem__)
        top_avg = max(range(len(avg)), key=avg.__getitem__)
        rows.append({
            **{k: ex.get(k) for k in ("id", "app", "style", "target_id", "hard")},
            "pred_id": ids[top], "p_top": probs[top], "correct": ids[top] == ex["target_id"],
            "correct_avg": ids[top_avg] == ex["target_id"], "flip": top != top_rev,
            "target_in_candidates": ex["target_id"] in ids,
            "ms": ms, "tokens": tokens, "n_options": len(options),
        })
        if n % 25 == 0:
            print(f"{n}/{len(examples)} acc={statistics.mean(r['correct'] for r in rows):.3f} "
                  f"p50={sorted(r['ms'] for r in rows)[len(rows) // 2]:.0f}ms", flush=True)

    summary = summarize(rows)
    summary.update(run=args.run, bits=args.bits, menu_k=args.menu_k, layout=args.layout,
                   peak_memory_gb=mx.get_peak_memory() / 1e9)
    out = ROOT / "data/results" / args.name
    out.mkdir(parents=True, exist_ok=True)
    (out / "rows.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
