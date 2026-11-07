"""One table of every model on every benchmark, from the run summaries copied back from the GPU boxes.

Reads runs/kev-box/kev_scores.json (Kev before/after training), runs/webagents (MindAct, WebLINX), runs/list-baselines
(small LLMs, model2vec) and runs/vlm (screenshot models), all scored on the fixed samples in data/records_s.
Prints markdown: element accuracy (%) per set, then median latency (ms) per set.

  python3 scripts/eval/results_table.py
"""
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "runs"
COLS = ["m2w_task", "m2w_website", "m2w_domain", "weblinx", "androidcontrol", "ssv2", "sspro", "osworldg", "mmbench",
        "omniact", "val", "heldout_site"]
HEAD = ["M2W task", "M2W web", "M2W domain", "WebLINX", "AndroidControl", "SS-v2", "SS-Pro", "OSWorld-G", "MMBench",
        "OmniACT", "Mac val", "NeetCode"]
ALIAS = {"screenspot_v2": "ssv2", "screenspot_pro": "sspro", "osworld_g": "osworldg", "mmbench_gui": "mmbench",
         "mac_val": "val", "neetcode": "heldout_site"}
ALIAS.update({f"eval_{c}": c for c in COLS})


def canon(s):
    return ALIAS.get(s, s)


def load():
    """-> {(group, model): {set: (accuracy, median_ms, n)}}"""
    out = {}
    kev = RUNS / "kev-box/kev_scores.json"
    if kev.exists():
        for k, v in json.load(open(kev)).items():
            run, s = k.split("/")
            name = {"eval-base-0.8b": "Kev 0.8B (before training)", "eval-sft-0.8b-v1": "Kev 0.8B (after training)"}[run]
            out.setdefault(("Kev", name), {})[canon(s)] = (v["element_accuracy"], None, v["n"])
    for p in sorted(RUNS.glob("webagents/*/*/summary.json")):
        m, s = p.parts[-3:-1]
        d = json.load(open(p))
        group = "WebLINX" if m.startswith("wl-") else "MindAct"
        out.setdefault((group, m), {})[canon(s)] = (d["element_accuracy"], d["latency_ms"]["median"], d["n"])
    for p in sorted(RUNS.glob("list-baselines/*/summary.json")):
        d = json.load(open(p))
        for s, v in d["per_set"].items():
            out.setdefault(("Small LLM / embedding, same list", d["model"].removeprefix("hf:")), {})[canon(s)] = (
                v["acc"], v.get("median_ms"), v["n"])
    for p in sorted(RUNS.glob("vlm/*/*/summary.json")):
        m, s = p.parts[-3:-1]
        d = json.load(open(p))
        out.setdefault(("Screenshot model", m), {})[canon(s)] = (d["accuracy"], d["latency_ms"]["median"], d["n"])
    return out


def table(rows, idx, fmt):
    lines = ["| Group | Model | " + " | ".join(HEAD) + " |", "|---|---|" + "---|" * len(HEAD)]
    for (g, m), sets in rows.items():
        cells = []
        for c in COLS:
            v = sets.get(c)
            cells.append(fmt(v[idx]) if v and v[idx] is not None else "–")
        lines.append(f"| {g} | {m} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main():
    order = ["Kev", "MindAct", "WebLINX", "Small LLM / embedding, same list", "Screenshot model"]
    rows = dict(sorted(load().items(), key=lambda kv: (order.index(kv[0][0]), kv[0][1])))
    print("### Element accuracy (%)\n")
    print(table(rows, 0, lambda x: f"{100 * x:.1f}"))
    print("\n### Median latency per decision (ms)\n")
    print(table(rows, 1, lambda x: f"{x:.0f}"))
    print("\n### n scored per set\n")
    print(table(rows, 2, lambda x: str(x)))
    med = {k: statistics.median([v[1] for v in s.values() if v[1] is not None] or [0]) for k, s in rows.items()}
    print("\nMedian of per-set medians (ms): " + ", ".join(f"{m} {v:.0f}" for (g, m), v in med.items() if v))


if __name__ == "__main__":
    main()
