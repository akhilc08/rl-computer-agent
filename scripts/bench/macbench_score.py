"""Score predictions on MacBench, for list-picking models (option ids) or screenshot models (points).

Every command in the track counts. A command with no prediction counts as wrong, so a list model is charged for
targets its list does not contain (1,548 pixels-track commands) and point and list models share one denominator.

Predictions, one JSON object per line, keyed by command id:
  {"id": "<command id>", "prediction": {"probabilities": {"element": {"<option id>": p, ...}}}}   (Kev/Jev format)
  {"id": "<command id>", "option": "<option id>" | "none"}
  {"id": "<command id>", "point": [x, y]} | {"id": ..., "point": null}       (image pixels; null = none)

  python scripts/bench/macbench_score.py --bench data/macbench/v1 --track ax --pred runs/x/predictions.jsonl [--json out.json]
"""
import argparse
import json
from pathlib import Path

KINDS = ("ax", "ocr", "icon", "menu", "none")


def answer(p, cmd):
    """(answer, confidence): answer is an option id, "none", or None for no prediction."""
    if p is None:
        return None, None
    if "point" in p:
        if p["point"] is None:
            return "none", None
        x, y = p["point"]
        hit = any(b and b[0] <= x <= b[0] + b[2] and b[1] <= y <= b[1] + b[3]
                  for b in (t.get("box_px") for t in cmd["target_boxes"]))
        return ("hit" if hit else "miss"), None
    if "option" in p:
        return p["option"], None
    probs = (p.get("prediction") or {}).get("probabilities", {}).get("element") or {}
    if not probs:
        return None, None
    k = max(probs, key=probs.get)
    return k, probs[k]


def correct(ans, cmd):
    if ans is None:
        return False
    if cmd["absent"]:
        return ans == "none"
    return ans == "hit" or ans in cmd["targets"]


def pct(a, b):
    return round(100 * a / b, 1) if b else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", default="data/macbench/v1")
    ap.add_argument("--track", choices=["ax", "pixels"], required=True)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--json")
    args = ap.parse_args()
    cmds = [c for c in map(json.loads, open(Path(args.bench) / "commands.jsonl")) if args.track in c["tracks"]]
    preds = {p["id"]: p for p in map(json.loads, open(args.pred))}
    tally = {k: [0, 0] for k in ("all", *KINDS, "ambiguous", "irreversible")}
    false_abstain = [0, 0]
    confident = [0, 0]                       # answers at >= 0.9 confidence: wrong, total
    covered = 0
    for c in cmds:
        p = preds.get(c["id"])
        covered += p is not None
        ans, conf = answer(p, c)
        ok = correct(ans, c)
        for k in ("all", c["target_kind"]) + (("ambiguous",) if c["ambiguous"] else ()) + (("irreversible",) if c["irreversible"] else ()):
            tally[k][0] += ok
            tally[k][1] += 1
        if not c["absent"]:
            false_abstain[0] += ans == "none"
            false_abstain[1] += 1
        if conf is not None and conf >= 0.9:
            confident[0] += not ok
            confident[1] += 1
    out = {"track": args.track, "commands": len(cmds), "with_prediction": covered,
           "accuracy": {k: pct(*v) for k, v in tally.items()}, "n": {k: v[1] for k, v in tally.items()},
           "false_abstain": pct(*false_abstain), "error_at_conf_0.9": pct(*confident), "n_conf_0.9": confident[1]}
    print(json.dumps(out, indent=1))
    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
