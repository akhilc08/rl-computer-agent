"""Point-grounding training data for screenshot models (UI-Venus / Qwen3-VL) from the sandbox train split.

One row per screen: the screenshot and up to all of its Claude-drafted commands, each with its answer in the
Venus format: "[x,y]" in 0-1000 relative coordinates (centre of the first target box that lies on the image), or
"[-1,-1]" when the target is absent. Menu-bar targets (no box) and targets entirely off the image are dropped.
Only apps in the frozen train split are used (MacBench's 23 test apps never appear); screens on the drop list are
skipped.

  python3 scripts/train/vlm_points_prep.py --out data/vlm_points/train.jsonl
"""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DS = ROOT / "data/sandbox_ds"
LABELS = ("label_train2", "label_train3")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "data/vlm_points/train.jsonl"))
    args = ap.parse_args()
    split = json.load(open(DS / "splits.json"))["apps"]
    drop = {l.split("\t")[0] for l in open(DS / "drop_screens.tsv") if l.strip()}
    screens = {s["id"]: s for s in map(json.loads, open(DS / "screens.jsonl"))}
    by_screen, stats = {}, {"absent": 0, "point": 0, "no_box": 0, "off_image": 0, "not_train": 0, "dropped": 0}
    for name in LABELS:
        for c in map(json.loads, open(DS / name / "commands.jsonl")):
            s = screens.get(c["screen"])
            if not s or split.get(s["app"]) != "train":
                stats["not_train"] += 1
                continue
            if s["id"] in drop:
                stats["dropped"] += 1
                continue
            W, H = s["img_size"]
            (fx, fy, _, _), sc = s["frame"], s["scale"]
            if c["absent"]:
                ans = "[-1,-1]"
                stats["absent"] += 1
            else:
                ans = None
                for t in c["target_boxes"]:
                    b = t.get("box")
                    if not b:
                        continue
                    x1, y1 = max(0, (b[0] - fx) * sc), max(0, (b[1] - fy) * sc)
                    x2, y2 = min(W, (b[0] + b[2] - fx) * sc), min(H, (b[1] + b[3] - fy) * sc)
                    if x2 > x1 and y2 > y1:
                        ans = f"[{round((x1 + x2) / 2 / W * 1000)},{round((y1 + y2) / 2 / H * 1000)}]"
                        break
                if ans is None:
                    stats["no_box" if all(not t.get("box") for t in c["target_boxes"]) else "off_image"] += 1
                    continue
                stats["point"] += 1
            by_screen.setdefault(s["id"], {"screen": s["id"], "app": s["app"], "shot": s["shot"], "cmds": []})
            by_screen[s["id"]]["cmds"].append({"id": c["id"], "command": c["command"], "answer": ans})
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for r in by_screen.values():
            f.write(json.dumps(r) + "\n")
    print(f"{len(by_screen)} screens, {len({r['app'] for r in by_screen.values()})} apps, {stats}")


if __name__ == "__main__":
    main()
