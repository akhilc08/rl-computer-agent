"""Point-grounding training data for screenshot models (UI-Venus / Qwen3-VL) from the sandbox train split.

One row per screen: the screenshot and up to all of its Claude-drafted commands, each with its answer in the
Venus format: "[x,y]" in 0-1000 relative coordinates (centre of the first target box that lies on the image), or
"[-1,-1]" when the target is absent. Menu-bar targets (no box) and targets entirely off the image are dropped.
Only apps in the frozen train split are used (MacBench's test apps never appear); screens on the drop list are
skipped. Every command also carries "boxes": all its on-image target boxes as [x1,y1,x2,y2] in 0-1000 coords (the
RL reward, scripts/train/vlm_grpo.py, counts a point inside any of them). With --dev_apps, those train apps go to
--dev_out instead (a held-out dev set for RL checkpoint selection; keep them out of SFT too).

  python3 scripts/train/vlm_points_prep.py --out data/vlm_points/train.jsonl
  python3 scripts/train/vlm_points_prep.py --out data/vlm_rl/train.jsonl --dev_out data/vlm_rl/dev.jsonl --dev_apps RL_DEV
"""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DS = ROOT / "data/sandbox_ds"
LABELS = ("label_train2", "label_train3")
# RL dev apps: 4 native (2 AX-poor-heavy) + 4 web, from the train split, held out of SFT and RL training.
RL_DEV = ["Grapher", "Activity Monitor", "Inkscape", "Thunderbird",
          "web:imdb.com", "web:nasa.gov", "web:usps.com", "web:allrecipes.com"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "data/vlm_points/train.jsonl"))
    ap.add_argument("--dev_out")
    ap.add_argument("--dev_apps", nargs="*", default=[], help="train apps to hold out; RL_DEV = the fixed RL dev list")
    args = ap.parse_args()
    dev_apps = set(RL_DEV if args.dev_apps == ["RL_DEV"] else args.dev_apps)
    assert not dev_apps or args.dev_out, "--dev_apps needs --dev_out"
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
                boxes = []
            else:
                ans, boxes = None, []
                for t in c["target_boxes"]:
                    b = t.get("box")
                    if not b:
                        continue
                    x1, y1 = max(0, (b[0] - fx) * sc), max(0, (b[1] - fy) * sc)
                    x2, y2 = min(W, (b[0] + b[2] - fx) * sc), min(H, (b[1] + b[3] - fy) * sc)
                    if x2 > x1 and y2 > y1:
                        if ans is None:
                            ans = f"[{round((x1 + x2) / 2 / W * 1000)},{round((y1 + y2) / 2 / H * 1000)}]"
                        boxes.append([round(x1 / W * 1000, 1), round(y1 / H * 1000, 1),
                                      round(x2 / W * 1000, 1), round(y2 / H * 1000, 1)])
                if ans is None:
                    stats["no_box" if all(not t.get("box") for t in c["target_boxes"]) else "off_image"] += 1
                    continue
                stats["point"] += 1
            by_screen.setdefault(s["id"], {"screen": s["id"], "app": s["app"], "shot": s["shot"], "cmds": []})
            by_screen[s["id"]]["cmds"].append({"id": c["id"], "command": c["command"], "answer": ans,
                                                  "boxes": boxes})
    missing = dev_apps - {r["app"] for r in by_screen.values()}
    assert not missing, f"dev apps not in the train split: {missing}"
    parts = {args.out: [r for r in by_screen.values() if r["app"] not in dev_apps]}
    if args.dev_out:
        parts[args.dev_out] = [r for r in by_screen.values() if r["app"] in dev_apps]
    for path, rows in parts.items():
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        print(f"{out}: {len(rows)} screens, {len({r['app'] for r in rows})} apps, "
              f"{sum(len(r['cmds']) for r in rows)} commands")
    print(stats)


if __name__ == "__main__":
    main()
