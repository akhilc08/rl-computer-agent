"""Turn the raw EC2 Mac sandbox runs into a clean screen set with an app-level split.

  .venv/bin/python scripts/data/sandbox_dataset.py clean   # data/sandbox_aws/*/*  ->  data/sandbox_ds/screens.jsonl
  .venv/bin/python scripts/data/sandbox_dataset.py split   # -> data/sandbox_ds/splits.json (frozen, by app/site)

clean: one row per screenshot (states that share a screenshot are the same screen). Drops blank or
near-uniform frames, then near-duplicates inside each app: same window size and colour mode, 256-bit
difference hash within HAMMING bits, and AX (role, label) sets at least JACCARD alike. Each kept screen
carries its AX elements as ground-truth targets (source "ax"), the detector boxes the collector clicked
on it (source "det", no label yet), and an ax_poor flag (fewer than AX_POOR labelled window elements).

split: whole apps/sites go to train/val/test, never single screens. Apps are shuffled with a fixed seed
inside four strata (web/native x ax-poor/ok) so AX-poor apps land in every split.
"""
import argparse
import collections
import hashlib
import json
import random
import re
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data/sandbox_aws"
OUT = ROOT / "data/sandbox_ds"
HAMMING, JACCARD, AX_POOR, MIN_STD = 12, 0.8, 8, 1.0
SPLIT = {"train": 0.7, "val": 0.1, "test": 0.2}


def dhash(img, n=16):
    g = np.asarray(img.convert("L").resize((n + 1, n), Image.BILINEAR), dtype=np.int16)
    return np.packbits((g[:, 1:] > g[:, :-1]).ravel())


def hamming(a, b):
    return int(np.unpackbits(a ^ b).sum())


def app_key(s):
    return s["target"] if s["target"].startswith("web:") else s["app"]


def condition(s, modes):
    m = re.search(r"mode=(\w+)&size=(\d+x\d+)", s.get("url") or "") or re.search(r"mode=(\w+)&size=(\d+x\d+)", modes.get(s["id"], ""))
    return (m.group(1), m.group(2)) if m else ("light", f"{s['frame'][2]}x{s['frame'][3]}")


def load_runs():
    for states in sorted(RAW.glob("*/*/states.jsonl")):
        run = states.parent
        modes, det = {}, collections.defaultdict(list)
        if (run / "transitions.jsonl").exists():
            for line in open(run / "transitions.jsonl"):
                t = json.loads(line)
                modes[t["before"]] = t.get("mode") or ""
                if t["kind"] == "pixel_click" and t.get("box"):
                    det[t["before"]].append(t["box"])
        byshot = {}
        for line in open(states):
            s = json.loads(line)
            row = byshot.setdefault(s["shot"], {**s, "det": []})
            row["det"] += [b for b in det.get(s["id"], []) if b not in row["det"]]
            row.setdefault("_modes", modes)
        yield run, byshot.values()


def clean():
    OUT.mkdir(parents=True, exist_ok=True)
    stats = collections.Counter()
    kept = collections.defaultdict(list)                  # (app, mode, size) -> [(hash, axset)]
    rows = []
    for run, states in load_runs():
        rel = run.relative_to(RAW)
        for s in states:
            stats["shots"] += 1
            path = run / s["shot"]
            try:
                img = Image.open(path)
                img.load()
            except OSError:
                stats["unreadable"] += 1
                continue
            if float(np.asarray(img.convert("L").resize((256, 160))).std()) < MIN_STD:
                stats["blank"] += 1
                continue
            mode, size = condition(s, s.pop("_modes"))
            app = app_key(s)
            axset = {(e["role"], e["label"]) for e in s["elements"] if e["label"]}
            h = dhash(img)
            near = False
            for h2, ax2 in kept[(app, mode, size)]:
                if hamming(h, h2) <= HAMMING and len(axset & ax2) >= JACCARD * max(1, len(axset | ax2)):
                    near = True
                    break
            if near:
                stats["near_duplicate"] += 1
                continue
            kept[(app, mode, size)].append((h, axset))
            labelled = [e for e in s["elements"] if e["label"]]
            sid = hashlib.sha1(f"{rel}/{s['shot']}".encode()).hexdigest()[:12]
            rows.append({
                "id": sid, "app": app, "kind": "web" if app.startswith("web:") else "native",
                "title": s.get("title"), "url": s.get("url"), "mode": mode, "size": size,
                "shot": str(path.relative_to(ROOT)), "img_size": list(img.size), "scale": s.get("scale"),
                "frame": s["frame"], "ax_poor": len(labelled) < AX_POOR, "n_ax": len(s["elements"]),
                "ms": s.get("ms"),
                "targets": [{"id": e["id"], "source": "ax", "role": e["role"], "label": e["label"],
                             "context": e.get("context", []), "enabled": e.get("enabled", True), "box": e["frame"]}
                            for e in s["elements"]] +
                           [{"id": f"m{i}", "source": "ax_menubar", "role": e["role"], "label": e["label"],
                             "context": e.get("context", []), "enabled": e.get("enabled", True), "box": e.get("frame")}
                            for i, e in enumerate(s.get("menubar", []))] +
                           [{"id": f"k{i}", "source": "det_clicked", "role": "", "label": "", "box": b}
                            for i, b in enumerate(s["det"])],
            })
            stats["kept"] += 1
    with open(OUT / "screens.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    apps = collections.Counter(r["app"] for r in rows)
    stats["apps"] = len(apps)
    stats["ax_poor"] = sum(r["ax_poor"] for r in rows)
    print(json.dumps(stats, indent=1))
    print("screens per app: min", min(apps.values()), "median", sorted(apps.values())[len(apps) // 2], "max", max(apps.values()))


def split(seed=0):
    rows = [json.loads(l) for l in open(OUT / "screens.jsonl")]
    if (OUT / "splits.json").exists():
        raise SystemExit("splits.json already exists; splits are frozen. Delete it on purpose to re-split.")
    by_app = collections.defaultdict(list)
    for r in rows:
        by_app[r["app"]].append(r)
    strata = collections.defaultdict(list)
    for app, rs in by_app.items():
        poor = sum(r["ax_poor"] for r in rs) / len(rs) >= 0.5
        strata[(rs[0]["kind"], poor)].append(app)
    rng = random.Random(seed)
    assign = {}
    for key in sorted(strata):
        apps = sorted(strata[key])
        rng.shuffle(apps)
        n = len(apps)
        cut1, cut2 = round(n * SPLIT["train"]), round(n * (SPLIT["train"] + SPLIT["val"]))
        for i, app in enumerate(apps):
            assign[app] = "train" if i < cut1 else "val" if i < cut2 else "test"
    json.dump({"seed": seed, "rule": "by app/site, stratified web/native x ax-poor", "apps": dict(sorted(assign.items()))},
              open(OUT / "splits.json", "w"), indent=1)
    c = collections.Counter()
    for r in rows:
        c[(assign[r["app"]], "screens")] += 1
        c[(assign[r["app"]], "ax_poor")] += r["ax_poor"]
    for sp in SPLIT:
        print(sp, sum(v == sp for v in assign.values()), "apps", c[(sp, "screens")], "screens", c[(sp, "ax_poor")], "ax-poor")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["clean", "split"])
    a = ap.parse_args()
    clean() if a.step == "clean" else split()
