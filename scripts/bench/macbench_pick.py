"""Pick test-app screens for a MacBench labelling round and lay them out like sandbox_label.py prep.

Spreads ~N screens evenly over every test app (small apps give all they have), alternates hard (AX-poor or 3+
detector-only icons) and ordinary screens, alternates light/dark, and drops near-duplicates: screens of the same
app whose option labels overlap by Jaccard >= --dup (light/dark twins of one state, re-captures). Skips screens
already in any label*/chosen.json, the drop list, and anything showing machine identity (About This Mac,
serial numbers, hardware UUIDs).

  .venv/bin/python scripts/bench/macbench_pick.py --n 1000 --dir label_test
  -> data/sandbox_ds/label_test/{screens/<id>.txt,.jpg, batches/batch_<k>.txt, parts/, chosen.json}
Label the batches with the same prompt as label_train*, then: sandbox_label.py check --dir label_test
"""
import argparse
import collections
import json
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/sandbox"))
import sandbox_label as sl  # noqa: E402

IDENTITY = re.compile(r"about this mac|serial number|hardware uuid|provisioning udid|ec2-\d|\.ec2\.internal", re.I)


def labels(o):
    return {x["label"].lower() for x in o["options"] if x["label"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--dir", default="label_test")
    ap.add_argument("--split", default="test")
    ap.add_argument("--batch", type=int, default=25)
    ap.add_argument("--dup", type=float, default=0.9)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--apps", nargs="*", help="only these apps/sites (e.g. val-split sites promoted to the test set)")
    args = ap.parse_args()
    rng = random.Random(args.seed)
    screens, opts, split = sl.load()
    drop = {l.split("\t")[0] for l in open(sl.DS / "drop_screens.tsv") if l.strip()}
    taken = set()
    for f in sl.DS.glob("label*/chosen.json"):
        taken |= set(json.load(open(f)))
    skipped = collections.Counter()
    pools = collections.defaultdict(list)
    for sid, s in screens.items():
        if split.get(s["app"]) != args.split or (args.apps and s["app"] not in args.apps):
            continue
        o = opts.get(sid)
        if not o or not o["options"]:
            skipped["no options"] += 1
        elif sid in taken:
            skipped["already labelled"] += 1
        elif sid in drop:
            skipped["drop list"] += 1
        elif IDENTITY.search(" ".join([s.get("title") or "", *labels(o)])):
            skipped["machine identity"] += 1
        else:
            pools[s["app"]].append(sid)
    # Order each app's pool hard/easy and light/dark alternating, then greedily drop near-duplicates.
    apps = sorted(pools)
    per_app = {}
    for app in apps:
        groups = collections.defaultdict(list)
        for sid in pools[app]:
            s = screens[sid]
            groups[(s["ax_poor"] or sl.n_icons(opts[sid]) >= 3, s["mode"])].append(sid)
        for g in groups.values():
            rng.shuffle(g)
        order, keys = [], sorted(groups)
        while any(groups.values()):
            for k in keys:
                if groups[k]:
                    order.append(groups[k].pop())
        kept, seen = [], []
        for sid in order:
            lab = labels(opts[sid])
            if any(lab and (len(lab & l2) / len(lab | l2)) >= args.dup for l2 in seen):
                skipped["near-duplicate"] += 1
                continue
            kept.append(sid)
            seen.append(lab)
        per_app[app] = kept
    # Even spread: raise a per-app cap until the total reaches N.
    cap = 1
    while sum(min(len(v), cap) for v in per_app.values()) < args.n and cap < max(map(len, per_app.values())):
        cap += 1
    chosen = [sid for app in apps for sid in per_app[app][:cap]][: args.n]
    lab_dir = sl.DS / args.dir
    for sub in ("screens", "batches", "parts"):
        (lab_dir / sub).mkdir(parents=True, exist_ok=True)
    for sid in chosen:
        (lab_dir / "screens" / f"{sid}.txt").write_text(sl.option_text(screens[sid], opts[sid]))
        sl.picture(screens[sid], opts[sid], lab_dir / "screens" / f"{sid}.jpg")
    rng.shuffle(chosen)
    for k in range(0, len(chosen), args.batch):
        (lab_dir / "batches" / f"batch_{k // args.batch}.txt").write_text("\n".join(chosen[k:k + args.batch]) + "\n")
    json.dump({sid: args.split for sid in chosen}, open(lab_dir / "chosen.json", "w"), indent=0)
    by_app = collections.Counter(screens[s]["app"] for s in chosen)
    print(f"{len(chosen)} screens from {len(by_app)} apps (cap {cap}/app), "
          f"{(len(chosen) + args.batch - 1) // args.batch} batches; skipped {dict(skipped)}")
    print(dict(sorted(by_app.items())))


if __name__ == "__main__":
    main()
