"""Build Kev training and eval records from our labelled Mac commands and the converted public sets.

Mac labels (data/labels/commands.jsonl + parts/part_*.jsonl, with parts/review_*.jsonl corrections) get every question: element, action, text
source, irreversible, ambiguous. Option lists follow serving: all window/page items, then the menu-bar
items the prefilter ranks highest for that command, then "none". Public sets get element and action only.

Splits:
  train            Mac labels + OS-Atlas
  val              Mac labels only (~10% of screens, by screen)
  heldout_site     every Mac command on the --heldout_sites sites (never trained on; the generalisation test)
  eval_ssv2 / eval_sspro / eval_osworldg / eval_mmbench   the standard grounding benchmarks
  eval_omniact     all of OmniACT (native Mac apps); its splits share screens, so none of it is trained on
  eval_synthetic   the 300 synthetic commands the untrained baseline was first scored on
  train_mix        train with --cap applied and Mac labels repeated --mac_repeat times (what the run trains on)

Training screens that look like any benchmark screen (perceptual hash within --dup_bits of 256) are dropped,
so a near-copy of a benchmark screenshot can't leak in through another dataset (OS-Atlas shares screens with
ScreenSpot-v2).

Examples whose target didn't make it into the option list (menu prefilter or the 254-option cap) are
dropped from training and counted, since Kev can't get them right however it's trained.

  .venv/bin/python scripts/train/build_records.py [--heldout_sites neetcode]
"""
import argparse
import glob
import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from rlca.candidates import MAX_OPTIONS, MenuPrefilter  # noqa: E402
from rlca.elements import Element  # noqa: E402
from rlca.records import record  # noqa: E402

PUB = ROOT / "data/public"

OUT = ROOT / "data/records"
MENU_K = 40
PUBLIC_ACTIONS = {"click": "click", "double_click": "click", "right_click": "click", "type": "type"}


def frac(key, salt="val"):
    return int(hashlib.sha256(f"{salt}:{key}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF


def mac_options(screen, command, prefilter):
    """Serving's option list for one command: page/window items, then the top MENU_K menu items."""
    page = [o for o in screen["options"] if o["source"] != "menubar"]
    menu = [o for o in screen["options"] if o["source"] == "menubar"]
    if menu:
        els = [Element(id=o["id"], source="menubar", role=o["role"], label=o["label"], context=o["context"]) for o in menu]
        scores = prefilter.scores(command, els)
        menu = [menu[i] for i in sorted(range(len(menu)), key=lambda i: -scores[i])[:MENU_K]]
    return (page + menu)[: MAX_OPTIONS - 1]


def mac_records(heldout_sites, prefilter):
    screens = {json.loads(l)["id"]: json.loads(l) for l in open(ROOT / "data/labels/screens.jsonl")}
    files = [ROOT / "data/labels/commands.jsonl", *sorted(glob.glob(str(ROOT / "data/labels/parts/part_*.jsonl")))]
    # Audit corrections (parts/review_*.jsonl) replace the original row with the same id.
    fixes = {}
    for f in sorted(glob.glob(str(ROOT / "data/labels/parts/review_*.jsonl"))):
        for line in open(f):
            r = json.loads(line)
            fixes[r["id"]] = r
    for f in files:
        for line in open(f):
            c = json.loads(line)
            c = fixes.get(c["id"], c)
            s = screens[c["screen"]]
            opts = mac_options(s, c["command"], prefilter)
            ids = {o["id"] for o in opts}
            targets = [t for t in c["targets"] if t in ids]
            if c["targets"] and not targets:
                yield "dropped_prefilter", None
                continue
            site = s["site"] or s["app"]
            split = "heldout_site" if site in heldout_sites else ("val" if frac(s["id"]) < 0.1 else "train")
            text_source = c.get("text_source") or "none"
            yield split, record(s["app"], c["command"], opts, targets, action=c["verb"], text_source=text_source,
                                irreversible=c["irreversible"], ambiguous=c["ambiguous"],
                                meta={"source": "mac_labels", "id": c["id"], "screen": s["id"], "site": site, "style": c["style"]})


def synthetic_records(prefilter):
    """The 300-command synthetic set the untrained baseline was scored on, as eval records (before/after on the same set)."""
    for line in open(ROOT / "data/synthetic/baseline_v0.jsonl"):
        ex = json.loads(line)
        snap = json.load(open(ROOT / "data/snapshots" / ex["snapshot"]))
        screen = {"options": [{**e, "box": e.get("frame")} for e in snap["elements"] if e["source"] == "window" or e.get("enabled", True)]}
        opts = mac_options(screen, ex["command"], prefilter)
        targets = [ex["target_id"]] if ex.get("target_id") else []
        if targets and targets[0] not in {o["id"] for o in opts}:
            yield "dropped_prefilter", None
            continue
        yield "eval_synthetic", record(snap["app"], ex["command"], opts, targets, action=ex.get("verb"),
                                       meta={"source": "synthetic_v0", "id": ex["id"], "style": ex.get("style")})


def mix(train, caps, mac_repeat, seed=0):
    """Training mix: cap the big literal-instruction sets so they don't drown the Mac labels, and repeat the Mac labels."""
    rng = random.Random(seed)
    by_src = defaultdict(list)
    for r in train:
        by_src[r["_meta"]["source"]].append(r)
    out = []
    for src, recs in by_src.items():
        if src in caps and len(recs) > caps[src]:
            recs = rng.sample(recs, caps[src])
        out += recs * (mac_repeat if src == "mac_labels" else 1)
    rng.shuffle(out)
    return out


BENCHMARKS = {"screenspot_v2": "eval_ssv2", "screenspot_pro": "eval_sspro", "osworld_g": "eval_osworldg",
              "mmbench_gui": "eval_mmbench", "omniact": "eval_omniact", "androidcontrol": "eval_androidcontrol"}
TRAIN_SETS = ("os_atlas_macos", "os_atlas_linux")


def dhash(path, n=16):
    """256-bit difference hash of a screenshot: near-identical screens differ in a few bits."""
    from PIL import Image
    im = Image.open(path).convert("L").resize((n + 1, n), Image.BILINEAR)
    px = im.tobytes()
    return sum(1 << i for i in range(n * n) if px[(i // n) * (n + 1) + i % n] > px[(i // n) * (n + 1) + i % n + 1])


def screen_hashes(name):
    """{image: hash} for one converted set, cached next to it."""
    d = ROOT / "data/converted" / name
    cache = d / "dhash.json"
    hashes = json.loads(cache.read_text()) if cache.exists() else {}
    for line in open(d / "screens.jsonl"):
        img = json.loads(line)["image"]
        if img not in hashes:
            try:
                hashes[img] = dhash(PUB / img)
            except Exception:       # truncated / unreadable screenshots
                hashes[img] = None
    cache.write_text(json.dumps(hashes))
    return hashes


def benchmark_lookalikes(max_bits):
    """Training-set images within max_bits of any benchmark screen."""
    bench = [h for name in BENCHMARKS if (ROOT / "data/converted" / name / "screens.jsonl").exists()
             for h in screen_hashes(name).values() if h is not None]
    out = set()
    for name in TRAIN_SETS:
        for img, h in screen_hashes(name).items():
            if h is not None and any(bin(h ^ b).count("1") <= max_bits for b in bench):
                out.add(img)
    return out


def public_records(dup_bits, none_examples=True):
    lookalikes = benchmark_lookalikes(dup_bits)
    for name in (*BENCHMARKS, *TRAIN_SETS):
        d = ROOT / "data/converted" / name
        if not (d / "examples.jsonl").exists():
            continue
        screens = {json.loads(l)["image"]: json.loads(l) for l in open(d / "screens.jsonl")}
        for line in open(d / "examples.jsonl"):
            ex = json.loads(line)
            if name in TRAIN_SETS and ex["image"] in lookalikes:
                yield "dropped_benchmark_lookalike", None
                continue
            source = name
            if not ex["target_ids"] and name not in BENCHMARKS:
                # The pixel sources missed the target, so given this list the right answer is "none". Train on these
                # (capped via <set>_none in --cap): without them Kev never answers "none" on public screens.
                if not none_examples:
                    yield "dropped_unmatched", None
                    continue
                source = f"{name}_none"
            opts = screens[ex["image"]]["elements"][: MAX_OPTIONS - 1]
            ids = {o["id"] for o in opts}
            targets = [t for t in ex["target_ids"] if t in ids]
            if ex["target_ids"] and not targets:
                yield "dropped_cap", None
                continue
            action = PUBLIC_ACTIONS.get(ex.get("action"))
            if action is None:
                yield "dropped_action", None
                continue
            split = BENCHMARKS.get(name, "train")
            app = ex.get("app") or ex.get("platform") or name
            yield split, record(app, ex["instruction"], opts, targets, action=action,
                                text_source=("user" if action == "type" else None),
                                meta={"source": source, "id": ex["id"], "split": ex["split"],
                                      **({"platform": ex["platform"]} if ex.get("platform") else {})})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--heldout_sites", default="neetcode", help="comma-separated Mac sites/apps kept out of training")
    ap.add_argument("--cap", default="os_atlas_macos=6000,os_atlas_linux=6000,os_atlas_macos_none=1000,os_atlas_linux_none=1000", help="per-source caps for train_mix.jsonl")
    ap.add_argument("--dup_bits", type=int, default=6, help="hash distance at or under which a training screen counts as a benchmark copy")
    ap.add_argument("--mac_repeat", type=int, default=2, help="times each Mac label appears in train_mix.jsonl")
    args = ap.parse_args()
    held = set(filter(None, args.heldout_sites.split(",")))
    OUT.mkdir(parents=True, exist_ok=True)
    counts = Counter()
    by_split = defaultdict(list)
    prefilter = MenuPrefilter()
    for split, rec in [*mac_records(held, prefilter), *synthetic_records(prefilter), *public_records(args.dup_bits)]:
        counts[split] += 1
        if rec is not None:
            by_split[split].append(rec)
    for split, recs in by_split.items():
        with open(OUT / f"{split}.jsonl", "w") as f:
            for r in recs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    caps = {k: int(v) for k, v in (kv.split("=") for kv in filter(None, args.cap.split(",")))}
    train_mix = mix(by_split["train"], caps, args.mac_repeat)
    with open(OUT / "train_mix.jsonl", "w") as f:
        for r in train_mix:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    by_split["train_mix"] = train_mix
    src = Counter((s, r["_meta"]["source"]) for s, recs in by_split.items() for r in recs)
    for (s, name), n in sorted(src.items()):
        print(f"{s:14s} {name:16s} {n:6d}")
    print({k: v for k, v in counts.items() if k.startswith("dropped")})


if __name__ == "__main__":
    main()
