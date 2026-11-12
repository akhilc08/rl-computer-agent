"""Assemble the public MacBench release folder (what goes on Hugging Face) from a build.

Copies the screenshots, screens.jsonl, commands.jsonl, the two ready-made lists, the scoring script and the
labelling guide; writes README.md (the dataset card, from macbench_card.md with this build's counts), DATASHEET.md
and LICENSE. Leaves out the internal privacy/ and review/ folders. Refuses to run while the build has open privacy
flags of a blocking type or (unless --allow_unverified) unverified commands.

  .venv/bin/python scripts/bench/macbench_release.py --bench data/macbench/v1 --out data/macbench/release/v1
"""
import argparse
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
BLOCKING = ("uuid", "ec2_host", "ec2_instance")      # machine identifiers must be gone before release


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", default=str(ROOT / "data/macbench/v1"))
    ap.add_argument("--out", default=str(ROOT / "data/macbench/release/v1"))
    ap.add_argument("--allow_unverified", action="store_true", help="for a dry run before the human check")
    args = ap.parse_args()
    bench, out = Path(args.bench), Path(args.out)
    stats = json.load(open(bench / "stats.json"))
    hits = json.load(open(bench / "privacy/summary.json"))["hits"]
    open_flags = {k: hits[k] for k in BLOCKING if hits.get(k)}
    if open_flags:
        raise SystemExit(f"privacy flags still open: {open_flags}; rebuild with redaction and rescan")
    cmds = [json.loads(l) for l in open(bench / "commands.jsonl")]
    verified = sum(c["verified"] for c in cmds)
    if verified < len(cmds) and not args.allow_unverified:
        raise SystemExit(f"{len(cmds) - verified} of {len(cmds)} commands unverified; run the human check first")
    if out.exists():
        shutil.rmtree(out)
    (out / "tools").mkdir(parents=True)
    shutil.copytree(bench / "images", out / "images")
    for f in ("screens.jsonl", "commands.jsonl", "list_ax.jsonl", "list_pixels.jsonl", "stats.json"):
        shutil.copyfile(bench / f, out / f)
    shutil.copyfile(HERE / "macbench_score.py", out / "tools/score.py")
    shutil.copyfile(ROOT / "scripts/labels/label_guide.txt", out / "tools/label_guide.txt")
    fields = dict(stats, verified=verified, unverified=len(cmds) - verified)
    for name in ("macbench_card.md", "macbench_datasheet.md"):
        text = (HERE / name).read_text()
        for k, v in fields.items():
            text = text.replace("{" + k + "}", f"{v:,}" if isinstance(v, int) else str(v))
        (out / ("README.md" if name == "macbench_card.md" else "DATASHEET.md")).write_text(text)
    (out / "LICENSE").write_text(
        "MacBench labels, metadata and option lists: Creative Commons Attribution-NonCommercial 4.0 International\n"
        "(CC BY-NC 4.0), https://creativecommons.org/licenses/by-nc/4.0/\n\n"
        "tools/score.py: MIT License.\n\n"
        "Screenshots show Apple's macOS interface and third-party websites. Those rights stay with their owners;\n"
        "the screenshots are included for non-commercial research use only.\n")
    n = sum(1 for _ in (out / "images").iterdir())
    print(f"release at {out}: {n} images, {len(cmds)} commands ({verified} verified)")


if __name__ == "__main__":
    main()
