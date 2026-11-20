"""Independent label check of MacBench with a non-Claude model (OpenAI, through the Codex CLI on the user's ChatGPT
login). The commands were drafted by Claude, so a different model family judges them.

One call per screen: the released (redacted) screenshot with every command's target boxes drawn and numbered,
plus the list of commands with their flags. The judge returns, per command, correct / wrong_target / bad_command
and a short reason. Disagreements go to a human. Resumable; runs --workers calls at once.

  python3 scripts/bench/macbench_crosscheck.py --bench data/macbench/v1 [--limit 5] [--workers 4]
  -> <bench>/crosscheck/verdicts.jsonl  {"screen", "id", "verdict", "reason"}
"""
import argparse
import json
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

COLORS = ["#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4", "#00a0a0", "#f032e6", "#808000", "#9a6324", "#000075"]
SCHEMA = {
    "type": "object", "additionalProperties": False, "required": ["verdicts"],
    "properties": {"verdicts": {"type": "array", "items": {
        "type": "object", "additionalProperties": False, "required": ["n", "verdict", "reason"],
        "properties": {"n": {"type": "integer"}, "verdict": {"type": "string", "enum": ["correct", "wrong_target", "bad_command"]},
                       "reason": {"type": "string"}}}}}}
PROMPT = """You are checking labels of a GUI grounding benchmark. The attached image is a macOS screenshot (a window, sometimes with the menu bar). Each command below was written for this screen, and its target element is drawn on the image as a coloured box labelled at its top-left corner with the number of every command that targets it (e.g. \"1,9\" when commands #1 and #9 share it). Some commands have several acceptable boxes, each carrying its number.

For each command decide:
- correct: a person giving this command on this screen would mean the boxed element (any one of its boxes is fine). For a command marked ABSENT (no box), correct means the thing it asks for really is not visible on this screen. For a MENU-BAR target (no box), correct means the named menu item fits the command.
- wrong_target: the command is sensible but the box points at the wrong element, or an ABSENT command's target is actually visible.
- bad_command: the command is unclear or unnatural, or a flag is clearly wrong (IRREVERSIBLE on an action that is easily undone or harmless, or vice versa; AMBIGUOUS on a command with one obvious target).
Be strict but fair: small box offsets are fine; casual wording is fine. Do not run any tools; just look at the image.
Return one verdict per command number, with a reason of at most 15 words.

Commands:
{cmds}"""


def draw(img_path, cmds, out):
    img = Image.open(img_path).convert("RGB")
    d = ImageDraw.Draw(img)
    size = max(14, img.width // 90)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf", size)
    except OSError:
        font = ImageFont.load_default()
    w = max(2, img.width // 700)
    boxes = {}                                    # one box drawn once, labelled with every command that targets it
    for n, c in enumerate(cmds, 1):
        for b in c["target_boxes"]:
            if b.get("box_px"):
                boxes.setdefault(tuple(round(v) for v in b["box_px"]), []).append(n)
    for k, ((x, y, bw, bh), ns) in enumerate(boxes.items()):
        col = COLORS[k % len(COLORS)]
        label = ",".join(map(str, ns))
        d.rectangle([x, y, x + bw, y + bh], outline=col, width=w)
        tw = d.textlength(label, font=font) + 4
        tx, ty = x, max(0, y - size - 2)
        d.rectangle([tx, ty, tx + tw, ty + size + 2], fill=col)
        d.text((tx + 2, ty), label, fill="white", font=font)
    img.save(out, quality=90)


def describe(n, c):
    flags = [f for f in ("absent", "ambiguous", "irreversible") if c[f]]
    s = f"#{n}: \"{c['command']}\" (action: {c['action']}"
    if c.get("text"):
        s += f", types \"{c['text']}\""
    s += ")"
    if c["absent"]:
        s += " ABSENT (no box)"
    elif all(not b.get("box_px") for b in c["target_boxes"]):
        s += " MENU-BAR target: " + "; ".join(b.get("label") or "" for b in c["target_boxes"])
    if c["ambiguous"]:
        s += " AMBIGUOUS"
    if c["irreversible"]:
        s += " IRREVERSIBLE"
    return s


def check(screen, cmds, bench, work, model):
    tmp = Path(tempfile.mkdtemp(dir=work))
    img = tmp / "screen.jpg"
    draw(bench / "images" / f"{screen}.jpg", cmds, img)
    (tmp / "schema.json").write_text(json.dumps(SCHEMA))
    prompt = PROMPT.format(cmds="\n".join(describe(n, c) for n, c in enumerate(cmds, 1)))
    cmd = ["codex", "exec", "--skip-git-repo-check", "--ephemeral", "-s", "read-only", "-i", str(img),
           "--output-schema", str(tmp / "schema.json"), "-o", str(tmp / "out.json"), "-C", str(tmp)]
    if model:
        cmd += ["-m", model]
    r = subprocess.run(cmd + ["-"], input=prompt, capture_output=True, text=True, timeout=900)
    try:
        v = json.loads((tmp / "out.json").read_text())["verdicts"]
    except (OSError, json.JSONDecodeError, KeyError):
        raise RuntimeError(f"{screen}: codex failed: {r.stderr[-400:]}")
    by_n = {x["n"]: x for x in v}
    return [{"screen": screen, "id": c["id"], "verdict": by_n.get(n, {}).get("verdict", "missing"),
             "reason": by_n.get(n, {}).get("reason", "")} for n, c in enumerate(cmds, 1)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", default="data/macbench/v1")
    ap.add_argument("--limit", type=int, default=0, help="first N screens")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--model", default="")
    args = ap.parse_args()
    bench = Path(args.bench)
    outdir = bench / "crosscheck"
    work = outdir / "work"
    work.mkdir(parents=True, exist_ok=True)
    out = outdir / "verdicts.jsonl"
    done = {json.loads(l)["screen"] for l in open(out)} if out.exists() else set()
    by_screen = {}
    for c in map(json.loads, open(bench / "commands.jsonl")):
        by_screen.setdefault(c["screen"], []).append(c)
    todo = [s for s in by_screen if s not in done][: args.limit or None]
    print(f"{len(todo)} screens to check ({len(done)} done)", flush=True)
    with open(out, "a") as f, ThreadPoolExecutor(args.workers) as ex:
        futs = {ex.submit(check, s, by_screen[s], bench, work, args.model): s for s in todo}
        for i, fu in enumerate(futs, 1):
            try:
                for row in fu.result():
                    f.write(json.dumps(row) + "\n")
                f.flush()
            except Exception as e:  # noqa: BLE001 - keep going, the screen is retried on the next run
                print(e, flush=True)
            if i % 10 == 0:
                print(f"{i}/{len(todo)}", flush=True)


if __name__ == "__main__":
    main()
