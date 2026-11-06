"""WebLINX test splits -> Kev eval records, with element lists from WebLINX's own released candidate ranking.

WebLINX (McGill-NLP/WebLINX-full) is conversational web navigation: an instructor chats with a navigator who acts in
the browser. Its papers report five test splits, test_iid plus four out-of-domain ones (test_web: unseen sites,
test_vis: instructor can't see the screen, test_geo: unseen regions, test_cat: unseen site categories), all here.
Turns: the turns WebLINX's own eval keeps (the rows of WebLINX-full data/<split>.csv, built by their
select_turns_and_candidates_for_prompts: good screenshot, element present) whose action targets an element: click,
text_input, change (a <select>) and submit (an element action too; it counts toward WebLINX's element score). A fixed
random sample of --sample turns per split (seed 0).

Element list: WebLINX's released Dense Markup Ranking candidates (candidates/<split>.jsonl, McGill-NLP/MiniLM-L6-dmr
scores over every element of the page), sorted by rank and de-duplicated by uid exactly as their
select_candidates_for_turn does. Kev and MindAct pick from the top --top_k (50, as for Mind2Web); WebLINX's own action
models see the top 10 of that list (their candidates.k). Turns whose target isn't in the top k are counted (recall@k)
and kept out of the records, so element accuracy over all turns = record accuracy x recall.

Kev state: the website, the latest instructor utterance as the command, and the last 5 turns before it (chat and
actions, WebLINX's own num_prev_turns) as "Done so far".

Writes data/records/eval_weblinx.jsonl (records; _meta carries what webagent_baselines.py needs to rebuild the native
WebLINX prompt from the raw demo: demo, turn index, split, top-k uids, gold uid, intent, top-10 DMR candidates) and
data/converted/weblinx/{screens,examples}.jsonl (convert_public.py format: screenshot, the top-k candidates as
elements with pixel boxes, the target's gt_box in pixels; turns without a screenshot or box are left out of that file
only). Downloads only the files the sample needs (replay.json, metadata.json, the turn's page HTML and screenshot)
into data/public/weblinx/demonstrations/<demo>/. Nothing here loads a pickle.

  python scripts/data/convert_weblinx.py                  (GPU box; CPU only, ~10 GB of candidate files)
"""
import argparse
import csv
import json
import random
import re
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from rlca.records import record  # noqa: E402

REPO = "McGill-NLP/WebLINX-full"
SPLITS = ["test_iid", "test_web", "test_vis", "test_geo", "test_cat"]
ELEM_ACTION = re.compile(r"^(click|text_input|change|submit)\(")
ACTION = {"click": "click", "submit": "click", "textInput": "type", "change": "select"}
PREV_INTENTS = {"click", "change", "textInput", "scroll", "load", "say", "submit"}   # their select_turns filter
KEY = re.compile(rb'"demo_name":\s*"([^"]+)",\s*"turn_index":\s*(\d+)')
ATTR = re.compile(r"([\w:-]+)='(.*?)'(?=\s+[\w:-]+='|\s*$)", re.S)
LABEL_ATTRS = ("aria-label", "title", "placeholder", "alt", "value", "name", "aria-labelledby", "type")
csv.field_size_limit(1 << 30)


def fetch(root, files, workers=8):
    """hf_hub_download each missing file into root (same layout as the HF repo), retrying on rate limits."""
    from huggingface_hub import hf_hub_download
    todo = [f for f in dict.fromkeys(files) if not (root / f).exists()]

    def one(f):
        for k in range(10):
            try:
                return hf_hub_download(REPO, f, repo_type="dataset", local_dir=root)
            except Exception as e:   # noqa: BLE001  (429s and transient network errors)
                if "404" in str(e) or "EntryNotFound" in type(e).__name__:
                    return None
                time.sleep(min(30 * (k + 1), 120))
        print(f"gave up on {f}", flush=True)
    with ThreadPoolExecutor(workers) as ex:
        for i, _ in enumerate(ex.map(one, todo)):
            if (i + 1) % 200 == 0:
                print(f"  fetched {i + 1}/{len(todo)}", flush=True)


def sample_turns(root, n):
    """-> {split: [(demo, turn_index)]}, gold uids: seed-0 sample of the element-action rows of WebLINX's eval CSV.
    The gold uid is the CSV action's (their eval target); some replay elements lack data-webtasks-id."""
    out, gold = {}, {}
    for split in SPLITS:
        # the OOD CSVs name demos "<annotator>_<task words>_<demo id>"; the demo folders and candidates use the id
        rows = []
        for r in csv.DictReader(open(root / f"data/{split}.csv")):
            if ELEM_ACTION.match(r["action"]):
                rows.append((r["demo"].split("_")[-1], int(r["turn"])))
                gold[rows[-1]] = (re.findall(r'uid="([^"]+)"', r["action"]) or [None])[-1]
        out[split] = [rows[j] for j in sorted(random.Random(0).sample(range(len(rows)), min(n, len(rows))))]
    return out, gold


def ranked_candidates(path, want, top_k):
    """Stream one candidates jsonl -> {(demo, turn): {"top": first top_k by rank, uid-deduped, "all_uids": n,
    "by_uid": {uid: cand} for every candidate}}; only the wanted turns are parsed."""
    rows = defaultdict(list)
    with open(path, "rb") as f:
        for line in f:
            m = KEY.search(line, 0, 4000) or KEY.search(line)
            if m and (m[1].decode(), int(m[2])) in want:
                c = json.loads(line)
                c.pop("query", None)
                rows[(c["demo_name"], c["turn_index"])].append(c)
    out = {}
    for key, cands in rows.items():
        seen, ded = set(), []
        for c in sorted(cands, key=lambda c: c["rank"]):
            if c["uid"] not in seen:
                seen.add(c["uid"])
                ded.append(c)
        out[key] = {"top": ded[:top_k], "n": len(ded), "rank": {c["uid"]: i for i, c in enumerate(ded)},
                    "by_uid": {c["uid"]: c for c in ded}}
    return out


def png_size(path):
    """[w, h] from the PNG header (no image library needed)."""
    with open(path, "rb") as f:
        head = f.read(24)
    return [int.from_bytes(head[16:20], "big"), int.from_bytes(head[20:24], "big")]


def clean(s, n=300):
    return " ".join(str(s or "").split())[:n]


def elem_label(ed):
    """Element text for Kev: its text, else its most telling attribute (inputs and icons have no text)."""
    if clean(ed.get("text")):
        return clean(ed["text"])
    attrs = dict(ATTR.findall(ed.get("attributes") or ""))
    for a in LABEL_ATTRS:
        if clean(attrs.get(a)):
            return clean(attrs[a])
    return ""


def bbox(ed):
    """'x=195.2 y=220.3 width=96.7 height=31' -> [x, y, w, h] (CSS px in the viewport) or None."""
    v = dict(re.findall(r"(x|y|width|height)=(-?[\d.]+)", ed.get("bbox") or ""))
    return [float(v[k]) for k in ("x", "y", "width", "height")] if len(v) == 4 else None


def describe_turn(t):
    """One previous turn as a short line for Kev's "Done so far"."""
    a = (t.get("action") or {}).get("arguments") or {}
    intent = (t.get("action") or {}).get("intent")
    if t.get("type") == "chat":
        return f'{t.get("speaker")} said "{clean(t.get("utterance"), 200)}"'
    el = a.get("element") or {}
    tag, txt = (el.get("tagName") or "element").lower(), clean(el.get("textContent"), 60)
    attrs = el.get("attributes") or {}
    txt = txt or clean(attrs.get("aria-label") or attrs.get("placeholder") or attrs.get("title") or attrs.get("name"), 60)
    if intent == "textInput":
        return f'type "{clean(a.get("text"), 100)}" into [{tag}] {txt}'
    if intent == "change":
        return f'set [{tag}] {txt} to "{clean(a.get("value"), 100)}"'
    if intent in ("click", "submit"):
        return f"{intent} [{tag}] {txt}"
    if intent == "load":
        return f'load {clean(a.get("url") or (a.get("properties") or {}).get("url"), 120)}'
    return intent or "action"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(ROOT / "data/public/weblinx"))
    ap.add_argument("--out", default=str(ROOT / "data/records"))
    ap.add_argument("--converted", default=str(ROOT / "data/converted/weblinx"))
    ap.add_argument("--sample", type=int, default=500, help="turns per split (seed 0)")
    ap.add_argument("--top_k", type=int, default=50)
    ap.add_argument("--wl_k", type=int, default=10, help="candidates WebLINX's action models see (their candidates.k)")
    ap.add_argument("--splits", nargs="+", default=SPLITS, choices=SPLITS)
    a = ap.parse_args()
    root = Path(a.root)
    fetch(root, ["splits.json"] + [f"data/{s}.csv" for s in a.splits] + [f"candidates/{s}.jsonl" for s in a.splits])
    picks, csv_gold = sample_turns(root, a.sample)
    picks = {s: v for s, v in picks.items() if s in a.splits}
    demos = sorted({d for v in picks.values() for d, _ in v})
    print(f"{sum(map(len, picks.values()))} turns from {len(demos)} demos", flush=True)
    fetch(root, [f"demonstrations/{d}/{f}" for d in demos for f in ("replay.json", "metadata.json")])
    replays = {d: json.load(open(root / f"demonstrations/{d}/replay.json"))["data"] for d in demos}
    fetch(root, [f"demonstrations/{d}/{sub}/{replays[d][i]['state'][key]}" for v in picks.values() for d, i in v
                 for sub, key in (("pages", "page"), ("screenshots", "screenshot"))
                 if (replays[d][i].get("state") or {}).get(key)])

    out, conv = Path(a.out), Path(a.converted)
    out.mkdir(parents=True, exist_ok=True)
    conv.mkdir(parents=True, exist_ok=True)
    stats, written = {}, set()
    with open(out / "eval_weblinx.jsonl", "w") as fr, open(conv / "screens.jsonl", "w") as fs, \
            open(conv / "examples.jsonl", "w") as fe:
        for split, turns in picks.items():
            cands = ranked_candidates(root / f"candidates/{split}.jsonl", set(turns), a.top_k)
            st = stats[split] = defaultdict(int)
            for demo, ti in turns:
                st["turns"] += 1
                rep, turn = replays[demo], replays[demo][ti]
                act = turn["action"]
                intent, args = act["intent"], act.get("arguments") or {}
                st[f"intent_{intent}"] += 1
                el = args.get("element") or {}
                gold = csv_gold.get((demo, ti)) or (el.get("attributes") or {}).get("data-webtasks-id")
                c = cands.get((demo, ti))
                if c is None or gold is None:
                    st["no_candidates" if c is None else "no_gold_uid"] += 1
                    continue
                top = c["top"]
                k_gold = next((k for k, x in enumerate(top) if x["uid"] == gold), None)
                gold_rank = c["rank"].get(gold)
                st["in_top10"] += gold_rank is not None and gold_rank < a.wl_k
                # Kev's state: latest instructor utterance + last 5 turns before this one (minus that utterance).
                prev_turns = [(j, t) for j, t in enumerate(rep[:ti]) if t.get("type") == "chat"
                              or (t.get("action") or {}).get("intent") in PREV_INTENTS]
                inst = [(j, t) for j, t in prev_turns if t.get("type") == "chat" and t.get("speaker") == "instructor"]
                utt_j, utt = (inst[-1][0], clean(inst[-1][1].get("utterance"), 1000)) if inst else (None, "")
                prev = [describe_turn(t).replace("; ", ", ") for j, t in prev_turns if j != utt_j][-5:]
                state = turn.get("state") or {}
                md = args.get("metadata") or {}
                url = md.get("url") or ""
                site = urlparse(url).netloc.removeprefix("www.") or "web"
                shot = f"demonstrations/{demo}/screenshots/{state['screenshot']}" if state.get("screenshot") else None
                vw, vh = md.get("viewportWidth"), md.get("viewportHeight")
                opts = [{"id": f"e{k}", "role": x["elem_dict"].get("tag") or "element", "label": elem_label(x["elem_dict"])}
                        for k, x in enumerate(top)]
                value = args.get("text") if intent == "textInput" else args.get("value") if intent == "change" else None
                # gt box: the target's own bbox at that turn (DMR doc; replay element as fallback), scaled to pixels
                g = c["by_uid"].get(gold)
                box = bbox(g["elem_dict"]) if g else None
                if box is None and el.get("bbox"):
                    b = el["bbox"]
                    box = [b["x"], b["y"], b["width"], b["height"]]
                px_box, size, elements = None, None, None
                if shot and (root / shot).exists() and box and box[2] > 0 and box[3] > 0:
                    size = png_size(root / shot)
                    s = size[0] / vw if vw else 1.0
                    px_box = [round(v * s) for v in box]
                    elements = [{"id": f"e{k}", "role": o["role"], "label": o["label"],
                                 "box": [round(v * s) for v in (bbox(x["elem_dict"]) or [0, 0, 0, 0])],
                                 "sources": ["weblinx_dmr"]} for k, (o, x) in enumerate(zip(opts, top))]
                meta = {"source": "weblinx", "id": f"{demo}-{ti}", "split": split, "demo": demo, "turn_index": ti,
                        "website": site, "url": url, "intent": intent, "value": value, "utterance": utt, "prev": prev,
                        "viewport": [vh, vw], "cand_uids": [x["uid"] for x in top], "gold_uid": gold,
                        "gold_rank": gold_rank, "n_cands": c["n"], "screenshot": shot, "gt_box": px_box,
                        "wl_cands": [{k: v for k, v in x.items()} for x in top[: a.wl_k]]}
                if px_box:
                    st["with_box"] += 1
                    rel = f"weblinx/{shot}"
                    if rel not in written:
                        written.add(rel)
                        fs.write(json.dumps({"image": rel, "size": size, "elements": elements}) + "\n")
                    fe.write(json.dumps({"id": f"weblinx-{demo}-{ti}", "image": rel, "split": "eval", "wl_split": split,
                                         "platform": "web", "app": site, "instruction": utt, "history": prev,
                                         "action": ACTION[intent], "text": value, "gt_box": px_box,
                                         "target_ids": [f"e{k_gold}"] if k_gold is not None else []},
                                        ensure_ascii=False) + "\n")
                if k_gold is None:
                    continue
                st["in_top_k"] += 1
                command = f"{utt or 'None'}\nDone so far: {'; '.join(prev) or 'None'}"
                rec = record(f"{site} (web)", command, opts, [f"e{k_gold}"], action=ACTION[intent], meta=meta)
                fr.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n = st["turns"]
            print(f"{split}: {n} turns, target in top {a.top_k} for {st['in_top_k']} ({st['in_top_k'] / max(n, 1):.1%}), "
                  f"in top {a.wl_k} {st['in_top10'] / max(n, 1):.1%}; with screenshot box {st['with_box']}; {dict(st)}",
                  flush=True)
    (conv / "stats.json").write_text(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
