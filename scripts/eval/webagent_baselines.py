"""Run the published open element-list web agents (MindAct, WebLINX) on our Kev eval records, to compare with Kev on
element accuracy, op accuracy and time per decision.

Each model gets its own prompt, candidate presentation and output parsing (sources next to each family below).
Our records carry no HTML (except Mind2Web, see --m2w_root), so candidates become a synthetic page: <html><body> with
one element per candidate, tag = our role, text = our label, consecutive candidates sharing a context ("· in: A › B")
wrapped in a <div> labelled with it. That tree then goes through the model's own pruning/serialisation code.
MindAct's tournament runs over the top --top_k of Mind2Web records (ranker-ordered, like theirs) but over every
candidate of the other sets: those are in screen order, not ranked, and a top-50 cut would drop the target in
11-68% of decisions (Kev sees all of them too). Dropped everywhere: the app name (neither format has a slot) and "disabled". Gold = record target ids (or label);
a "None of the above" / say / scroll / unparsed answer counts as picking "none". Batch size 1 per decision, greedy.
Latency = GPU-synchronised wall time per decision from prompt building to parsed pick (all tournament calls, and the
DMR ranking for WebLINX); the first 3 decisions of a run are warm-up and left out of the speed summary. Rows append,
so a rerun resumes.

Writes runs/webagents/<model>/<set>/rows.jsonl and summary.json. GPU box only (not the Mac):
  git clone https://github.com/OSU-NLP-Group/Mind2Web ~/Mind2Web     (all models import its dom_utils; pure python)
  pip install lxml sentence-transformers                            (sentence-transformers: WebLINX DMR ranker only)
  python scripts/eval/webagent_baselines.py --model mindact-large --sets m2w_task ssv2 val [--limit 50]
"""
import argparse
import hashlib
import json
import random
import re
import statistics
import sys
import time
import types
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SETS = {"m2w_task": "eval_m2w_task.jsonl", "m2w_website": "eval_m2w_website.jsonl", "m2w_domain": "eval_m2w_domain.jsonl",
        "ssv2": "eval_ssv2.jsonl", "sspro": "eval_sspro.jsonl", "osworldg": "eval_osworldg.jsonl", "mmbench": "eval_mmbench.jsonl", "omniact": "eval_omniact.jsonl", "val": "val.jsonl",
        "heldout_site": "heldout_site.jsonl"}
MODELS = {f"mindact-{s}": (f"osunlp/MindAct_ActionPrediction_flan-t5-{s}", "mindact") for s in ("base", "large", "xl")}
MODELS.update({"wl-flan-base": ("McGill-NLP/flan-t5-base-weblinx", "wl_flan"),
               "wl-flan-large": ("McGill-NLP/flan-t5-large-weblinx", "wl_flan"),
               "wl-llama-1.3b": ("McGill-NLP/Sheared-LLaMA-1.3B-weblinx", "wl_llama"),
               "wl-mindact-base": ("McGill-NLP/MindAct-base-weblinx", "wl_m2w"),
               "wl-mindact-large": ("McGill-NLP/MindAct-large-weblinx", "wl_m2w")})
WARMUP = 3
DESC = re.compile(r'(?P<role>\S+) "(?P<label>.*)"(?: · in: (?P<ctx>.*?))?(?P<off> · disabled)?$', re.S)
BAD_XML = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


def parse_record(rec):
    """-> task, previous actions, candidates [(id, role, label, context)], gold ids, gold op, from our record format."""
    head, _, cmd = rec["state"].partition("\nUser command: ")
    task, _, done = cmd.partition("\nDone so far: ")       # Mind2Web records: last 3 action_reprs joined by "; "
    q = rec["questions"]
    cands = []
    for cid, d in q["element"]["criteria"].items():
        if cid == "none":
            continue
        m = DESC.match(d)
        cands.append((cid, m["role"], m["label"], tuple(m["ctx"].split(" › ")) if m["ctx"] else ()) if m else (cid, "item", d, ()))
    return {"id": rec["_meta"]["id"], "app": head.removeprefix("App: "), "task": task, "cands": cands,
            "prev": [] if done in ("", "None") else done.split("; "), "meta": rec["_meta"],
            "gold": set(q["element"].get("target") or [q["element"]["label"]]), "op_gold": q.get("action", {}).get("label")}


def tag_of(role):
    t = re.sub(r"[^a-z0-9]", "", role.lower()) or "item"
    if t == "text":     # Mind2Web/WebLINX DOM code treats <text> as a text node, not an element
        return "span"
    return "x" + t if t[0].isdigit() else t


def synth_tree(cands, uids, label_attr, pad):
    """Synthetic DOM; every node gets a backend_node_id so the Mind2Web/WebLINX pruning works as on real pages."""
    from lxml import etree
    root = etree.Element("html", backend_node_id="_r0")
    body = etree.SubElement(root, "body", backend_node_id="_r1")
    parent, cur, nodes = body, (), {}
    for cid, role, label, ctx in cands:
        if ctx != cur:
            parent = etree.SubElement(body, "div", {"backend_node_id": f"_g{len(body)}",
                                                    label_attr: BAD_XML.sub("", " › ".join(ctx))}) if ctx else body
            cur = ctx
        nodes[cid] = el = etree.SubElement(parent, tag_of(role), backend_node_id=uids[cid])
        el.text = f" {BAD_XML.sub('', label)} " if pad else BAD_XML.sub("", label)
    return root, nodes


# ---- MindAct (Mind2Web) ----
# Source: https://github.com/OSU-NLP-Group/Mind2Web src/action_prediction/. The prompt is their own
# format_input_multichoice (dataloader.py: pruned-HTML context, "Based on the HTML webpage above, try to complete the
# following task:", Task, Previous actions (last 5, one per line, "None"), "A. None of the above", choices = first 10
# words of each candidate's tree repr), imported, not copied. Tokenisation = MultiChoiceDataset (context 512 tokens,
# no special tokens, + question 512 tokens); generation and parsing = ActionEvaluatorMultiChoice.evaluate_dataset
# (metric.py): shuffle the top-k, groups of 5, one generate per group (not batched), max_new_tokens=50, winners
# re-queued until <=1 candidate remains; postprocess_action parses "B.\nAction: CLICK\nValue: ...". fp32: the
# checkpoints are fp32 and evaluate.py loads them without a dtype (it also calls to_bettertransformer(), an
# optimum speed-up we skip). Tokenizer google/flan-t5-<size> (conf/model/*.yaml).
def mindact(repo, args):
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    try:
        import datasets  # noqa: F401  (dataloader.py imports load_dataset at module level; unused here)
    except ImportError:
        sys.modules["datasets"] = types.SimpleNamespace(load_dataset=None)
    from dataloader import format_input_multichoice
    from metric import ActionEvaluatorMultiChoice
    tok = AutoTokenizer.from_pretrained("google/flan-t5-" + repo.rsplit("-", 1)[1])
    model = AutoModelForSeq2SeqLM.from_pretrained(repo).cuda().eval()
    ev, raw = ActionEvaluatorMultiChoice(tok), M2WRaw(args.m2w_root)
    from lxml import etree

    def predict(r):
        real = raw.lookup(r)                      # real cleaned_html + full action history when available
        if real:
            sample, back = real
        else:
            back = {c[0]: c[0] for c in r["cands"]}
            root, _ = synth_tree(r["cands"], back, "aria_label", pad=True)
            sample = {"cleaned_html": etree.tostring(root, encoding="unicode"),
                      "confirmed_task": r["meta"].get("task", r["task"]), "previous_actions": r["meta"].get("prev", r["prev"])}
        pool = list(back)
        random.Random(r["id"]).shuffle(pool)      # theirs: unseeded random.shuffle; seeded per decision for reruns
        final, calls = None, 0
        while len(pool) > 1:
            group, pool = pool[:5], pool[5:]
            ctx, inp, _, choices = format_input_multichoice(sample, group, -1)
            ids = (tok(ctx, truncation=True, max_length=512, add_special_tokens=False)["input_ids"]
                   + tok(inp, add_special_tokens=True, truncation=True, max_length=512)["input_ids"])
            x = torch.tensor([ids], device="cuda")
            out = model.generate(input_ids=x, attention_mask=torch.ones_like(x), eos_token_id=model.config.eos_token_id,
                                 max_new_tokens=50, do_sample=False)
            calls += 1
            text = tok.batch_decode(out, skip_special_tokens=True)[0]
            el, act = ev.postprocess_action(text) if text.strip() else ("A", "")   # theirs crashes on ""
            if el[0] != "A":
                try:
                    pid = choices[ord(el[0]) - ord("B")][0]
                    pool.append(pid)
                    final = (pid, act)
                except IndexError:
                    pass
        if not pool or final is None:
            return {"pick": "none", "op_raw": None, "calls": calls, "real_html": bool(real)}
        op = final[1].split()[0] if final[1].strip() else None
        return {"pick": back[final[0]], "op_raw": op, "op": {"CLICK": "click", "TYPE": "type", "SELECT": "select"}.get(op),
                "calls": calls, "real_html": bool(real), "value": final[1].partition(" ")[2] or None}
    return predict, {"dtype": "float32"}


class M2WRaw:
    """Raw Mind2Web step for a record: its cleaned_html, with our e<k> ids mapped through _meta.backend_ids (written by
    convert_mind2web.py in candidate order), and the full action history (MindAct shows the last 5; our state has 3)."""
    def __init__(self, root):
        self.root, self.splits = Path(root), {}

    def lookup(self, r):
        m = r["meta"]
        if m.get("source") != "mind2web" or not m.get("backend_ids") or not (self.root / m["split"]).exists():
            return None
        if m["split"] not in self.splits:
            self.splits[m["split"]] = {t["annotation_id"]: t for f in sorted((self.root / m["split"]).glob("*.json"))
                                       for t in json.load(open(f))}
        ann, i = r["id"].rsplit("-", 1)
        t, i = self.splits[m["split"]].get(ann), int(i)
        if t is None:
            return None
        ids = {c[0] for c in r["cands"]}
        back = {b: f"e{k}" for k, b in enumerate(m["backend_ids"]) if f"e{k}" in ids}
        return {"cleaned_html": t["actions"][i]["cleaned_html"], "confirmed_task": t["confirmed_task"],
                "previous_actions": t["action_reprs"][:i]}, back


# ---- WebLINX ----
# Source: https://github.com/McGill-NLP/weblinx modeling/ (flan/eval.py, flan/processing.py, flan/processing_m2w.py,
# llama/processing.py, llama/eval.py, dmr/processing.py) and weblinx/processing/{prompt,dom,truncation,outputs}.py,
# checked against the training prompts shipped as inputs.json in the HF repos. Pipeline: DMR ranker
# (McGill-NLP/MiniLM-L6-dmr, cosine) picks the top --wl_k=10 (candidates.k) of our top-k; the action model writes
# click(uid="...") etc. Our conversation is one instructor turn: utterance "[00:00] <command>", previous turns =
# that say(...) turn. Not reproducible from our data, so changed: no screen bbox ([[bbox]] dropped from candidate docs,
# empty in DMR docs), fixed viewport (--viewport), synthetic page HTML, uids are hashes in WebLINX's 8-4-4 hex shape,
# Mind2Web "Done so far" actions dropped (no uids to write them as turns), HTML cut at 700 tokens by plain truncation
# (theirs shortens attributes first), dynamic-RoPE x2 and flash-attn of llama/eval.py not set (RoPE scaling is a
# no-op under 4096 tokens; sdpa instead). bf16 like their eval; max_new_tokens 256; T5 input truncated at 2048.
SYS_WL = ("You are an AI assistant with a deep understanding of HTML{sp}and you must predict actions based on a user "
          "request, which will be executed. Use one of the following, replacing [] with an appropriate value: "
          "change(value=[str], uid=[str]) ; click(uid=[str]) ; load(url=[str]) ; say(speaker=\"navigator\", "
          "utterance=[str]) ; scroll(x=[int], y=[int]) ; submit(uid=[str]) ;text_input(text=[str], uid=[str]) ;\n"
          "The user's first and last 4 utterances are: {utt} ;\nViewport size: {h}h x {w}w ;\n"
          "Only the last 5 turns are provided.")
SYS_M2W = ("You will find above the HTML elements available for the current webpage.\nYou are an AI assistant tasked "
           "with helping a user (aka Instructor) by answering with the action needed to perform a task on a webpage.\n"
           "Here are the instructor's utterances, truncated to first and last 4 instances preceded by the relative "
           "timestamp: {utt} ; \nOnly the last 5 actions are available.")
CANDS_WL = "Here are the top candidates for this turn: {}\n"
FINAL_WL = "Please select the best action using the correct format, do not provide any other information or explanation."
WL_OPS = {"click": "click", "submit": "click", "text_input": "type", "textInput": "type", "change": "select"}
ESC = [("&quot;", '"'), ("&amp;", "&"), ("&lt;", "<"), ("&gt;", ">"), ("&nbsp;", " "), ("&ndash;", "-"), ("&rsquo;", "'"),
       ("&lsquo;", "'"), ("&ldquo;", '"'), ("&rdquo;", '"'), ("&#39;", "'"), ("&#40;", "("), ("&#41;", ")")]


def tree_repr_simple(tree):
    """weblinx.processing.dom.get_tree_repr_simple (without the postfix)."""
    from lxml import etree
    s = re.sub(r"<text>(.*?)</text>", r"\1", etree.tostring(tree, method="html", encoding="unicode"))
    s = re.sub(r"<(.+?)>", r"(\1", re.sub(r"</(.+?)>", ")", s.replace("/>", "$/$>"))).replace("$/$", ")")
    for k, v in ESC:
        s = s.replace(k, v)
    s = re.sub(r"\s+", " ", s).strip()
    return re.sub(r"\( +\(", "((", re.sub(r"\) +\)", "))", s))


def parse_wl(text):
    """-> (intent, args) like weblinx.processing.outputs.parse_predicted_output_string; m2w's action(intent=...) too."""
    i = text.find("(")
    if i < 0:
        return "", {}
    intent, quote, j = re.search(r"\w*$", text[:i]).group(0), False, len(text)
    for k in range(i, len(text)):
        quote = quote != (text[k] == '"')
        if text[k] == ")" and not quote:
            j = k
            break
    parts, cur, quote = [], "", False
    for ch in text[i + 1:j]:
        quote = quote != (ch == '"')
        if ch == "," and not quote:
            parts, cur = parts + [cur], ""
        else:
            cur += ch
    kv = {k.strip(): v.strip().strip('"') for k, _, v in (p.partition("=") for p in parts + [cur] if "=" in p)}
    return (kv.get("intent", "") if intent == "action" else intent), kv


def weblinx(repo, kind, args):
    import torch
    from data_utils.dom_utils import get_tree_repr, prune_tree
    from sentence_transformers import SentenceTransformer
    from sentence_transformers.util import cos_sim
    from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(repo)
    cls = AutoModelForCausalLM if kind == "wl_llama" else AutoModelForSeq2SeqLM
    extra = {"attn_implementation": "sdpa"} if kind == "wl_llama" else {}
    model = cls.from_pretrained(repo, dtype=torch.bfloat16, **extra).cuda().eval()
    dmr = SentenceTransformer("McGill-NLP/MiniLM-L6-dmr", device="cuda")
    dmr.max_seq_length = 512                      # dmr/conf/config.yaml
    h, w = args.viewport.split("x")

    def predict(r):
        rid = r["id"]
        uids = {c[0]: (lambda x: f"{x[:8]}-{x[8:12]}-{x[12:16]}")(hashlib.md5(f"{rid}/{c[0]}".encode()).hexdigest())
                for c in r["cands"]}
        back = {u: c for c, u in uids.items()}
        cmd = r["task"].replace("\n", "\\n")[:200]
        utt = f"[00:00] {r['task']}"
        root, nodes = synth_tree(r["cands"], uids, "aria-label", pad=False)
        tree = root.getroottree()
        info = {c[0]: (nodes[c[0]].tag, tree.getpath(nodes[c[0]]), c[2]) for c in r["cands"]}
        # DMR stage (dmr/processing.py format_turn_for_input + convert_elem_dict_to_str_legacy).
        query = (f"Viewport(height={h}, width={w}) ---- Instructor Utterances: {utt} ---- "
                 f'Previous Turns:say(speaker="instructor", utterance="{cmd}", timestamp="00:00")')
        docs = [f"[[tag]] {t}\n[[xpath]] {xp}\n[[text]] {tx}\n[[bbox]] \n[[attributes]] data-webtasks-id={uids[c]!r}\n"
                f"[[children]] " for c, (t, xp, tx) in info.items()]
        enc = dmr.encode([query] + docs, convert_to_tensor=True, show_progress_bar=False)
        scores = cos_sim(enc[0], enc[1:])[0].tolist()
        top = [c for _, c in sorted(zip(scores, info), key=lambda p: -p[0])][: args.wl_k]
        top_u = [uids[c] for c in top]
        if kind == "wl_m2w":                      # flan/processing_m2w.py build_prompt_records_for_m2w
            pruned = prune_tree(root, set(top_u))
            html, idm = get_tree_repr(pruned, id_mapping={})
            ch = {n.attrib["backend_node_id"]: " ".join(get_tree_repr(n, id_mapping=idm)[0].split()[:10])
                  for n in pruned.xpath("//*[@backend_node_id]")}
            cand_str = "".join(f"(uid = {u}) {(d if len(d) <= 300 else d[:297] + '...')}\n"
                               for u in top_u for d in [ch[u].replace("\n", " ")])
            sys_p = html + "\n" + SYS_M2W.format(utt=utt) + "\n" + CANDS_WL.format(cand_str)
            prompt = (sys_p + "\n" + f'Assistant: action(intent="say", speaker="instructor", utterance="{cmd}")\n'
                      + f"User: {FINAL_WL}\n" + "\nAssistant:")
        else:                                     # llama/processing.py build_prompt_records_for_llama_truncated
            pruned = prune_tree(root, set(top_u), max_depth=1, max_children=5, max_sibling=2)
            for n in pruned.iter():
                if "backend_node_id" in n.attrib:
                    n.attrib["data-webtasks-id"] = n.attrib.pop("backend_node_id")
            html = tree_repr_simple(pruned)
            ids = tok(html, add_special_tokens=False)["input_ids"]
            html = (tok.decode(ids[:700]) if len(ids) > 700 else html) + "\nAbove are the pruned HTML contents of the page."
            cand_str = "".join(f"(uid = {uids[c]}) [[tag]] {info[c][0]} [[xpath]] {info[c][1]} "
                               + (f"[[text]] {info[c][2][:200]} " if info[c][2] else "")
                               + f"[[attributes]] data-webtasks-id='{uids[c]}' \n" for c in top)
            # inputs.json: the Sheared-LLaMA run's template had no space in "HTMLand"; flan's has "HTML and".
            sys_p = (html + SYS_WL.format(sp="" if kind == "wl_llama" else " ", utt=utt, h=h, w=w) + "\n"
                     + CANDS_WL.format(cand_str))
            user = f'say(speaker="instructor", utterance="{cmd}") {FINAL_WL}'
            prompt = (f"<s>[INST] <<SYS>>\n{sys_p}\n<</SYS>>\n\n{user}[/INST]" if kind == "wl_llama"  # format_prompt_llama
                      else sys_p + "\n" + f"User: {user}\n" + "\nAssistant:")              # format_prompt_for_flan
        with torch.autocast("cuda", dtype=torch.bfloat16):
            if kind == "wl_llama":
                x = tok(prompt, add_special_tokens=False, return_tensors="pt").to("cuda")
                out = model.generate(**x, max_new_tokens=256, do_sample=False, pad_token_id=tok.eos_token_id)
                text = tok.decode(out[0, x["input_ids"].shape[1]:], skip_special_tokens=True)
            else:
                x = tok(prompt, truncation=True, max_length=2048, return_tensors="pt").to("cuda")
                text = tok.decode(model.generate(**x, max_new_tokens=256, do_sample=False)[0], skip_special_tokens=True)
        intent, kv = parse_wl(text)
        pick = back.get(kv.get("uid")) if intent in WL_OPS else None
        return {"pick": pick or "none", "op_raw": intent or None, "op": WL_OPS.get(intent) if pick else None,
                "calls": 1, "dmr_hit": bool(set(top) & r["gold"]), "raw": text}
    return predict, {"dtype": "bfloat16", "wl_k": args.wl_k, "viewport": args.viewport}


def summarize(rows, meta):
    lat = [r["latency_ms"] for r in rows if not r["warmup"]]
    with_op = [r for r in rows if r["op_gold"]]
    hits_op = [r for r in with_op if r["hit"]]
    q = statistics.quantiles(lat, n=10, method="inclusive")[-1] if len(lat) >= 2 else (lat[0] if lat else None)
    mean = lambda xs: sum(xs) / len(xs) if xs else None  # noqa: E731
    s = {**meta, "n": len(rows), "element_accuracy": mean([r["hit"] for r in rows]),
         "op_accuracy": mean([r["op"] == r["op_gold"] for r in with_op]), "n_with_op": len(with_op),
         "op_accuracy_on_element_hits": mean([r["op"] == r["op_gold"] for r in hits_op]),
         "n_pick_none": sum(r["pick"] == "none" for r in rows), "mean_calls": mean([r["n_model_calls"] for r in rows]),
         "latency_ms": {"n": len(lat), "median": statistics.median(lat) if lat else None, "p90": q}}
    if any("op_gold_raw" in r for r in rows):
        raw = [r for r in rows if r.get("op_gold_raw")]
        s["op_accuracy_raw_m2w"] = mean([r["op_raw"] == r["op_gold_raw"] for r in raw])
    if any("dmr_hit" in r for r in rows):
        s["dmr_recall_at_k"] = mean([r["dmr_hit"] for r in rows])
    if any("real_html" in r for r in rows):
        s["n_real_html"] = sum(r["real_html"] for r in rows)
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=MODELS)
    ap.add_argument("--sets", nargs="+", default=list(SETS), choices=SETS)
    ap.add_argument("--records_dir", default="data/records")
    ap.add_argument("--limit", type=int, default=0, help="first N decisions per set")
    ap.add_argument("--out", default="runs/webagents")
    ap.add_argument("--top_k", type=int, default=50, help="Mind2Web: first K (ranker-ordered) candidates; other sets use all")
    ap.add_argument("--wl_k", type=int, default=10, help="WebLINX: candidates kept by the DMR ranker")
    ap.add_argument("--viewport", default="746x1536", help="WebLINX: HxW written in the prompt (we have none)")
    ap.add_argument("--mind2web_repo", default="~/Mind2Web", help="clone of OSU-NLP-Group/Mind2Web")
    ap.add_argument("--m2w_root", default="data/public/mind2web", help="raw Mind2Web test splits (real HTML)")
    args = ap.parse_args()
    import torch
    import transformers
    sys.path.insert(0, str(Path(args.mind2web_repo).expanduser() / "src/action_prediction"))
    sys.path.insert(0, str(Path(args.mind2web_repo).expanduser() / "src"))
    args.m2w_root = str(ROOT / args.m2w_root)
    repo, kind = MODELS[args.model]
    predict, info = mindact(repo, args) if kind == "mindact" else weblinx(repo, kind, args)
    meta = {"model": repo, "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
            "transformers": transformers.__version__, "top_k": args.top_k, **info}
    ran = 0
    for set_name in args.sets:
        src = ROOT / args.records_dir / SETS[set_name]
        if not src.exists():
            print(f"{set_name}: no {src}, skipped", flush=True)
            continue
        recs = [json.loads(line) for line in open(src)]
        if set_name == "val":
            recs = [x for x in recs if x["_meta"].get("source") == "mac_labels"]
        recs = recs[: args.limit or None]
        out = ROOT / args.out / args.model / set_name
        out.mkdir(parents=True, exist_ok=True)
        rows_path = out / "rows.jsonl"
        done = {json.loads(line)["id"] for line in open(rows_path)} if rows_path.exists() else set()
        with open(rows_path, "a") as f:
            for i, rec in enumerate(recs):
                r = parse_record(rec)
                if r["id"] in done:
                    continue
                if r["meta"].get("source") == "mind2web":   # ranker-ordered; other sets are in screen order, so
                    r["cands"] = r["cands"][: args.top_k]   # cutting them would drop the target (it does in ~40%)
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                with torch.inference_mode():
                    p = predict(r)
                torch.cuda.synchronize()
                ms = (time.perf_counter() - t0) * 1000
                row = {"id": r["id"], "pick": p.pop("pick"), "gold": sorted(r["gold"]), "op": p.pop("op", None),
                       "op_gold": r["op_gold"], "n_model_calls": p.pop("calls"), "latency_ms": round(ms, 1),
                       "warmup": ran < WARMUP, "n_cands": len(r["cands"]), **p}
                row["hit"] = row["pick"] in r["gold"]
                if r["meta"].get("op"):
                    row["op_gold_raw"] = r["meta"]["op"]
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                f.flush()
                ran += 1
                if ran % 50 == 0:
                    print(f"{set_name} {i + 1}/{len(recs)}", flush=True)
        rows = [json.loads(line) for line in open(rows_path)]
        keep = {x["_meta"]["id"] for x in recs}
        s = summarize([x for x in rows if x["id"] in keep], {**meta, "set": set_name})
        (out / "summary.json").write_text(json.dumps(s, indent=2))
        print(f"{set_name}: element acc {s['element_accuracy']} op acc {s['op_accuracy']} n {s['n']} "
              f"median {s['latency_ms']['median']} ms p90 {s['latency_ms']['p90']} ms calls {s['mean_calls']}", flush=True)


if __name__ == "__main__":
    main()
