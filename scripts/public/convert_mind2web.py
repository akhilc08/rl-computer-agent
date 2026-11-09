"""Mind2Web test splits -> Kev eval records, with element lists from the page's own HTML.

Mind2Web gives every page element of each step as a candidate (pos = the target, neg = the rest), so this is the
element-list benchmark: no pixel parser. Like MindAct, a cross-encoder first ranks the candidates and Kev picks
from the top --top_k (MindAct uses 50); the ranker is MindAct's released DeBERTa-v3 candidate model, fed
MindAct's own candidate text (ancestors + target from their DOM helpers, vendored in mind2web_dom.py) in their
order (candidate, query). We don't load the dataset's precomputed scores_all_data.pkl (a downloaded pickle), so
ranks are recomputed here and recall@k is reported next to accuracy. Steps whose target isn't in the top k are kept out of the
records and counted, so element accuracy over all steps = record accuracy x recall.

State: website, the task, and the last 3 actions done (Mind2Web is multi-step; each step is one decision).

  python scripts/public/convert_mind2web.py --root data/public/mind2web --out data/records   (GPU box)
"""
import argparse
import glob
import json
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from mind2web_dom import get_tree_repr, prune_tree  # noqa: E402
from rlca.records import record  # noqa: E402

OPS = {"CLICK": "click", "TYPE": "type", "SELECT": "select", "HOVER": "click", "ENTER": "click"}


def format_candidate(dom_tree, backend_id):
    """-> (ranker text, element text). Ranker text is Mind2Web's candidate_generation/dataloader.py format_candidate."""
    node_tree = prune_tree(dom_tree, [backend_id])
    c_node = node_tree.xpath("//*[@backend_node_id]")[0]
    ancestor_repr = ""
    if c_node.getparent() is not None:
        c_node.getparent().remove(c_node)
        ancestor_repr, _ = get_tree_repr(node_tree, id_mapping={})
    subtree_repr, _ = get_tree_repr(c_node, id_mapping={})
    subtree_repr = " ".join(subtree_repr.split()[:100]) if subtree_repr.strip() else ""
    if ancestor_repr.strip():
        ancestor_repr = re.sub(r"\s*\(\s*", "/", ancestor_repr)
        ancestor_repr = re.sub(r"\s*\)\s*", "", ancestor_repr)
        ancestor_repr = " ".join(ancestor_repr.split()[-50:])
    return f"ancestors: {ancestor_repr}\n" + f"target: {subtree_repr}", subtree_repr


def kev_option(subtree_repr):
    """(button id=3 meta=submit Search) -> ("button", "submit Search") for Kev's id: description line."""
    m = re.match(r"\((\S+)(.*)", subtree_repr)
    if not m:
        return "element", subtree_repr
    body = re.sub(r"\bid=\S+", "", m.group(2)).replace("meta=", "")
    return m.group(1), " ".join(body.replace("(", " ").replace(")", " ").split())[:300]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(ROOT / "data/public/mind2web"))
    ap.add_argument("--out", default=str(ROOT / "data/records"))
    ap.add_argument("--top_k", type=int, default=50)
    ap.add_argument("--ranker", default="osunlp/MindAct_CandidateGeneration_deberta-v3-base")
    ap.add_argument("--limit", type=int, default=0, help="tasks per split (quick check)")
    ap.add_argument("--splits", nargs="+", default=["task", "website", "domain"])
    ap.add_argument("--sample", type=int, default=0, help="convert a fixed random sample of N steps per split (seed 0)")
    a = ap.parse_args()
    import torch
    from lxml import etree
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.ranker)
    ranker = AutoModelForSequenceClassification.from_pretrained(a.ranker).cuda().eval().half()

    @torch.no_grad()
    def rank(query, docs, bs=32):
        scores = []
        for i in range(0, len(docs), bs):
            enc = tok(docs[i:i + bs], [query] * len(docs[i:i + bs]), truncation=True, max_length=256, padding=True,
                      return_tensors="pt").to("cuda")
            scores += ranker(**enc).logits[:, -1].float().tolist()
        return scores

    for split in (f"test_{x}" for x in a.splits):
        tasks = [t for f in sorted(glob.glob(f"{a.root}/{split}/*.json")) for t in json.load(open(f))]
        if a.limit:
            tasks = tasks[: a.limit]
        steps = [(t, i) for t in tasks for i in range(len(t["actions"]))]
        if a.sample:
            steps = [steps[j] for j in sorted(random.Random(0).sample(range(len(steps)), min(a.sample, len(steps))))]
        n = hit = 0
        with open(Path(a.out) / f"eval_m2w_{split.removeprefix('test_')}.jsonl", "w") as f:
            for t, i in steps:
                act = t["actions"][i]
                tree = etree.fromstring(act["cleaned_html"])
                cands = [(c, True) for c in act["pos_candidates"]] + [(c, False) for c in act["neg_candidates"]]
                texts = [format_candidate(tree, c["backend_node_id"]) for c, _ in cands]
                prev_list = t["action_reprs"][max(0, i - 3):i]
                prev = "; ".join(prev_list)
                query = f"task is: {t['confirmed_task']}\nPrevious actions: {prev}"
                scores = rank(query, [r for r, _ in texts])
                top = sorted(range(len(cands)), key=lambda j: -scores[j])[: a.top_k]
                n += 1
                if not any(cands[j][1] for j in top):
                    continue
                hit += 1
                opts = [dict(zip(("role", "label"), kev_option(texts[j][1])), id=f"e{k}") for k, j in enumerate(top)]
                targets = [f"e{k}" for k, j in enumerate(top) if cands[j][1]]
                op = act["operation"]
                state_app = f"{t['website']} (web)"
                command = f"{t['confirmed_task']}\nDone so far: {prev or 'None'}"
                rec = record(state_app, command, opts, targets, action=OPS.get(op["op"]),
                             meta={"source": "mind2web", "id": f"{t['annotation_id']}-{i}", "split": split,
                                   "domain": t["domain"], "website": t["website"], "op": op["op"], "value": op["value"],
                                   "task": t["confirmed_task"], "prev": prev_list, "action_index": i,
                                   "backend_ids": [cands[j][0]["backend_node_id"] for j in top]})
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"{split}: {n} steps, target in top {a.top_k} for {hit} ({hit / max(n, 1):.1%})", flush=True)


if __name__ == "__main__":
    main()
