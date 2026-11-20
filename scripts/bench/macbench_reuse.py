"""Carry predictions from one MacBench build to the next where the model's input did not change, and list what
still has to be run.

List models (Kev, Jev): a prediction is reused when the command's ready-made list record (command, options,
targets) is identical in both builds. Screenshot models: a row is reused when the command text and the screenshot
bytes are identical.

  python3 scripts/bench/macbench_reuse.py lists --old data/macbench/v1 --new data/macbench/v1.1 --track ax \
      --pred old/predictions.jsonl --out_pred reused.jsonl --out_todo todo_list.jsonl
  python3 scripts/bench/macbench_reuse.py points --old data/macbench/v1 --new data/macbench/v1.1 \
      --rows old/rows.jsonl --ids new_subset.txt --out_rows reused_rows.jsonl --out_todo todo_ids.txt
"""
import argparse
import hashlib
import json
from pathlib import Path


def lists(args):
    key = lambda r: json.dumps({k: r[k] for k in r if k != "_meta"}, sort_keys=True)
    old = {r["_meta"]["id"]: key(r) for r in map(json.loads, open(Path(args.old) / f"list_{args.track}.jsonl"))}
    preds = {p["id"]: p for p in map(json.loads, open(args.pred))}
    reused = todo = 0
    with open(args.out_pred, "w") as fp, open(args.out_todo, "w") as ft:
        for line in open(Path(args.new) / f"list_{args.track}.jsonl"):
            r = json.loads(line)
            i = r["_meta"]["id"]
            if old.get(i) == key(r) and i in preds:
                fp.write(json.dumps(preds[i]) + "\n")
                reused += 1
            else:
                ft.write(line)
                todo += 1
    print(f"{args.track}: reused {reused}, to run {todo}")


def points(args):
    def digest(bench, screen, cache={}):
        k = (bench, screen)
        if k not in cache:
            cache[k] = hashlib.sha1((Path(bench) / "images" / f"{screen}.jpg").read_bytes()).hexdigest()
        return cache[k]
    old = {c["id"]: c for c in map(json.loads, open(Path(args.old) / "commands.jsonl"))}
    new = {c["id"]: c for c in map(json.loads, open(Path(args.new) / "commands.jsonl"))}
    rows = {r["id"]: r for r in map(json.loads, open(args.rows))}
    want = open(args.ids).read().split()
    reused = todo = 0
    with open(args.out_rows, "w") as fr, open(args.out_todo, "w") as ft:
        for i in want:
            c, o = new[i], old.get(i)
            same = (o is not None and i in rows and o["command"] == c["command"] and o["target_boxes"] == c["target_boxes"]
                    and digest(args.old, o["screen"]) == digest(args.new, c["screen"]))
            if same:
                fr.write(json.dumps(rows[i]) + "\n")
                reused += 1
            else:
                ft.write(i + "\n")
                todo += 1
    print(f"points: reused {reused}, to run {todo}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="mode", required=True)
    a = sub.add_parser("lists")
    a.add_argument("--old", required=True)
    a.add_argument("--new", required=True)
    a.add_argument("--track", choices=["ax", "pixels"], required=True)
    a.add_argument("--pred", required=True)
    a.add_argument("--out_pred", required=True)
    a.add_argument("--out_todo", required=True)
    b = sub.add_parser("points")
    b.add_argument("--old", required=True)
    b.add_argument("--new", required=True)
    b.add_argument("--rows", required=True)
    b.add_argument("--ids", required=True)
    b.add_argument("--out_rows", required=True)
    b.add_argument("--out_todo", required=True)
    args = ap.parse_args()
    {"lists": lists, "points": points}[args.mode](args)


if __name__ == "__main__":
    main()
