"""Convert AndroidControl's low-level test steps into Kev's element-list format.

AndroidControl (Li et al., 2024, Google Research) records human demonstrations on real Android apps. Every
step has a direct low-level instruction ("tap the search bar"), a screenshot, the accessibility tree and the
gold action. Here the element list comes from the real accessibility tree, not a pixel parser, so this is the
"direct command + real accessibility list" benchmark.

The data is gzipped TFRecord shards on GCS (gs://gresearch/android_control, about 2.5 GB each). Test episodes
are spread over all 20 shards, so shards are streamed over HTTPS in order, only test episodes are kept, and
streaming stops once --pool_steps test steps are collected. A fixed random sample (--n, --seed) of those steps
is then converted. Nothing is written to disk apart from the sampled screenshots. No TensorFlow or android_env
needed: the TFRecord framing is read by hand and the protos are declared below (field numbers from
android_env/proto/a11y and tensorflow/core/example).

Elements: visible accessibility nodes with a label or an interaction (clickable, editable, checkable, ...).
An unlabelled control takes the text of the plain text nodes inside it, and those text nodes are dropped, so a
list row is one element as an agent would see it. Same box + same label duplicates are merged, and so are
plain (non-interactive) nodes stacked on one box (the status bar's hidden overflow icons). The on-screen keyboard
is left out (Kev types text instead of tapping keys); it still hides the app nodes beneath it.
Targets: a click / long press maps to the smallest element containing the point. A type step (input_text) maps
to the focused editable field, else the field whose text holds the typed text on the next screen, else the only
editable field. Other actions (scroll, open_app, back, home, wait) and unmatched steps are dropped and counted.

Writes data/converted/androidcontrol/{screens,examples,stats}.jsonl/json and screenshots to
data/public/androidcontrol/images/. Split "eval" (it's a public benchmark; keep it out of training).

  python scripts/data/convert_androidcontrol.py [--n 2000] [--seed 0] [--pool_steps 4000]
"""
import argparse
import collections
import gzip
import io
import json
import random
import shutil
import statistics
import struct
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PUB = ROOT / "data/public/androidcontrol"
OUT = ROOT / "data/converted/androidcontrol"
BASE = "https://storage.googleapis.com/gresearch/android_control"
N_SHARDS = 20


# ---- protos (only the fields we read; the rest are skipped as unknown fields) ----

def _messages():
    from google.protobuf import descriptor_pb2, descriptor_pool, message_factory
    F = descriptor_pb2.FieldDescriptorProto
    T = {"int32": F.TYPE_INT32, "int64": F.TYPE_INT64, "bool": F.TYPE_BOOL, "string": F.TYPE_STRING,
         "bytes": F.TYPE_BYTES, "float": F.TYPE_FLOAT}

    def msg(fd, name, fields, nested=()):
        m = fd.message_type.add(name=name)
        for n in nested:
            m.nested_type.add().CopyFrom(n)
        for num, (fname, ftype, rep) in fields.items():
            f = m.field.add(name=fname, number=num, label=F.LABEL_REPEATED if rep else F.LABEL_OPTIONAL)
            if ftype in T:
                f.type = T[ftype]
            else:
                f.type, f.type_name = F.TYPE_MESSAGE, ftype
        return m

    fd = descriptor_pb2.FileDescriptorProto(name="rlca_ac.proto", package="rlca_ac", syntax="proto3")
    msg(fd, "ProtoRect", {1: ("left", "int32", 0), 2: ("top", "int32", 0), 3: ("right", "int32", 0), 4: ("bottom", "int32", 0)})
    msg(fd, "Node", {1: ("unique_id", "int32", 0), 2: ("bounds_in_screen", ".rlca_ac.ProtoRect", 0),
                     3: ("class_name", "string", 0), 4: ("content_description", "string", 0), 5: ("hint_text", "string", 0),
                     6: ("package_name", "string", 0), 7: ("text", "string", 0), 10: ("view_id_resource_name", "string", 0),
                     12: ("is_checkable", "bool", 0), 13: ("is_checked", "bool", 0), 14: ("is_clickable", "bool", 0),
                     15: ("is_editable", "bool", 0), 16: ("is_enabled", "bool", 0), 18: ("is_focused", "bool", 0),
                     19: ("is_long_clickable", "bool", 0), 21: ("is_scrollable", "bool", 0), 22: ("is_selected", "bool", 0),
                     23: ("is_visible_to_user", "bool", 0), 25: ("child_ids", "int32", 1), 27: ("depth", "int32", 0),
                     31: ("tooltip_text", "string", 0)})
    msg(fd, "Tree", {1: ("nodes", ".rlca_ac.Node", 1)})
    msg(fd, "Window", {1: ("bounds_in_screen", ".rlca_ac.ProtoRect", 0), 3: ("id", "int32", 0), 4: ("layer", "int32", 0),
                       5: ("title", "string", 0), 6: ("window_type", "int32", 0), 8: ("is_active", "bool", 0),
                       11: ("tree", ".rlca_ac.Tree", 0)})
    msg(fd, "Forest", {1: ("windows", ".rlca_ac.Window", 1)})
    # tf.train.Example
    msg(fd, "BytesList", {1: ("value", "bytes", 1)})
    msg(fd, "Int64List", {1: ("value", "int64", 1)})
    msg(fd, "FloatList", {1: ("value", "float", 1)})
    msg(fd, "Feature", {1: ("bytes_list", ".rlca_ac.BytesList", 0), 2: ("float_list", ".rlca_ac.FloatList", 0),
                        3: ("int64_list", ".rlca_ac.Int64List", 0)})
    entry = descriptor_pb2.DescriptorProto(name="FeatureEntry", options=descriptor_pb2.MessageOptions(map_entry=True))
    entry.field.add(name="key", number=1, type=F.TYPE_STRING, label=F.LABEL_OPTIONAL)
    entry.field.add(name="value", number=2, type=F.TYPE_MESSAGE, type_name=".rlca_ac.Feature", label=F.LABEL_OPTIONAL)
    msg(fd, "Features", {1: ("feature", ".rlca_ac.Features.FeatureEntry", 1)}, nested=[entry])
    msg(fd, "Example", {1: ("features", ".rlca_ac.Features", 0)})
    pool = descriptor_pool.DescriptorPool()
    pool.Add(fd)
    get = lambda n: message_factory.GetMessageClass(pool.FindMessageTypeByName(f"rlca_ac.{n}"))
    return get("Example"), get("Forest")


def records(url):
    """Yield raw records from a gzipped TFRecord file streamed over HTTPS (CRCs not checked)."""
    with urllib.request.urlopen(url, timeout=120) as r, gzip.GzipFile(fileobj=io.BufferedReader(r, 1 << 20)) as f:
        while True:
            head = f.read(12)
            if len(head) < 12:
                return
            (n,) = struct.unpack("<Q", head[:8])
            data = f.read(n)
            f.read(4)
            if len(data) < n:
                return
            yield data


def episodes(Example, test_ids, pool_steps, log):
    """Stream shards in order; yield test episodes as dicts until pool_steps test steps have been seen."""
    seen = 0
    for k in range(N_SHARDS):
        url = f"{BASE}/android_control-{k:05d}-of-{N_SHARDS:05d}"
        log(f"shard {k}: streaming ({seen} test steps so far)")
        for raw in records(url):
            ex = Example.FromString(raw).features.feature
            eid = ex["episode_id"].int64_list.value[0]
            if eid not in test_ids:
                continue
            ep = {"episode_id": eid, "shard": k, "goal": ex["goal"].bytes_list.value[0].decode(),
                  "screenshots": list(ex["screenshots"].bytes_list.value),
                  "trees": list(ex["accessibility_trees"].bytes_list.value),
                  "widths": list(ex["screenshot_widths"].int64_list.value),
                  "heights": list(ex["screenshot_heights"].int64_list.value),
                  "actions": [json.loads(a) for a in ex["actions"].bytes_list.value],
                  "steps": [s.decode() for s in ex["step_instructions"].bytes_list.value]}
            seen += len(ep["actions"])
            yield ep
            if seen >= pool_steps:
                log(f"stopped in shard {k} with {seen} test steps")
                return


# ---- elements ----

ROLES = {"EditText": "textfield", "AutoCompleteTextView": "textfield", "SearchAutoComplete": "textfield",
         "MultiAutoCompleteTextView": "textfield", "Button": "button", "ImageButton": "button",
         "MaterialButton": "button", "FloatingActionButton": "button", "ExtendedFloatingActionButton": "button",
         "Chip": "button", "CheckBox": "checkbox", "CheckedTextView": "checkbox", "Switch": "switch",
         "SwitchCompat": "switch", "SwitchMaterial": "switch", "ToggleButton": "switch", "RadioButton": "radiobutton",
         "TextView": "text", "ImageView": "image", "SeekBar": "slider", "RatingBar": "slider", "Spinner": "popupbutton",
         "ProgressBar": "progress", "WebView": "webview", "TabView": "tab", "RecyclerView": "list", "ListView": "list",
         "GridView": "list", "ScrollView": "scrollarea", "HorizontalScrollView": "scrollarea", "ViewPager": "group"}


def role_of(n):
    if n.is_editable:
        return "textfield"
    short = n.class_name.rsplit(".", 1)[-1].rsplit("$", 1)[-1]
    role = ROLES.get(short) or ("group" if short.endswith("Layout") or short in ("View", "ViewGroup") else short.lower() or "view")
    if role in ("text", "image", "group", "view") and (n.is_clickable or n.is_long_clickable):
        role = "button"
    return role


def own_label(n):
    return " ".join((n.text or n.content_description or n.hint_text or n.tooltip_text).split())


def interactive(n):
    return n.is_clickable or n.is_long_clickable or n.is_editable or n.is_checkable


def clip(a, b):
    return (max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3]))


def area(r):
    return max(0, r[2] - r[0]) * max(0, r[3] - r[1])


def elements_of(forest, W, H):
    """-> elements (dicts, ids e0..) and the package of the top app window. Boxes are x, y, w, h in screen pixels
    (AndroidControl screenshots are the full device screen, so these are screenshot pixels too)."""
    screen = (0, 0, W, H)
    wins = sorted(forest.windows, key=lambda w: w.layer, reverse=True)      # top window first
    covers, out, app, ime = [], [], None, []
    for w in wins:
        wb = w.bounds_in_screen
        wrect = clip((wb.left, wb.top, wb.right, wb.bottom), screen)
        nodes = {n.unique_id: n for n in w.tree.nodes}
        children = {i: [c for c in n.child_ids if c in nodes] for i, n in nodes.items()}
        has_parent = {c for cs in children.values() for c in cs}
        if app is None and w.window_type == 1 and w.tree.nodes:
            app = w.tree.nodes[0].package_name
        # visible rect = own bounds clipped by every ancestor, the window and the screen; DFS order = reading order
        rects, order = {}, []
        stack = [(i, wrect) for i in sorted(nodes, reverse=True) if i not in has_parent]
        while stack:
            i, parent = stack.pop()
            if i in rects:
                continue
            b = nodes[i].bounds_in_screen
            rects[i] = clip((b.left, b.top, b.right, b.bottom), parent)
            order.append(i)
            stack.extend((c, rects[i]) for c in reversed(children[i]))
        visible = {i for i in order if nodes[i].is_visible_to_user and area(rects[i]) > 0 and rects[i][2] - rects[i][0] >= 2
                   and rects[i][3] - rects[i][1] >= 2 and not any(clip(rects[i], c) == rects[i] for c in covers)}

        def descendants(i):
            st = list(children[i])
            while st:
                j = st.pop(0)
                yield j
                st[:0] = children[j]

        absorbed, keep = set(), []
        for i in order:
            n = nodes[i]
            if i not in visible or i in absorbed:
                continue
            label, kids = own_label(n), []
            if interactive(n):
                for j in descendants(i):
                    m = nodes[j]
                    if j in visible and j not in absorbed and not interactive(m) and own_label(m):
                        if not label or own_label(m) in label:
                            kids.append(j)
                if not label:
                    label = " ".join(dict.fromkeys(own_label(nodes[j]) for j in kids))[:200]
                absorbed.update(kids)
            elif not label:
                continue
            rid = n.view_id_resource_name.rsplit("/", 1)[-1]
            if not label and rid:
                label = rid.replace("_", " ")
            keep.append((i, label, rid))
        if w.window_type == 2:      # input method: hides what's under it, but its keys aren't elements
            covers.append(wrect)
            ime.append([wrect[0], wrect[1], wrect[2] - wrect[0], wrect[3] - wrect[1]])
            continue
        for i, label, rid in keep:
            n, r = nodes[i], rects[i]
            flags = [f for f, v in (("clickable", n.is_clickable), ("long_clickable", n.is_long_clickable),
                                    ("editable", n.is_editable), ("checkable", n.is_checkable), ("checked", n.is_checked),
                                    ("selected", n.is_selected), ("focused", n.is_focused), ("disabled", not n.is_enabled)) if v]
            box = [r[0], r[1], r[2] - r[0], r[3] - r[1]]
            if any(e["box"] == box and (e["label"] == label or not (interactive(n) or e["flags"])) for e in out):
                continue
            out.append({"role": role_of(n), "label": label, "box": box, "class": n.class_name, "resource_id": rid,
                        "flags": flags, "window": w.window_type, "layer": w.layer, "sources": ["ax"]})
        covers.append(wrect)
    out = [{"id": f"e{k}", **e} for k, e in enumerate(out)]
    return out, app, ime


def inside(box, pt):
    return box[0] <= pt[0] <= box[0] + box[2] and box[1] <= pt[1] <= box[1] + box[3]


def iou(a, b):
    ix = max(0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    u = a[2] * a[3] + b[2] * b[3] - ix * iy
    return ix * iy / u if u else 0.0


def tap_target(els, pt):
    """-> target ids for a tap: the smallest element containing the point (topmost window first). If that one isn't
    clickable (a label inside a row), the smallest clickable element around it also counts."""
    hits = [e for e in els if inside(e["box"], pt)]
    if not hits:
        return []
    top = max(e["layer"] for e in hits)
    hits = sorted((e for e in hits if e["layer"] == top), key=lambda e: e["box"][2] * e["box"][3])
    ids = [hits[0]["id"]]
    if not {"clickable", "long_clickable"} & set(hits[0]["flags"]):
        ids += [e["id"] for e in hits[1:] if {"clickable", "long_clickable"} & set(e["flags"])][:1]
    return ids


def type_target(els, text, next_forest, W, H):
    """-> (target ids, how) for input_text, which carries no point."""
    fields = [e for e in els if "editable" in e["flags"]]
    focused = [e for e in fields if "focused" in e["flags"]]
    if focused:
        return [focused[0]["id"]], "focused"
    if next_forest is not None and text.strip():
        nxt, _, _ = elements_of(next_forest, W, H)
        for f in nxt:
            if "editable" in f["flags"] and text.strip().lower() in f["label"].lower():
                same = [e for e in fields if (f["resource_id"] and e["resource_id"] == f["resource_id"]) or iou(e["box"], f["box"]) > 0.5]
                if same:
                    return [same[0]["id"]], "next_screen"
    if len(fields) == 1:
        return [fields[0]["id"]], "only_field"
    return [], None


# ---- main ----

def convert_step(ep, i, Forest):
    """-> (row or None, drop reason or None, elements, size) for step i of an episode."""
    a = ep["actions"][i]
    kind = a["action_type"]
    W, H = ep["widths"][i], ep["heights"][i]
    if kind not in ("click", "long_press", "input_text"):
        return None, f"action:{kind}", None
    forest = Forest.FromString(ep["trees"][i])
    els, app, ime = elements_of(forest, W, H)
    row = {"id": f"ac-{ep['episode_id']}-{i}", "split": "eval", "platform": "android", "app": app,
           "instruction": ep["steps"][i], "goal": ep["goal"], "episode_id": ep["episode_id"], "step": i,
           "repeated_instruction": ep["steps"][i] in ep["steps"][:i][-1:] or ep["steps"][i] in ep["steps"][i + 1:i + 2]}
    if kind == "input_text":
        nxt = Forest.FromString(ep["trees"][i + 1]) if i + 1 < len(ep["trees"]) else None
        ids, how = type_target(els, a["text"], nxt, W, H)
        if not ids:
            return None, "type: no focused or edited field", None
        row.update(action="type", text=a["text"], type_target=how)
    else:
        pt = (a["x"], a["y"])
        if not (0 <= pt[0] < W and 0 <= pt[1] < H):
            return None, "tap off screen", None
        ids = tap_target(els, pt)
        if not ids and any(inside(r, pt) for r in ime):
            return None, "tap on keyboard", None
        if not ids:
            return None, "tap inside no element", None
        row.update(action="click" if kind == "click" else "long_press", gt_point=list(pt))
    box = next(e["box"] for e in els if e["id"] == ids[0])
    row.update(target_ids=ids, gt_box=box)
    return row, None, {"size": [W, H], "elements": els}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2000, help="test steps to sample")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pool_steps", type=int, default=4000, help="stop streaming once this many test steps are seen")
    ap.add_argument("--quality", type=int, default=90, help="JPEG quality of saved screenshots")
    args = ap.parse_args()
    from PIL import Image
    log = lambda s: print(s, flush=True)
    Example, Forest = _messages()
    with urllib.request.urlopen(f"{BASE}/splits.json", timeout=60) as r:
        test_ids = set(json.load(r)["test"])
    with urllib.request.urlopen(f"{BASE}/test_subsplits.json", timeout=60) as r:
        subsplits = {k: set(v) for k, v in json.load(r).items()}

    # Convert every pooled step now (trees are small); keep the screenshot only for steps that convert.
    tmp = OUT / "_tmp_png"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    pool, shards = {}, set()
    for ep in episodes(Example, test_ids, args.pool_steps, log):
        shards.add(ep["shard"])
        for i in range(len(ep["actions"])):
            row, why, screen = convert_step(ep, i, Forest)
            key = (ep["episode_id"], i)
            pool[key] = (row, why, screen)
            if row:
                (tmp / f"{key[0]}_{i}.png").write_bytes(ep["screenshots"][i])
    keys = sorted(pool)
    sample = sorted(random.Random(args.seed).sample(keys, min(args.n, len(keys))))
    log(f"pool: {len(keys)} test steps from {len({k[0] for k in keys})} episodes in shards {sorted(shards)}; sampled {len(sample)}")

    img_dir = PUB / "images"
    shutil.rmtree(img_dir, ignore_errors=True)
    img_dir.mkdir(parents=True)
    dropped, kept = collections.Counter(), []
    with open(OUT / "screens.jsonl", "w") as fs, open(OUT / "examples.jsonl", "w") as fe:
        for key in sample:
            row, why, screen = pool[key]
            if not row:
                dropped[why] += 1
                continue
            rel = f"androidcontrol/images/{key[0]}_{key[1]}.jpg"
            Image.open(tmp / f"{key[0]}_{key[1]}.png").convert("RGB").save(PUB.parent / rel, quality=args.quality)
            row["subsplit"] = sorted(k for k, v in subsplits.items() if key[0] in v)
            fs.write(json.dumps({"image": rel, **screen}) + "\n")
            fe.write(json.dumps({**row, "image": rel}) + "\n")
            kept.append((row, len(screen["elements"])))
    shutil.rmtree(tmp)
    stats = {"pool_steps": len(keys), "pool_episodes": len({k[0] for k in keys}), "shards_read": sorted(shards),
             "sampled": len(sample), "sampled_episodes": len({k[0] for k in sample}), "kept": len(kept),
             "dropped": dict(dropped.most_common()), "kept_by_action": dict(collections.Counter(r["action"] for r, _ in kept)),
             "type_target": dict(collections.Counter(r.get("type_target") for r, _ in kept if r["action"] == "type")),
             "elements_per_screen_median": statistics.median(n for _, n in kept) if kept else 0,
             "elements_per_screen_mean": round(statistics.mean(n for _, n in kept), 1) if kept else 0,
             "two_target_ids": sum(len(r["target_ids"]) > 1 for r, _ in kept)}
    # tap steps in the whole sample (not just kept) that fell inside no element
    taps = sum((pool[k][0] is not None and pool[k][0]["action"] != "type")
               or pool[k][1] in ("tap inside no element", "tap off screen", "tap on keyboard") for k in sample)
    stats["tap_steps_sampled"] = taps
    stats["tap_inside_no_element_share"] = round(dropped["tap inside no element"] / max(taps, 1), 4)
    (OUT / "stats.json").write_text(json.dumps(stats, indent=1))
    log(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
