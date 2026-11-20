"""Run open screenshot-grounding VLMs on our benchmark examples, to compare with Kev on accuracy, speed, calibration.

Each model gets its official grounding prompt, image budget and output parsing (sources next to each one below).
Predictions are mapped back to original image pixels; a hit is the point inside gt_box. OmniAct rows with only a
click point (no box) are skipped and counted. Batch size 1, greedy, bf16, flash-attention if installed else sdpa.
Latency is GPU-synchronised wall time per example from prompt building to parsed point (image decode excluded);
the first 3 examples of a run are warm-up and left out of the speed summary. Confidence for calibration:
Qwen3-VL / UI-Venus = geometric-mean probability of the generated digit tokens; GUI-Actor = pointer-attention
mass of the chosen region. Rows append, so a rerun resumes.

Writes runs/vlm/<model>/<set>/rows.jsonl and summary.json. GPU box only (not the Mac):
  python scripts/eval/vlm_grounding.py --model venus-2b --sets screenspot_v2 screenspot_pro [--limit 50]
  models: venus-2b venus-8b qwen3vl-2b showui-2b (need transformers>=4.57, qwen-vl-utils; qwen3vl also qwen-agent and
          the Qwen3-VL repo's cookbooks/ via --qwen_cookbook)
          gui-actor-2b (own venv: transformers==4.51.3, qwen-vl-utils==0.0.8, accelerate,
          `pip install -e GUI-Actor --no-deps`; its pointer-head code does not run on transformers>=4.52)
          tinyclick (Florence-2 remote code: same venv as gui-actor, plus timm and einops)
  sets: the five public sets (default) and our Mac labels: mac_val (records val.jsonl) and neetcode
        (heldout_site.jsonl), built from the records + screens.jsonl + explorer states (see mac_examples);
        macbench: MacBench v1's pixels track (data/macbench/v1, see macbench_examples). Absent-target commands are
        kept: a refusal counts as the right answer there, any point as wrong.
"""
import argparse
import importlib.util
import json
import math
import re
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from score_kev import coverage_at, ece  # noqa: E402

SETS = ["screenspot_v2", "screenspot_pro", "osworld_g", "mmbench_gui", "omniact", "androidcontrol", "weblinx"]
MODELS = {"venus-2b": ("inclusionAI/UI-Venus-1.5-2B", "venus"), "venus-8b": ("inclusionAI/UI-Venus-1.5-8B", "venus"),
          "qwen3vl-2b": ("Qwen/Qwen3-VL-2B-Instruct", "qwen3vl"), "gui-actor-2b": ("microsoft/GUI-Actor-2B-Qwen2-VL", "gui_actor"),
          "showui-2b": ("showlab/ShowUI-2B", "showui"), "tinyclick": ("Krystianz/TinyClick", "tinyclick")}
WARMUP = 3
RECORDS = {"screenspot_v2": "eval_ssv2", "screenspot_pro": "eval_sspro", "osworld_g": "eval_osworldg",
           "mmbench_gui": "eval_mmbench", "omniact": "eval_omniact", "androidcontrol": "eval_androidcontrol", "weblinx": "eval_weblinx"}
MAC_SETS = {"mac_val": "val", "neetcode": "heldout_site"}     # our Mac labels: set -> records file (see mac_examples)


def attn_impl():
    return "flash_attention_2" if importlib.util.find_spec("flash_attn") else "sdpa"


def digit_conf(model, tok, out, n_prompt):
    """Geometric-mean probability of the generated tokens that carry coordinate digits (None if there are none)."""
    lp = model.compute_transition_scores(out.sequences, out.scores, normalize_logits=True)[0]
    ids = out.sequences[0, n_prompt:]
    picked = [float(lp[i]) for i, t in enumerate(ids.tolist()) if re.search(r"\d", tok.decode([t]))]
    return math.exp(sum(picked) / len(picked)) if picked else None


GREEDY = dict(do_sample=False, temperature=None, top_p=None, top_k=None, return_dict_in_generate=True, output_scores=True)


# ---- UI-Venus-1.5 (Qwen3-VL based) ----
# Source: https://github.com/inclusionAI/UI-Venus/blob/UI-Venus-1.5/models/grounding/ui_venus1_5_gd.py and
# eval_screenspot_pro.py on that branch (the HF cards only show vLLM serving). User-only message, image then text,
# trailing "." stripped from the instruction, refusal variant of the prompt (the eval default), greedy, 128 new
# tokens. Output "[x,y]" in 0-1000 normalised coords (also accepts a box or two points -> centre); [-1,-1] = refusal.
# Image budget: their code passes nothing, so qwen_vl_utils' default max_pixels (16384*28*28) applies.
VENUS_PROMPT = ("Output the center point of the position corresponding to the following instruction: \n{}. \n\n"
                "The output should just be the coordinates of a point, in the format [x,y]. Additionally, if the task "
                "is infeasible (e.g., the task is not related to the image), the output should be [-1,-1].")


def parse_venus(text):
    """-> (x, y) in 0-1000, (-1, -1) for a refusal, or None. Mirrors UIVenusGroundV15._parse_point."""
    t = text.strip()
    two = re.fullmatch(r"\[\s*(-?\d+)\s*,\s*(-?\d+)\s*\],\s*\[\s*(-?\d+)\s*,\s*(-?\d+)\s*\]", t.replace(" ", ""))
    if two:
        a = [int(v) for v in two.groups()]
        return (a[0] + a[2]) / 2, (a[1] + a[3]) / 2
    m = re.search(r"\[([^\]]*)\]", t)
    try:
        nums = [float(v) for v in m.group(1).split(",")] if m else []
    except ValueError:
        return None
    if len(nums) == 4:
        return (nums[0] + nums[2]) / 2, (nums[1] + nums[3]) / 2
    return tuple(nums) if len(nums) == 2 else None


def venus(repo, args):
    import torch
    from qwen_vl_utils import process_vision_info
    from transformers import AutoModelForImageTextToText, AutoProcessor
    model = AutoModelForImageTextToText.from_pretrained(repo, dtype=torch.bfloat16, attn_implementation=attn_impl(),
                                                        device_map="cuda").eval()
    proc = AutoProcessor.from_pretrained(repo)
    max_px = args.max_pixels or 16384 * 28 * 28

    def predict(img, instruction, _set):
        image = {"type": "image", "image": img, "max_pixels": max_px}
        msgs = [{"role": "user", "content": [image, {"type": "text", "text": VENUS_PROMPT.format(instruction.removesuffix("."))}]}]
        text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        images, videos = process_vision_info(msgs)
        inputs = proc(text=[text], images=images, videos=videos, padding=True, return_tensors="pt").to(model.device)
        out = model.generate(**inputs, max_new_tokens=128, **GREEDY)
        n = inputs.input_ids.shape[1]
        raw = proc.batch_decode(out.sequences[:, n:], skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        p = parse_venus(raw)
        status = "unparsed" if p is None else "refused" if p == (-1, -1) else "ok"
        pt = (p[0] / 1000 * img.width, p[1] / 1000 * img.height) if status == "ok" else None
        return {"point": pt, "status": status, "raw": raw, "_conf": lambda: digit_conf(model, proc.tokenizer, out, n)}
    return predict, {"max_pixels": max_px, "conf": "digit_token_prob"}


# ---- Qwen3-VL-2B-Instruct (plain baseline) ----
# Source: https://github.com/QwenLM/Qwen3-VL/blob/main/cookbooks/computer_use.ipynb ("Computer Use with Local
# Model"). System "You are a helpful assistant." plus the computer_use tool (cookbooks/utils/agent_function_call.py)
# rendered by qwen_agent's NousFnCallPrompt with display 1000x1000; user message = instruction text then image.
# Output: <tool_call>{"name": "computer_use", "arguments": {"action": ..., "coordinate": [x, y]}}</tool_call>,
# coordinate in 0-1000 relative space (the cookbook maps it with x/1000*width). Image budget: the cookbook's
# 16..6400 visual tokens (patch 16, merge 2 -> 32x32 px per token); the cookbook caps generation at 2048 tokens.
def qwen3vl(repo, args):
    import torch
    from qwen_agent.llm.fncall_prompts.nous_fncall_prompt import ContentItem, Message, NousFnCallPrompt
    from transformers import AutoModelForImageTextToText, AutoProcessor
    sys.path.insert(0, str(Path(args.qwen_cookbook).expanduser()))
    from utils.agent_function_call import ComputerUse
    model = AutoModelForImageTextToText.from_pretrained(repo, dtype=torch.bfloat16, attn_implementation=attn_impl(),
                                                        device_map="cuda").eval()
    proc = AutoProcessor.from_pretrained(repo)
    unit = proc.image_processor.patch_size ** 2 * proc.image_processor.merge_size ** 2
    min_px, max_px = unit * 16, args.max_pixels or unit * 6400
    tool = ComputerUse(cfg={"display_width_px": 1000, "display_height_px": 1000}).function

    def predict(img, instruction, _set):
        msgs = NousFnCallPrompt().preprocess_fncall_messages(
            messages=[Message(role="system", content=[ContentItem(text="You are a helpful assistant.")]),
                      Message(role="user", content=[ContentItem(text=instruction), ContentItem(image="file://screen.png")])],
            functions=[tool], lang=None)
        text = proc.apply_chat_template([m.model_dump() for m in msgs], tokenize=False, add_generation_prompt=True)
        inputs = proc(text=[text], images=[img], padding=True, return_tensors="pt",
                      size={"shortest_edge": min_px, "longest_edge": max_px}).to(model.device)
        out = model.generate(**inputs, max_new_tokens=2048, **GREEDY)
        n = inputs.input_ids.shape[1]
        raw = proc.batch_decode(out.sequences[:, n:], skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]
        try:
            xy = json.loads(raw.split("<tool_call>\n")[1].split("\n</tool_call>")[0])["arguments"]["coordinate"]
        except (IndexError, KeyError, TypeError, json.JSONDecodeError):    # tolerate a malformed wrapper
            m = re.search(r'"coordinate"\s*:\s*\[\s*(-?[\d.]+)\s*,\s*(-?[\d.]+)', raw)
            xy = [float(m.group(1)), float(m.group(2))] if m else None
        pt = (float(xy[0]) / 1000 * img.width, float(xy[1]) / 1000 * img.height) if xy and len(xy) >= 2 else None
        return {"point": pt, "status": "ok" if pt else "unparsed", "raw": raw,
                "_conf": lambda: digit_conf(model, proc.tokenizer, out, n)}
    return predict, {"max_pixels": max_px, "conf": "digit_token_prob"}


# ---- ShowUI-2B (Qwen2-VL) ----
# Source: https://huggingface.co/showlab/ShowUI-2B (README grounding example). One user message: the system text,
# the image (min 256*28*28, max 1344*28*28 pixels), then the query. Output "[x, y]" in 0-1 relative coords.
SHOWUI_SYSTEM = ("Based on the screenshot of the page, I give a text description and you give its corresponding location. "
                 "The coordinate represents a clickable location [x, y] for an element, which is a relative coordinate "
                 "on the screenshot, scaled from 0 to 1.")


def showui(repo, args):
    import torch
    from qwen_vl_utils import process_vision_info
    from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
    min_px, max_px = 256 * 28 * 28, args.max_pixels or 1344 * 28 * 28
    model = Qwen2VLForConditionalGeneration.from_pretrained(repo, dtype=torch.bfloat16, attn_implementation=attn_impl(),
                                                            device_map="cuda").eval()
    proc = AutoProcessor.from_pretrained(repo, min_pixels=min_px, max_pixels=max_px)

    def predict(img, instruction, _set):
        msgs = [{"role": "user", "content": [{"type": "text", "text": SHOWUI_SYSTEM},
                                             {"type": "image", "image": img, "min_pixels": min_px, "max_pixels": max_px},
                                             {"type": "text", "text": instruction}]}]
        text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        images, videos = process_vision_info(msgs)
        inputs = proc(text=[text], images=images, videos=videos, padding=True, return_tensors="pt").to(model.device)
        out = model.generate(**inputs, max_new_tokens=128, **GREEDY)
        n = inputs.input_ids.shape[1]
        raw = proc.batch_decode(out.sequences[:, n:], skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        p = parse_venus(raw)    # same "[x, y]" shape, but 0-1
        pt = (p[0] * img.width, p[1] * img.height) if p else None
        return {"point": pt, "status": "ok" if pt else "unparsed", "raw": raw,
                "_conf": lambda: digit_conf(model, proc.tokenizer, out, n)}
    return predict, {"max_pixels": max_px, "conf": "digit_token_prob"}


# ---- GUI-Actor-2B (Qwen2-VL + pointer head) ----
# Source: https://huggingface.co/microsoft/GUI-Actor-2B-Qwen2-VL and https://github.com/microsoft/GUI-Actor
# (eval/screenSpot_pro.py, src/gui_actor/inference.py). Qwen2-VL system message below, user = image then the raw
# instruction, gui_actor's chat_template, use_placeholder=True (the assistant turn is pre-filled with
# pyautogui.click(<|pointer_start|><|pointer_pad|><|pointer_end|>) and the pointer head attends over image patches).
# Output: topk_points as 0-1 normalised (x, y) region centres; top-1 is the prediction. Image budget: the
# processor's max_pixels 5720064 (preprocessor_config.json); for ScreenSpot-Pro the official eval first resizes
# every screenshot to 3200*1800 pixels (--resize_to_pixels default), which we copy for that set only.
GUI_ACTOR_SYSTEM = ("You are a GUI agent. You are given a task and a screenshot of the screen. You need to perform a "
                    "series of pyautogui actions to complete the task.")


def gui_actor(repo, args):
    import torch
    from gui_actor.constants import DEFAULT_POINTER_END_TOKEN, DEFAULT_POINTER_PAD_TOKEN
    from gui_actor.inference import ForceFollowTokensLogitsProcessor, inference
    from gui_actor.modeling import Qwen2VLForConditionalGenerationWithPointer
    from transformers import AutoProcessor
    proc = AutoProcessor.from_pretrained(repo)
    tok = proc.tokenizer
    model = Qwen2VLForConditionalGenerationWithPointer.from_pretrained(
        repo, torch_dtype=torch.bfloat16, device_map="cuda:0", attn_implementation=attn_impl()).eval()
    lp = ForceFollowTokensLogitsProcessor(token_a_id=tok.encode(DEFAULT_POINTER_PAD_TOKEN)[0],
                                          forced_sequence=[tok.encode(DEFAULT_POINTER_END_TOKEN)[0]])
    max_px = args.max_pixels or proc.image_processor.max_pixels

    def prepare(img, set_name):
        if set_name == "screenspot_pro" and img.width * img.height != 3200 * 1800:
            r = (3200 * 1800 / (img.width * img.height)) ** 0.5
            return img.resize((int(img.width * r), int(img.height * r)))
        return img

    def predict(img, instruction, _set):
        conv = [{"role": "system", "content": [{"type": "text", "text": GUI_ACTOR_SYSTEM}]},
                {"role": "user", "content": [{"type": "image", "image": img, "max_pixels": max_px},
                                             {"type": "text", "text": instruction}]}]
        pred = inference(conv, model, tok, proc, logits_processor=lp, use_placeholder=True, topk=3)
        if not pred["topk_points"]:
            return {"point": None, "status": "unparsed", "raw": pred["output_text"]}
        x, y = pred["topk_points"][0]
        mass = pred["topk_values"][0] * len(pred["topk_points_all"][0])     # region mean attention x patches
        return {"point": None, "norm": (x, y), "status": "ok", "raw": pred["output_text"], "conf": mass,
                "topk": [[round(a, 4), round(b, 4)] for a, b in pred["topk_points"]],
                "topk_values": [round(v, 5) for v in pred["topk_values"]]}
    return predict, {"max_pixels": max_px, "conf": "pointer_region_mass", "prepare": prepare}


# ---- TinyClick (Florence-2-base, 0.27B) ----
# Source: the model card of the authors' MIT mirror https://huggingface.co/Krystianz/TinyClick (Samsung/TinyClick and
# github.com/SamsungLabs/TinyClick are no longer online; the card's usage code is unchanged, with the repo id swapped).
# Prompt: ("What to do to execute the command? " + command.strip()).lower(); processor(images, text, do_resize=True)
# (768x768); model.generate(**inputs) with the repo's generation_config (3 beams, no_repeat_ngram 3, max_length 20);
# decoded with skip_special_tokens=False. Output "click <loc_X><loc_Y>", X/Y in 0-999 bins of the image width and
# height; mapped with Florence-2's dequantisation (bin + 0.5) / 1000, as in the co-author's ONNX card
# (huggingface.co/sebpost/TinyClick-ONNX). The card runs fp32 (no dtype), so we do too. Florence-2's remote code
# (trust_remote_code) needs timm and einops and does not load on transformers>=4.52, so run it from the GUI-Actor
# venv (transformers 4.51.3). Confidence: the top beam's length-normalised sequence probability.
TINYCLICK_PROMPT = "What to do to execute the command? "


def parse_tinyclick(text):
    """-> (action, (x_bin, y_bin)) from the first <loc_X><loc_Y> pair, or (action, None)."""
    m = re.search(r"</s><s>(<[^>]+>|[^<\s]+)", text)
    p = re.search(r"<loc_(\d+)><loc_(\d+)>", text)
    return (m.group(1).strip() if m else None), ((int(p.group(1)), int(p.group(2))) if p else None)


TINYCLICK_MAX_TEXT = 400


def tinyclick(repo, args):
    from transformers import AutoModelForCausalLM, AutoProcessor
    proc = AutoProcessor.from_pretrained(repo, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(repo, trust_remote_code=True, attn_implementation="eager").to("cuda").eval()

    def predict(img, instruction, _set):
        text = (TINYCLICK_PROMPT + instruction.strip()).lower()
        ids = proc.tokenizer(text, add_special_tokens=False)["input_ids"]
        if len(ids) > TINYCLICK_MAX_TEXT:            # Florence-2's 1024 positions also hold 577 image tokens; long
            text = proc.tokenizer.decode(ids[:TINYCLICK_MAX_TEXT])   # WebLINX chat turns overflow it (CUDA assert)
        inputs = proc(images=img, text=text, return_tensors="pt", do_resize=True).to("cuda")
        out = model.generate(**inputs, return_dict_in_generate=True, output_scores=True)
        raw = proc.batch_decode(out.sequences, skip_special_tokens=False)[0]
        action, b = parse_tinyclick(raw)
        pt = ((b[0] + 0.5) / 1000 * img.width, (b[1] + 0.5) / 1000 * img.height) if b else None
        conf = math.exp(float(out.sequences_scores[0])) if getattr(out, "sequences_scores", None) is not None else None
        return {"point": pt, "status": "ok" if pt else "unparsed", "raw": raw, "conf": conf, "action": action}
    return predict, {"max_pixels": 768 * 768, "conf": "beam_sequence_prob", "dtype": "float32"}


def inside(box, pt):
    return box[0] <= pt[0] <= box[0] + box[2] and box[1] <= pt[1] <= box[1] + box[3]


def mac_examples(set_name, records_dir):
    """Our Mac sets, built from Kev's records (data/records_s/<file>.jsonl) + data/labels/screens.jsonl + the explorer
    states (data/{explore,native}/<run>/states.jsonl). instruction = the record's command; gt_boxes = the boxes of
    every gold target (a hit inside any counts). AX/OCR/detector boxes are in screen points; the screenshot is the
    window captured at `scale` pixels per point (frame[2] * scale == image width and frame[3] * scale == height on
    every labelled screen), so a box maps to pixels as ((x - frame_x) * scale, (y - frame_y) * scale, w * s, h * s),
    then is clipped to the image. Skipped and counted: target absent ("none"), no screenshot, gold targets with no
    box (menu-bar items) or entirely outside the window."""
    screens = {s["id"]: s for s in map(json.loads, open(ROOT / "data/labels/screens.jsonl"))}
    states, exs, skipped = {}, [], defaultdict(int)
    for r in map(json.loads, open(ROOT / records_dir / f"{MAC_SETS[set_name]}.jsonl")):
        q, meta = r["questions"]["element"], r["_meta"]
        s = screens.get(meta["screen"])
        targets = list(q.get("target") or ([q["label"]] if q["label"] != "none" else []))
        if not targets:
            skipped["absent"] += 1
            continue
        if not s or not s.get("shot") or not (ROOT / s["shot"]).exists():
            skipped["no_screenshot"] += 1
            continue
        run = (ROOT / s["shot"]).parent.parent
        if run not in states:
            states[run] = {f"{run.name}-{t['id']}": t for t in map(json.loads, open(run / "states.jsonl"))}
        st = states[run][s["id"]]
        (fx, fy, fw, fh), sc = st["frame"], st["scale"]
        W, H = round(fw * sc), round(fh * sc)
        boxes, by_id = [], {o["id"]: o for o in s["options"]}
        for t in targets:
            b = by_id.get(t, {}).get("box")
            if not b:
                continue
            x1, y1 = max(0, (b[0] - fx) * sc), max(0, (b[1] - fy) * sc)
            x2, y2 = min(W, (b[0] + b[2] - fx) * sc), min(H, (b[1] + b[3] - fy) * sc)
            if x2 > x1 and y2 > y1:
                boxes.append([round(x1, 1), round(y1, 1), round(x2 - x1, 1), round(y2 - y1, 1)])
        if not boxes:
            skipped["target_no_box" if all(not by_id.get(t, {}).get("box") for t in targets) else "target_offscreen"] += 1
            continue
        exs.append({"id": meta["id"], "shot": s["shot"], "instruction": r["state"].split("User command: ", 1)[1],
                    "gt_box": boxes[0], "gt_boxes": boxes, "platform": meta.get("site") or s.get("app"),
                    "screen": s["id"], "size": [W, H]})
    return exs, dict(skipped)


def macbench_examples(bench="data/macbench/v1"):
    """MacBench pixels-track commands as examples; gt_boxes are the targets' boxes in image pixels."""
    exs = []
    for c in map(json.loads, open(ROOT / bench / "commands.jsonl")):
        if "pixels" not in c["tracks"]:
            continue
        boxes = [b["box_px"] for b in c["target_boxes"] if b.get("box_px")]
        exs.append({"id": c["id"], "shot": f"{bench}/images/{c['screen']}.jpg", "instruction": c["command"],
                    "gt_box": boxes[0] if boxes else None, "gt_boxes": boxes, "absent": c["absent"],
                    "platform": c["app"], "screen": c["screen"]})
    return exs


def gt_box(ex, size):
    if ex.get("gt_box"):
        return ex["gt_box"]
    if ex.get("gt_norm"):
        x1, y1, x2, y2 = ex["gt_norm"]
        W, H = size
        return [x1 * W, y1 * H, (x2 - x1) * W, (y2 - y1) * H]
    return None


def summarize(rows, n_no_box, meta):
    hits = [r["hit"] for r in rows]
    lat = [r["latency_ms"] for r in rows if not r["warmup"]]
    plat = defaultdict(list)
    for r in rows:
        plat[r["platform"]].append(r["hit"])
    scored = [(r["conf"], r["hit"]) for r in rows if r.get("conf") is not None]
    confs, ch = [c for c, _ in scored], [h for _, h in scored]
    q = statistics.quantiles(lat, n=10, method="inclusive")[-1] if len(lat) >= 2 else (lat[0] if lat else None)
    return {**meta, "n": len(rows), "accuracy": sum(hits) / len(hits) if hits else None,
            "n_skipped_no_box": n_no_box, "n_unparsed": sum(r["status"] == "unparsed" for r in rows),
            "n_refused": sum(r["status"] == "refused" for r in rows),
            "latency_ms": {"n": len(lat), "median": statistics.median(lat) if lat else None, "p90": q},
            "per_platform": {p: {"n": len(h), "accuracy": sum(h) / len(h)} for p, h in sorted(plat.items())},
            "calibration": {"n": len(confs), "mean_conf": sum(confs) / len(confs) if confs else None,
                            "ece": ece(confs, ch) if confs else None,
                            "coverage_at_90": coverage_at(confs, ch, 0.90) if confs else None,
                            "coverage_at_95": coverage_at(confs, ch, 0.95) if confs else None}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=MODELS)
    ap.add_argument("--sets", nargs="+", default=SETS, choices=SETS + list(MAC_SETS) + ["macbench"])
    ap.add_argument("--limit", type=int, default=0, help="first N examples per set")
    ap.add_argument("--ids_from", help="keep only examples whose id is in this records dir (e.g. data/records_s, the fixed "
                    "500-example samples every model is scored on)")
    ap.add_argument("--splits", nargs="*", help="keep only these splits (e.g. eval test); default all")
    ap.add_argument("--data_root", default="data/public")
    ap.add_argument("--converted", default="data/converted")
    ap.add_argument("--out", default="runs/vlm")
    ap.add_argument("--max_pixels", type=int, default=0, help="override the model's image budget (e.g. to fit 24 GB)")
    ap.add_argument("--repo", help="load weights from here instead of the model's hub repo (e.g. a fine-tuned merge)")
    ap.add_argument("--kind", help="prompt/parse family to use instead of the model's own (venus for our fine-tunes)")
    ap.add_argument("--tag", help="output folder name instead of --model")
    ap.add_argument("--platforms", nargs="*", help="keep only these platforms (public sets)")
    ap.add_argument("--bench", default="data/macbench/v1", help="MacBench build for --sets macbench")
    ap.add_argument("--ids_file", help="keep only command ids listed in this file (one per line)")
    ap.add_argument("--qwen_cookbook", default="~/Qwen3-VL/cookbooks", help="Qwen3-VL repo cookbooks/ (has utils/)")
    args = ap.parse_args()
    import torch
    import transformers
    from PIL import Image
    repo, kind = MODELS[args.model]
    repo, kind = args.repo or repo, args.kind or kind
    predict, info = {"venus": venus, "qwen3vl": qwen3vl, "showui": showui, "gui_actor": gui_actor,
                     "tinyclick": tinyclick}[kind](repo, args)
    prepare = info.pop("prepare", lambda img, s: img)
    meta = {"model": repo, "attn": attn_impl(), "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
            "transformers": transformers.__version__, **info}
    data_root, ran = ROOT / args.data_root, 0
    for set_name in args.sets:
        mac_skipped = None
        if set_name == "macbench":
            exs = macbench_examples(args.bench)[: args.limit or None]
            print(f"macbench: {len(exs)} examples", flush=True)
        elif set_name in MAC_SETS:
            exs, mac_skipped = mac_examples(set_name, args.ids_from or "data/records_s")
            exs = exs[: args.limit or None]
            print(f"{set_name}: {len(exs)} examples, skipped {mac_skipped}", flush=True)
        else:
            src = ROOT / args.converted / set_name / "examples.jsonl"
            if not src.exists():
                print(f"{set_name}: no {src}, skipped", flush=True)
                continue
            exs = [json.loads(line) for line in open(src)]
            exs = [e for e in exs if not args.splits or e.get("split") in args.splits][: args.limit or None]
        if args.ids_from and set_name not in MAC_SETS and set_name != "macbench":
            rec = ROOT / args.ids_from / f"{RECORDS[set_name]}.jsonl"
            ids = {json.loads(line)["_meta"]["id"] for line in open(rec)}
            exs = [e for e in exs if e["id"] in ids or e["id"].removeprefix(f"{set_name}-") in ids]   # weblinx ids carry the set prefix
        if args.platforms:
            exs = [e for e in exs if e.get("platform") in args.platforms]
        if args.ids_file:
            keep_ids = set(open(args.ids_file).read().split())
            exs = [e for e in exs if e["id"] in keep_ids]
        out = ROOT / args.out / (args.tag or args.model) / set_name
        out.mkdir(parents=True, exist_ok=True)
        rows_path = out / "rows.jsonl"
        done = {json.loads(line)["id"] for line in open(rows_path)} if rows_path.exists() else set()
        n_no_box = 0
        with open(rows_path, "a") as f:
            for i, ex in enumerate(exs):
                if not ex.get("gt_box") and not ex.get("gt_norm") and not ex.get("absent"):
                    n_no_box += 1                    # OmniAct point-only answers: nothing to hit
                    continue
                if ex["id"] in done:
                    continue
                img = Image.open(ROOT / ex["shot"] if "shot" in ex else data_root / ex["image"]).convert("RGB")
                size = img.size                      # boxes and output points are in original pixels
                if "size" in ex and list(size) != ex["size"]:
                    print(f"{ex['id']}: image {size} but frame*scale {ex['size']}", flush=True)
                img = prepare(img, set_name)
                box = gt_box(ex, size)
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                with torch.inference_mode():
                    p = predict(img, ex["instruction"], set_name)
                torch.cuda.synchronize()
                ms = (time.perf_counter() - t0) * 1000
                if p.get("norm"):
                    p["point"] = (p["norm"][0] * size[0], p["norm"][1] * size[1])
                elif p["point"] and img.size != size:
                    p["point"] = (p["point"][0] * size[0] / img.width, p["point"][1] * size[1] / img.height)
                conf = p.pop("_conf")() if "_conf" in p else p.get("conf")
                pt = p["point"]
                row = {"id": ex["id"], "platform": ex.get("platform"), "point": [round(v, 1) for v in pt] if pt else None,
                       "hit": (pt is None) if ex.get("absent") else bool(pt and any(inside(b, pt) for b in ex.get("gt_boxes") or [box])),
                       "absent": bool(ex.get("absent")), "latency_ms": round(ms, 1), "warmup": ran < WARMUP,
                       "conf": conf, "status": p["status"], "raw": p["raw"], "gt_box": box, "size": list(size),
                       **{k: p[k] for k in ("topk", "topk_values", "action") if k in p},
                       **({"gt_boxes": ex["gt_boxes"], "screen": ex["screen"]} if "gt_boxes" in ex else {})}
                f.write(json.dumps(row) + "\n")
                f.flush()
                ran += 1
                if ran % 50 == 0:
                    print(f"{set_name} {i + 1}/{len(exs)}", flush=True)
        rows = [json.loads(line) for line in open(rows_path)]
        keep = {e["id"] for e in exs}
        s = summarize([r for r in rows if r["id"] in keep], n_no_box,
                      {**meta, "set": set_name, **({"skipped": mac_skipped} if mac_skipped is not None else {})})
        (out / "summary.json").write_text(json.dumps(s, indent=2))
        print(f"{set_name}: acc {s['accuracy']} n {s['n']} median {s['latency_ms']['median']} ms "
              f"p90 {s['latency_ms']['p90']} ms ece {s['calibration']['ece']}", flush=True)


if __name__ == "__main__":
    main()
