"""Pixel-only element detection with OmniParser v2: a YOLO icon/element detector and a
Florence-2 captioner fine-tuned to describe what each icon does.

Boxes come back in screen points, like AX frames and OCR lines. Settings follow
OmniParser's demo (box threshold 0.05, IoU 0.1, 64x64 crops, "<CAPTION>" prompt).
"""
import glob
from pathlib import Path

import torch

DETECT_GLOB = str(Path.home() / ".cache/huggingface/hub/models--microsoft--OmniParser-v2.0/snapshots/*/icon_detect/model.pt")
CAPTION_DIR = Path.home() / ".cache/rlca/omniparser-caption"   # written by scripts/setup/convert_omniparser_caption.py


def device():
    return "mps" if torch.backends.mps.is_available() else "cpu"


class Detector:
    def __init__(self, imgsz=1280, conf=0.05, iou=0.1):
        from ultralytics import YOLO
        self.model = YOLO(glob.glob(DETECT_GLOB)[0])
        self.imgsz, self.conf, self.iou = imgsz, conf, iou

    def boxes(self, image, frame, scale):
        """-> list of {"box": (x, y, w, h) in screen points, "conf"} for a window image."""
        r = self.model.predict(image, imgsz=self.imgsz, conf=self.conf, iou=self.iou, device=device(), verbose=False)[0]
        fx, fy = frame[0], frame[1]
        out = []
        for (x1, y1, x2, y2), c in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist()):
            out.append({"box": (round(fx + x1 / scale), round(fy + y1 / scale), round((x2 - x1) / scale), round((y2 - y1) / scale)),
                        "pixels": (int(x1), int(y1), int(x2), int(y2)), "conf": round(c, 3)})
        return out


class Captioner:
    def __init__(self, dtype=torch.float16):
        from transformers import AutoProcessor, Florence2ForConditionalGeneration
        self.processor = AutoProcessor.from_pretrained(CAPTION_DIR)
        self.model = Florence2ForConditionalGeneration.from_pretrained(CAPTION_DIR, dtype=dtype).to(device()).eval()
        self.dtype = dtype

    @torch.inference_mode()
    def captions(self, image, pixel_boxes, batch=64, max_new_tokens=20):
        """-> one short description per (x1, y1, x2, y2) pixel box of `image`."""
        crops = [image.crop(b).resize((64, 64)) for b in pixel_boxes]
        out = []
        for i in range(0, len(crops), batch):
            chunk = crops[i:i + batch]
            enc = self.processor(text=["<CAPTION>"] * len(chunk), images=chunk, return_tensors="pt").to(device())
            enc["pixel_values"] = enc["pixel_values"].to(self.dtype)
            ids = self.model.generate(**enc, max_new_tokens=max_new_tokens, num_beams=1, do_sample=False)
            out += [t.strip() for t in self.processor.batch_decode(ids, skip_special_tokens=True)]
        return out
