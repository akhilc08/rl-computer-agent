"""Patch vendored Kev so every forward pass pads to a multiple of KEV_PAD_BUCKET tokens.

Qwen3.5's linear-attention kernels (flash-linear-attention, Triton) autotune per sequence length; with our
varied prompt lengths that recompiled on nearly every example (5.4 s/example on an A10G). Bucketing to 256
tokens plus a warm Triton cache brought it to 1.7 s/example. Pads go after every real token and are masked.

Setup on a CUDA box (Kev's venv, torch 2.8 / Triton 3.4): uv pip install flash-linear-attention==0.4.1 and
causal-conv1d (built with CUDA_HOME=/usr/local/cuda-12.8 --no-build-isolation); fla 0.6 needs a newer Triton.

  python scripts/train/kev_pad_patch.py .vendor/kev/kev/model.py
"""
import sys
p = sys.argv[1]
s = open(p).read()
old = '''        if str(self.device) == "mps" and not self.training: L = -(-L // self.SHAPE_BUCKET) * self.SHAPE_BUCKET'''
new = '''        if str(self.device) == "mps" and not self.training: L = -(-L // self.SHAPE_BUCKET) * self.SHAPE_BUCKET
        # Any device, train or eval: the fla/Triton kernels autotune per sequence length, so unbucketed lengths recompile forever.
        elif self.PAD_BUCKET > 1: L = -(-L // self.PAD_BUCKET) * self.PAD_BUCKET'''
anchor = '''    SHAPE_BUCKET = int(os.environ.get("KEV_SHAPE_BUCKET", "64"))'''
assert old in s and anchor in s and "PAD_BUCKET" not in s
s = s.replace(old, new).replace(anchor, '''    PAD_BUCKET = int(os.environ.get("KEV_PAD_BUCKET", "0"))   # pad every pass to a multiple of this (0/1 = off)
''' + anchor)
open(p, "w").write(s)
print("patched", p)
