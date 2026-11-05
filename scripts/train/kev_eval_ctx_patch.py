"""Patch vendored Kev's benchmark so --data records can be scored at the serving context (KEV_EVAL_CONTEXT=serving).

The default --data context is the 1,024-token training branch limit, which skips our element questions
(~3,000 tokens: every option on screen). Serving accepts them, so eval should too.

  python scripts/train/kev_eval_ctx_patch.py .vendor/kev/kev/benchmark.py
"""
import sys

p = sys.argv[1]
s = open(p).read()
old = '''        context, skip_overlong = CONTEXT, True'''
new = '''        context, skip_overlong = (SERVING_CONTEXT if os.environ.get("KEV_EVAL_CONTEXT") == "serving" else CONTEXT), True'''
assert old in s and "KEV_EVAL_CONTEXT" not in s
s = s.replace(old, new)
if "SERVING_CONTEXT" not in s.split(new)[0]:
    s = s.replace("from kev.suite import ", "from kev.suite import SERVING_CONTEXT, ", 1)
if "\nimport os" not in s:
    s = "import os\n" + s
open(p, "w").write(s)
print("patched", p)
