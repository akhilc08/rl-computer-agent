"""Time one decision pass (prefill only, no generation) vs. number of candidates.

Builds a synthetic element list of N lines, runs a single forward pass, and reads
the next-token distribution over the A-Z labels, the way milestone 1 would.

Usage: python scripts/probes/prefill_latency.py MODEL [MODEL ...]
"""
import string
import sys
import time

import mlx.core as mx
from mlx_lm import load

ROLES = ["button", "menu item", "text field", "checkbox", "row", "tab", "link"]
WORDS = ["Playlist", "Library", "Search", "Home", "Settings", "Share", "Delete",
         "Rename", "Download", "Add to queue", "Liked Songs", "Profile", "Back"]


def build_prompt(n):
    lines = []
    for i in range(n):
        label = string.ascii_uppercase[i % 26] + (str(i // 26) if i >= 26 else "")
        lines.append(f"[{label}] {ROLES[i % len(ROLES)]} \"{WORDS[i % len(WORDS)]} {i}\" "
                     f"(section: {WORDS[(i * 7) % len(WORDS)]})")
    return ("Pick the element the user means.\nCommand: create a new playlist\n"
            "Elements:\n" + "\n".join(lines) + "\nAnswer: [")


def bench(name, sizes=(10, 26, 100, 300), reps=3):
    model, tok = load(name)
    letter_ids = [tok.encode(c, add_special_tokens=False)[-1] for c in string.ascii_uppercase]
    print(f"\n## {name}")
    for n in sizes:
        ids = mx.array([tok.encode(build_prompt(n))])
        model(ids)  # warm-up
        mx.eval(model(ids))
        times = []
        for _ in range(reps):
            t0 = time.perf_counter()
            logits = model(ids)[0, -1]
            probs = mx.softmax(logits[mx.array(letter_ids)], axis=-1)
            mx.eval(probs)
            times.append((time.perf_counter() - t0) * 1000)
        print(f"  N={n:4d}  tokens={ids.shape[1]:5d}  median={sorted(times)[reps // 2]:7.0f}ms")


if __name__ == "__main__":
    for name in sys.argv[1:]:
        try:
            bench(name)
        except Exception as e:
            print(f"\n## {name}: failed ({type(e).__name__}: {str(e)[:150]})")
