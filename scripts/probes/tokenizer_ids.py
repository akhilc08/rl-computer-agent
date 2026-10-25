"""Check which candidate-ID schemes are single tokens for each tokenizer.

Usage: python scripts/probes/tokenizer_ids.py
"""
import itertools
import string

from transformers import AutoTokenizer

TOKENIZERS = [
    "Qwen/Qwen3.5-4B",
    "Qwen/Qwen3-4B",
    "google/gemma-4-E4B-it",
    "meta-llama/Llama-3.2-3B-Instruct",
]


def n_tokens(tok, text):
    return len(tok.encode(text, add_special_tokens=False))


def report(name):
    try:
        tok = AutoTokenizer.from_pretrained(name)
    except Exception as e:  # gated or missing
        print(f"\n## {name}: could not load ({type(e).__name__}: {str(e)[:80]})")
        return
    print(f"\n## {name}")
    for n in ["7", "12", "37", "137", "999"]:
        ids = tok.encode(" " + n, add_special_tokens=False)
        print(f"  ' {n}' -> {len(ids)} tokens {tok.convert_ids_to_tokens(ids)}")
    # Largest N such that ' 1'..' N' are all single tokens (with leading space, after 'Answer:')
    max_single = 0
    for i in range(1, 1000):
        if n_tokens(tok, f"Answer: {i}") - n_tokens(tok, "Answer:") == 1:
            max_single = i
        else:
            break
    print(f"  numbers 1..N single-token after 'Answer:': N = {max_single}")
    for label, pool in [("A-Z", string.ascii_uppercase), ("a-z", string.ascii_lowercase)]:
        ok = sum(n_tokens(tok, f"Answer: {c}") - n_tokens(tok, "Answer:") == 1 for c in pool)
        print(f"  {label}: {ok}/{len(pool)} single-token")
    pairs = ["".join(p) for p in itertools.product(string.ascii_uppercase, repeat=2)]
    ok = sum(n_tokens(tok, f"Answer: {p}") - n_tokens(tok, "Answer:") == 1 for p in pairs)
    print(f"  two-letter AA-ZZ: {ok}/{len(pairs)} single-token")


if __name__ == "__main__":
    for name in TOKENIZERS:
        report(name)
