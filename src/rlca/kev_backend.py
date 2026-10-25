"""Kev as the decision model, quantized to 4-bit with MLX.

Kev's MLX path loads the bf16 base and merges the LoRA; we then quantize the
merged backbone in place. The pointer head stays fp32 (it is built before
quantization, from the unquantized embedding width).
"""
import time

ELEMENT_QUESTION = "Which on-screen element should be acted on to carry out the user's command?"


def load_kev(run, bits=4, group_size=64):
    import mlx.core as mx
    import mlx.nn as nn
    from kev.checkpoint import Checkpoint, LoadOptions

    tok, model = Checkpoint(run).load("mps", LoadOptions(backend="mlx"))
    if bits:
        nn.quantize(model.lm, group_size=group_size, bits=bits,
                    class_predicate=lambda _, m: hasattr(m, "to_quantized") and m.weight.shape[-1] % group_size == 0)
        mx.eval(model.lm.parameters())
        mx.clear_cache()
    mx.reset_peak_memory()
    return tok, model


LAYOUTS = ("state_title", "state", "instr")


def build_prompt(app, window_title, command, layout="state_title"):
    """-> (state, question instructions) for one command.

    state_title: command and window title in the state (first baseline; the title pulls the model toward elements
                 that share its words). state: same without the title. instr: the command in the question itself.
    """
    if layout == "state_title":
        return f"App: {app}\nWindow: {window_title or '(none)'}\nUser command: {command}", ELEMENT_QUESTION
    if layout == "state":
        return f"App: {app}\nUser command: {command}", ELEMENT_QUESTION
    if layout == "instr":
        return f"App: {app}", f'The user said: "{command}". Which on-screen element should be acted on?'
    raise ValueError(f"unknown layout {layout!r}; expected one of {LAYOUTS}")


def score_options(tok, model, state, options, instructions=ELEMENT_QUESTION):
    """-> (probabilities as a list, model milliseconds, prompt tokens) for one choice question."""
    import mlx.core as mx
    from kev.model import SERVE_MAX_BRANCH, SERVE_MAX_STATE, encode

    rec = {"state": state, "questions": [{"instr": instructions, "options": options, "label": 0}]}
    enc = encode(tok, rec, max_state=SERVE_MAX_STATE, max_branch=SERVE_MAX_BRANCH)
    t0 = time.perf_counter()
    probs = model.probs(enc)[0]
    mx.synchronize()
    ms = (time.perf_counter() - t0) * 1000
    return probs.tolist(), ms, len(enc["ids"])
