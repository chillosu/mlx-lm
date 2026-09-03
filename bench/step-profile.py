#!/usr/bin/env python3
"""Per-token decode step: Python graph-build time vs GPU execution time.
usage: step-profile.py <model dir> [steps]"""
import sys, time, statistics
import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache
model, tok = load(sys.argv[1]); steps = int(sys.argv[2]) if len(sys.argv) > 2 else 48
ids = tok.encode("Explain how a prompt cache works in an LLM server. " * 20)[:256]
cache = make_prompt_cache(model)
x = mx.array(ids)[None]
logits = model(x, cache=cache); mx.eval(logits)          # prefill
tokn = mx.argmax(logits[:, -1, :], axis=-1)
mx.eval(tokn)
build, exec_, total = [], [], []
for i in range(steps + 8):
    t0 = time.perf_counter()
    logits = model(tokn[:, None], cache=cache)           # lazy: builds the graph
    nxt = mx.argmax(logits[:, -1, :], axis=-1)
    t1 = time.perf_counter()
    mx.eval(nxt)                                         # runs it
    t2 = time.perf_counter()
    tokn = nxt
    if i >= 8: build.append(t1 - t0); exec_.append(t2 - t1); total.append(t2 - t0)
ms = lambda v: 1000 * statistics.median(v)
print(f"{sys.argv[1].split('/')[-1]}: step median {ms(total):.2f} ms  = build {ms(build):.2f} ms (python, lazy graph)  + exec {ms(exec_):.2f} ms (gpu)   -> {1000/ms(total):.1f} tok/s; build share {100*ms(build)/ms(total):.0f}%")
# how many ops per step? count nodes in the lazy graph via a cheap proxy: number of primitive evaluations isn't exposed; report layer count
inner = getattr(model, "model", None) or getattr(model, "language_model", None)
layers = getattr(inner, "layers", None) or getattr(getattr(inner, "model", None), "layers", None)
print(f"  layers={len(layers)}  build per layer {ms(build)/len(layers):.3f} ms")
