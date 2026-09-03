#!/usr/bin/env python3
"""Decode step time vs batch size. If exec time barely grows with batch, the step is launch/latency-bound, not bandwidth-bound."""
import sys, time, statistics
import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache
model, tok = load(sys.argv[1])
ids = tok.encode("Explain how a prompt cache works in an LLM server. " * 20)[:256]
out = []
for B in (1, 2, 4, 8, 16):
    cache = make_prompt_cache(model)
    x = mx.array([ids] * B)
    logits = model(x, cache=cache); mx.eval(logits)
    tokn = mx.argmax(logits[:, -1, :], axis=-1)
    mx.eval(tokn); times = []
    for i in range(40):
        t0 = time.perf_counter()
        logits = model(tokn[:, None], cache=cache); nxt = mx.argmax(logits[:, -1, :], axis=-1); mx.eval(nxt)
        if i >= 8: times.append(time.perf_counter() - t0)
        tokn = nxt
    ms = 1000 * statistics.median(times)
    out.append((B, ms))
    print(f"  B={B:2d}: step {ms:6.2f} ms  -> {B*1000/ms:6.1f} tok/s aggregate, {1000/ms:5.1f} per stream", flush=True)
b1 = out[0][1]
print(f"{sys.argv[1].split('/')[-1]}: step(B=8)/step(B=1) = {out[3][1]/b1:.2f}x  (1.0 = pure overhead-bound, 8.0 = pure compute/bandwidth-bound)")
