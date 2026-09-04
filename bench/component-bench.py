#!/usr/bin/env python3
"""Time each component of one decode step in isolation (each timed with its own eval, 200 iterations, median),
then scale by layer counts and compare to the measured step. usage: component-bench.py <model dir>"""
import sys, time, statistics
import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache
model, tok = load(sys.argv[1])
inner = getattr(model, "model", None) or getattr(model, "language_model", None); layers = inner.layers
ids = tok.encode("Explain how a prompt cache works in an LLM server. " * 20)[:256]
cache = make_prompt_cache(model); x = mx.array(ids)[None]
logits = model(x, cache=cache); mx.eval(logits)
tokn = mx.argmax(logits[:, -1, :], axis=-1)
def timeit(fn, n=200):
    for _ in range(10): mx.eval(fn())
    ts = []
    for _ in range(n):
        t0 = time.perf_counter(); mx.eval(fn()); ts.append(time.perf_counter() - t0)
    return 1000 * statistics.median(ts)
h = inner.embed_tokens(tokn[:, None]); mx.eval(h)
L = layers[0]
attn_layers = [l for l in layers]
moe_layers = [l for l in layers if type(l.mlp).__name__ == "MoE"]
dense_layers = [l for l in layers if type(l.mlp).__name__ != "MoE"]
# masks/cache for attention at current length
from mlx_lm.models.base import create_attention_mask
mask = create_attention_mask(h, cache[0]) if hasattr(cache[0], "offset") else None
t_norm = timeit(lambda: L.input_layernorm(h))
t_attn_full = timeit(lambda: [l for l in layers if not getattr(l, "use_sliding", False)][0].self_attn(h, None, None))
t_attn_swa = timeit(lambda: [l for l in layers if getattr(l, "use_sliding", False)][0].self_attn(h, None, None)) if any(getattr(l,"use_sliding",False) for l in layers) else 0
t_moe = timeit(lambda: moe_layers[0].mlp(h)) if moe_layers else 0
t_dense = timeit(lambda: dense_layers[0].mlp(h)) if dense_layers else 0
t_head = timeit(lambda: model.lm_head(inner.norm(h)) if hasattr(model, "lm_head") else inner.embed_tokens.as_linear(inner.norm(h)))
t_step = timeit(lambda: mx.argmax(model(tokn[:, None], cache=make_prompt_cache(model))[:, -1, :], axis=-1), n=20)
n_full = sum(1 for l in layers if not getattr(l, "use_sliding", False)); n_swa = len(layers) - n_full
est = len(layers) * 2 * t_norm + n_full * t_attn_full + n_swa * t_attn_swa + len(moe_layers) * t_moe + len(dense_layers) * t_dense + t_head
print(f"{sys.argv[1].split('/')[-1]}: layers={len(layers)} (full-attn {n_full}, sliding {n_swa}; moe {len(moe_layers)}, dense {len(dense_layers)})")
print(f"  rmsnorm            {t_norm:6.3f} ms  x{2*len(layers)}")
print(f"  attention (full)   {t_attn_full:6.3f} ms  x{n_full}   [no cache: attention over 1 token only]")
print(f"  attention (swa)    {t_attn_swa:6.3f} ms  x{n_swa}")
print(f"  moe block          {t_moe:6.3f} ms  x{len(moe_layers)}")
print(f"  dense mlp          {t_dense:6.3f} ms  x{len(dense_layers)}")
print(f"  final norm+lm_head {t_head:6.3f} ms  x1")
print(f"  sum of isolated components = {est:.2f} ms  vs measured full step (fresh cache) {t_step:.2f} ms")
