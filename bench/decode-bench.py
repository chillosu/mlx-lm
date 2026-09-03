#!/usr/bin/env python3
"""Decode microbenchmark + per-module profile for one model on one node.
usage: python decode-bench.py <model dir> [--gen 256] [--profile]
Reports prompt tok/s, generation tok/s (steady state), and, with --profile,
time share per module class during decode (attention / moe-mlp / other)."""
import argparse, time, collections, sys
import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load, stream_generate

ap = argparse.ArgumentParser(); ap.add_argument("model"); ap.add_argument("--gen", type=int, default=256)
ap.add_argument("--profile", action="store_true"); ap.add_argument("--prompt-tokens", type=int, default=512)
a = ap.parse_args()

model, tok = load(a.model)
prompt = ("Explain, in detail and step by step, how a prompt cache works in an LLM server, " * 40)
ids = tok.encode(prompt)[: a.prompt_tokens]
prompt = tok.decode(ids)
msgs = [{"role": "user", "content": prompt}]
text = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)

def run(max_tokens):
    t0 = time.perf_counter(); first = None; n = 0; last = None
    for r in stream_generate(model, tok, text, max_tokens=max_tokens):
        n += 1
        if first is None: first = time.perf_counter()
        last = r
    t1 = time.perf_counter()
    return n, first - t0, t1 - first, last

# warmup
run(8)
n, t_prompt, t_gen, last = run(a.gen)
print(f"prompt_tokens={last.prompt_tokens} prompt_tps={last.prompt_tps:.1f}  gen_tokens={n} gen_tps_steady={ (n-1)/t_gen:.2f}  (mlx_lm reports {last.generation_tps:.2f})  peak_mem={mx.get_peak_memory()/2**30:.1f} GiB", flush=True)

if a.profile:
    # Wrap at the CLASS level: nn.Module.__call__ is looked up on the type, so instance patching is a no-op.
    buckets = collections.defaultdict(float); counts = collections.defaultdict(int); patched = set()
    def wrap_class(cls, name):
        if cls in patched: return
        patched.add(cls); orig = cls.__call__
        def wrapped(self, *args, **kw):
            t = time.perf_counter(); out = orig(self, *args, **kw)
            mx.eval(out if not isinstance(out, tuple) else out[0]); buckets[name] += time.perf_counter() - t; counts[name] += 1
            return out
        cls.__call__ = wrapped
    inner = getattr(model, "model", None) or getattr(model, "language_model", None)
    layers = getattr(inner, "layers", None) or getattr(getattr(inner, "model", None), "layers", None)
    L0 = layers[0]
    for attr in ("self_attn", "attn", "attention"):
        if hasattr(L0, attr): wrap_class(type(getattr(L0, attr)), "attention:" + type(getattr(L0, attr)).__name__); break
    for attr in ("mlp", "block_sparse_moe", "moe", "feed_forward"):
        if hasattr(L0, attr):
            seen = {type(getattr(L, attr)) for L in layers}
            for c in seen: wrap_class(c, "mlp:" + c.__name__)
            break
    for attr in ("input_layernorm", "post_attention_layernorm"):
        if hasattr(L0, attr): wrap_class(type(getattr(L0, attr)), "norm:" + type(getattr(L0, attr)).__name__); break
    if hasattr(model, "lm_head"): wrap_class(type(model.lm_head), "lm_head:" + type(model.lm_head).__name__)
    t0 = time.perf_counter(); n = 0
    for r in stream_generate(model, tok, text, max_tokens=64): n += 1
    total = time.perf_counter() - t0
    acc = sum(buckets.values())
    print(f"profile over {n} tokens (per-module sync inflates absolute times): total {total:.2f}s, attributed {acc:.2f}s, unattributed {total-acc:.2f}s")
    for k, v in sorted(buckets.items(), key=lambda x: -x[1]):
        print(f"  {k:36} {v:7.2f}s  {100*v/acc:5.1f}%  calls={counts[k]}")
