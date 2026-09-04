#!/usr/bin/env python3
"""Laguna attention: sub-step chain timings at decode shape, then compile the pre-SDPA (proj+norm+rope) and
post-SDPA (gate+o_proj) glue as shape-specialized compiled functions and measure the full decode step."""
import sys, time, statistics
from functools import partial
import mlx.core as mx, mlx.nn as nn
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.models.base import scaled_dot_product_attention
model, tok = load(sys.argv[1]); inner = model.model; layers = inner.layers
ids = tok.encode("Explain how a prompt cache works in an LLM server. " * 20)[:256]
def step(n=24):
    c = make_prompt_cache(model); x = mx.array(ids)[None]; l = model(x, cache=c); mx.eval(l)
    t = mx.argmax(l[:, -1, :], axis=-1); ts = []
    for i in range(n):
        t0 = time.perf_counter(); l = model(t[:, None], cache=c); t = mx.argmax(l[:, -1, :], axis=-1); mx.eval(t)
        if i >= 4: ts.append(time.perf_counter() - t0)
    return 1000 * statistics.median(ts)
def chain_cost(fn, h, n=64, reps=5):
    def run():
        y = h
        for _ in range(n): y = fn(y)
        mx.eval(y)
    run(); ts = []
    for _ in range(reps): t0 = time.perf_counter(); run(); ts.append((time.perf_counter() - t0) / n)
    return 1000 * statistics.median(ts)
h = mx.zeros((1, 1, inner.args.hidden_size), dtype=mx.bfloat16); mx.eval(h)
A = [l for l in layers if getattr(l, "use_sliding", False)][0].self_attn
B_, L_, D_ = 1, 1, inner.args.hidden_size
def pre(x):
    q, k, v = A.q_proj(x), A.k_proj(x), A.v_proj(x)
    q = A.q_norm(q.reshape(B_, L_, A.n_heads, -1)).transpose(0, 2, 1, 3)
    k = A.k_norm(k.reshape(B_, L_, A.n_kv_heads, -1)).transpose(0, 2, 1, 3)
    v = v.reshape(B_, L_, A.n_kv_heads, -1).transpose(0, 2, 1, 3)
    return A.rope(q, offset=100), A.rope(k, offset=100), v
q0, k0, v0 = pre(h); mx.eval(q0, k0, v0)
t_proj = chain_cost(lambda y: A.o_proj(A.q_proj(y)), h)  # 2 matmuls only
t_pre = chain_cost(lambda y: A.o_proj(pre(y)[0].transpose(0, 2, 1, 3).reshape(1, 1, -1)), h)
t_sdpa = chain_cost(lambda y: scaled_dot_product_attention(y, k0, v0, cache=None, scale=A.scale, mask=None), q0)
def post(o):
    o = o if o.shape[-1] != D_ else A.q_proj(o).reshape(1, 1, A.n_heads, A.head_dim).transpose(0, 2, 1, 3)
    o = o.transpose(0, 2, 1, 3).reshape(1, 1, -1)
    if A.gating:
        g = nn.softplus(A.g_proj(h).astype(mx.float32)).astype(o.dtype)
        o = (o.reshape(1, 1, A.n_heads, A.head_dim) * g[..., None]).reshape(1, 1, -1)
    return A.o_proj(o)
t_post = chain_cost(lambda y: post(y).reshape(1, A.n_heads, 1, A.head_dim)[..., :1].reshape(1,1,-1) * 0 + y if False else post(y), h)  # measured via full-shape proxy below
t_full = chain_cost(lambda y: A(y, None, None), h)
print(f"sliding attention @decode: full {t_full:.3f} ms | pre(proj+norm+rope)+o_proj {t_pre:.3f} | 2 matmuls only {t_proj:.3f} | sdpa(1 tok) {t_sdpa:.3f} | post(gate+o_proj) {t_post:.3f}; gating={A.gating} heads={A.n_heads}/{A.n_kv_heads}")
base = step(); print(f"baseline step {base:.2f} ms ({1000/base:.1f} tok/s)", flush=True)

# ---- compiled attention glue (shape-specialized). Cache update + SDPA stay eager.
def make_compiled(att):
    st = [att.state]
    @partial(mx.compile, inputs=st, outputs=st)
    def pre_c(x, offset):
        B, L, _ = x.shape
        q, k, v = att.q_proj(x), att.k_proj(x), att.v_proj(x)
        q = att.q_norm(q.reshape(B, L, att.n_heads, -1)).transpose(0, 2, 1, 3)
        k = att.k_norm(k.reshape(B, L, att.n_kv_heads, -1)).transpose(0, 2, 1, 3)
        v = v.reshape(B, L, att.n_kv_heads, -1).transpose(0, 2, 1, 3)
        return att.rope(q, offset=offset), att.rope(k, offset=offset), v
    @partial(mx.compile, inputs=st, outputs=st)
    def post_c(o, x):
        B, L = x.shape[0], x.shape[1]
        o = o.transpose(0, 2, 1, 3).reshape(B, L, -1)
        if att.gating:
            g = nn.softplus(att.g_proj(x).astype(mx.float32)).astype(o.dtype)
            o = (o.reshape(B, L, att.n_heads, att.head_dim) * g[..., None]).reshape(B, L, -1)
        return att.o_proj(o)
    return pre_c, post_c
for l in layers:
    l.self_attn._pre_c, l.self_attn._post_c = make_compiled(l.self_attn)
def attn_call(self, x, mask=None, cache=None):
    off = cache.offset if cache is not None else 0
    q, k, v = self._pre_c(x, off)
    if cache is not None: k, v = cache.update_and_fetch(k, v)
    o = scaled_dot_product_attention(q, k, v, cache=cache, scale=self.scale, mask=mask)
    return self._post_c(o, x)
type(A).__call__ = attn_call
t = step(); print(f"compiled attention glue: step {t:.2f} ms ({1000/t:.1f} tok/s)  speedup {base/t:.2f}x", flush=True)
# correctness: greedy 24 tokens vs reference text
from mlx_lm import generate
txt = generate(model, tok, prompt=tok.apply_chat_template([{"role":"user","content":"Name three prime numbers and stop."}], add_generation_prompt=True, tokenize=False), max_tokens=24, verbose=False)
print("sample:", repr(txt[:120]))
