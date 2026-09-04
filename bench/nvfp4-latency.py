import time, statistics, mlx.core as mx
K, N = 3072, 9216; n = 64
x = mx.random.normal((1, 1, K)).astype(mx.bfloat16); w = mx.random.normal((N, K)).astype(mx.bfloat16); w2 = mx.random.normal((K, N)).astype(mx.bfloat16); mx.eval(x, w, w2)
def med(f, reps=7):
    f(); ts = []
    for _ in range(reps): t0 = time.perf_counter(); f(); ts.append(time.perf_counter() - t0)
    return 1e6 * statistics.median(ts)
for mode, gs, bits in (("affine", 64, 4), ("affine", 16, 4), ("nvfp4", 16, 4), ("mxfp4", 32, 4), ("affine", 64, 8)):
    try:
        q1 = mx.quantize(w, group_size=gs, bits=bits, mode=mode); q2 = mx.quantize(w2, group_size=gs, bits=bits, mode=mode); mx.eval(*q1, *q2)
        def g(v, q): return mx.quantized_matmul(v, *q, transpose=True, group_size=gs, bits=bits, mode=mode)
        def dep():
            y = x
            for _ in range(n): y = g(g(y, q1), q2)
            mx.eval(y)
        def ind():
            outs = [g(x, q1) for _ in range(2 * n)]; mx.eval(*outs)
        print(f"{mode:6} g{gs} b{bits}: dependent {med(dep)/(2*n):6.1f} us/kernel | independent {med(ind)/(2*n):6.1f} us/kernel", flush=True)
    except Exception as e: print(f"{mode} g{gs} b{bits}: ERR {str(e)[:80]}")
# and the actual Laguna q_proj/o_proj modules in a dependent chain
from mlx_lm import load
model, tok = load("/Users/chill/models/Laguna-S-2.1-NVFP4-mlx")
A = model.model.layers[1].self_attn
print("q_proj:", type(A.q_proj).__name__, getattr(A.q_proj, "mode", None), getattr(A.q_proj, "group_size", None), getattr(A.q_proj, "bits", None))
def dep_model():
    y = x
    for _ in range(n): y = A.o_proj(A.q_proj(y))
    mx.eval(y)
print(f"model q_proj+o_proj dependent chain: {med(dep_model)/(2*n):.1f} us/kernel")
