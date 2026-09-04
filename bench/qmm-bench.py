import time, statistics, mlx.core as mx
# GEMV at Laguna's q_proj shape (1x3072 @ 3072x9216) and o_proj (1x9216 @ 9216x3072), by quantization mode.
def bench(fn, n=200):
    for _ in range(20): mx.eval(fn())
    # chain-free: independent calls in one eval batch of 50 to amortize sync
    ts = []
    for _ in range(6):
        t0 = time.perf_counter(); outs = [fn() for _ in range(50)]; mx.eval(*outs); ts.append((time.perf_counter() - t0) / 50)
    return 1e3 * statistics.median(ts)
for (K, N) in ((3072, 9216), (9216, 3072), (3072, 3072)):
    x = mx.random.normal((1, 1, K)).astype(mx.bfloat16); w = mx.random.normal((N, K)).astype(mx.bfloat16); mx.eval(x, w)
    t_bf16 = bench(lambda: x @ w.T); mb = N * K * 2 / 1e6
    line = f"K={K} N={N}: bf16 {t_bf16:.4f} ms ({mb/t_bf16/1e3:.0f} GB/s)"
    for mode, bits, gs in (("affine", 4, 64), ("affine", 4, 32), ("nvfp4", 4, 16), ("affine", 8, 64)):
        try:
            qw, sc, *rest = mx.quantize(w, group_size=gs, bits=bits, mode=mode) if mode == "affine" else mx.quantize(w, group_size=gs, bits=bits, mode=mode)
            bs = rest[0] if rest else None
            kw = dict(group_size=gs, bits=bits, mode=mode)
            f = (lambda: mx.quantized_matmul(x, qw, sc, bs, transpose=True, **kw)) if bs is not None else (lambda: mx.quantized_matmul(x, qw, sc, transpose=True, **kw))
            t = bench(f); qmb = N * K * bits / 8 / 1e6
            line += f" | {mode}{bits}/g{gs} {t:.4f} ms ({qmb/t/1e3:.0f} GB/s eff)"
        except Exception as e: line += f" | {mode}{bits}/g{gs} ERR {str(e)[:40]}"
    print(line, flush=True)
