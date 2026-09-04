import time, statistics, mlx.core as mx
K, N = 3072, 9216
x = mx.random.normal((1, 1, K)).astype(mx.bfloat16); w = mx.random.normal((N, K)).astype(mx.bfloat16)
qw, sc, bs = mx.quantize(w, group_size=64, bits=4); w2 = mx.random.normal((K, N)).astype(mx.bfloat16); qw2, sc2, bs2 = mx.quantize(w2, group_size=64, bits=4)
mx.eval(x, qw, sc, bs, qw2, sc2, bs2)
def gemv(v, a, b, c): return mx.quantized_matmul(v, a, b, c, transpose=True, group_size=64, bits=4)
def med(f, reps=7):
    f(); ts = []
    for _ in range(reps): t0 = time.perf_counter(); f(); ts.append(time.perf_counter() - t0)
    return 1e6 * statistics.median(ts)
n = 64
def dep_chain():           # 64 dependent GEMV pairs (3072->9216->3072), like a layer stack
    y = x
    for _ in range(n): y = gemv(gemv(y, qw, sc, bs), qw2, sc2, bs2)
    mx.eval(y)
def indep():               # 128 independent GEMVs in one eval
    outs = [gemv(x, qw, sc, bs) for _ in range(n)] + [gemv(x[..., :K], qw, sc, bs) for _ in range(n)]
    mx.eval(*outs)
t_dep = med(dep_chain) / (2 * n); t_ind = med(indep) / (2 * n)
print(f"4-bit GEMV 3072x9216: dependent-chain {t_dep:.1f} us/kernel  vs independent {t_ind:.1f} us/kernel  (latency/throughput ratio {t_dep/t_ind:.1f}x)")
# fused qkv: one GEMV of N=9216+1024+1024 vs three GEMVs
wq, wk, wv = [mx.random.normal((nn_, K)).astype(mx.bfloat16) for nn_ in (9216, 1024, 1024)]
Q = [mx.quantize(w_, group_size=64, bits=4) for w_ in (wq, wk, wv)]; wf = mx.concatenate([wq, wk, wv], axis=0); F = mx.quantize(wf, group_size=64, bits=4); mx.eval(*[t for q in Q for t in q], *F)
def three(): 
    y = x
    for _ in range(n):
        a, b, c = gemv(y, *Q[0]), gemv(y, *Q[1]), gemv(y, *Q[2]); y = (a[..., :K] + b[..., :1] + c[..., :1])
    mx.eval(y)
def fused():
    y = x
    for _ in range(n):
        f = gemv(y, *F); y = f[..., :K] + f[..., 9216:9217] + f[..., 9217:9218]
    mx.eval(y)
print(f"per layer: three q/k/v GEMVs {med(three)/n:.1f} us  vs fused qkv GEMV {med(fused)/n:.1f} us")
# streams: can two independent GEMVs overlap?
s1, s2 = mx.new_stream(mx.default_device()), mx.new_stream(mx.default_device())
def two_serial():
    y = x
    for _ in range(n):
        a = gemv(y, qw, sc, bs); b = gemv(y, qw, sc, bs); y = (a + b)[..., :K]
    mx.eval(y)
def two_streams():
    y = x
    for _ in range(n):
        a = mx.quantized_matmul(y, qw, sc, bs, transpose=True, group_size=64, bits=4, stream=s1)
        b = mx.quantized_matmul(y, qw, sc, bs, transpose=True, group_size=64, bits=4, stream=s2)
        y = (a + b)[..., :K]
    mx.eval(y)
print(f"two independent GEMVs per layer: one stream {med(two_serial)/n:.1f} us  vs two streams {med(two_streams)/n:.1f} us")
