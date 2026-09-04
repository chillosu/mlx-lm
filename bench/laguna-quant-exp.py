import sys, time, statistics, collections
import mlx.core as mx, mlx.nn as nn
from mlx_lm import load, generate
from mlx_lm.models.cache import make_prompt_cache
from mlx.utils import tree_flatten
path = sys.argv[1]; model, tok = load(path)
# inventory: bytes by module class
by = collections.Counter(); cnt = collections.Counter()
for name, mod in model.named_modules():
    if isinstance(mod, (nn.Linear, nn.QuantizedLinear)) or type(mod).__name__ in ("SwitchLinear", "QuantizedSwitchLinear"):
        n = sum(v.size * v.dtype.size for k, v in tree_flatten(mod.parameters()))
        key = type(mod).__name__ + (":attn" if "self_attn" in name else ":moe" if "switch_mlp" in name else ":shared" if "shared" in name else ":other")
        by[key] += n; cnt[key] += 1
tot = sum(by.values())
for k, v in sorted(by.items(), key=lambda x: -x[1]): print(f"  {k:28} {v/2**30:7.2f} GiB  x{cnt[k]}")
print(f"  total {tot/2**30:.1f} GiB")
ids = tok.encode("Explain how a prompt cache works in an LLM server. " * 20)[:256]
def step(n=24):
    c = make_prompt_cache(model); l = model(mx.array(ids)[None], cache=c); mx.eval(l); t = mx.argmax(l[:, -1, :], axis=-1); ts = []
    for i in range(n):
        t0 = time.perf_counter(); l = model(t[:, None], cache=c); t = mx.argmax(l[:, -1, :], axis=-1); mx.eval(t)
        if i >= 4: ts.append(time.perf_counter() - t0)
    return 1000 * statistics.median(ts)
prompt = tok.apply_chat_template([{"role":"user","content":"Name three prime numbers and stop."}], add_generation_prompt=True, tokenize=False)
base = step(); ref = generate(model, tok, prompt=prompt, max_tokens=24, verbose=False)
print(f"baseline step {base:.2f} ms ({1000/base:.1f} tok/s)  sample: {ref[:80]!r}")
# quantize remaining bf16 Linear layers (attention projections etc.) to affine 4-bit, group 64
nn.quantize(model, group_size=64, bits=4, class_predicate=lambda p, m: isinstance(m, nn.Linear) and not isinstance(m, nn.QuantizedLinear) and m.weight.shape[-1] % 64 == 0 and "lm_head" not in p)
mx.eval(model.parameters())
q = step(); out = generate(model, tok, prompt=prompt, max_tokens=24, verbose=False)
print(f"after quantizing bf16 Linears: step {q:.2f} ms ({1000/q:.1f} tok/s)  speedup {base/q:.2f}x  sample: {out[:80]!r}")
