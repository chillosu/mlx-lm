"""Re-save Laguna NVFP4 with its bf16 Linear layers (attention q/k/v/o/g, shared experts, router-adjacent) quantized to affine 4-bit g64.
Keeps the NVFP4 experts as-is; writes per-path quantization overrides so mlx-lm loads the mixed model."""
import sys, json, shutil, glob, os
import mlx.core as mx, mlx.nn as nn
from mlx_lm import load
from mlx_lm.utils import save_model
src, dst = sys.argv[1], sys.argv[2]
model, tok = load(src)
cfg = json.load(open(os.path.join(src, "config.json")))
q = cfg.get("quantization") or {}
newly = []
def pred(p, m):
    ok = isinstance(m, nn.Linear) and not isinstance(m, nn.QuantizedLinear) and m.weight.shape[-1] % 64 == 0 and "lm_head" not in p
    if ok: newly.append(p)
    return ok
nn.quantize(model, group_size=64, bits=4, class_predicate=pred)
mx.eval(model.parameters())
for p in newly: q[p] = {"group_size": 64, "bits": 4, "mode": "affine"}
cfg["quantization"] = q; cfg["quantization_config"] = q
os.makedirs(dst, exist_ok=True)
save_model(dst, model, donate_model=True)
json.dump(cfg, open(os.path.join(dst, "config.json"), "w"), indent=2)
for f in glob.glob(os.path.join(src, "*")):
    b = os.path.basename(f)
    if b.endswith(".safetensors") or b in ("config.json", "model.safetensors.index.json"): continue
    if os.path.isfile(f): shutil.copy(f, dst)
print(f"quantized {len(newly)} Linear layers; saved to {dst}")
