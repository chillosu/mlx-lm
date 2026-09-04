# Copyright © 2026 Apple Inc.

"""Fused MoE router kernels (Metal).

At decode shapes the router of an MoE block is a chain of 5-9 tiny kernels
(softmax / sigmoid, bias add, argpartition, take_along_axis, sum, divide) each
paying a few microseconds of fixed launch cost. The kernels below do the whole
post-matmul router (scores, top-k selection, weight normalization) in ONE
kernel, one threadgroup per token, and are written to be bit-identical to the
eager chain they replace:

* the score math replays the exact expression, thread layout and reduction
  order of the corresponding MLX kernels (``softmax_single_row`` with
  ``precise=True``: 4 values per thread, per-thread sequential exp-sum,
  ``simd_sum`` across lanes, ``1 / sum`` then ``T(e * inv)``; ``Sigmoid``:
  ``1 / (1 + exp(|x|))``), so the intermediate scores round identically;
* top-k selection reproduces ``argpartition`` (a stable sort on Metal): each
  element's rank is the number of elements that sort after it, ties broken
  by index exactly like the stable sort, and the k selected indices are
  emitted in the same order the eager slice yields them;
* the normalizer is the sequential sum MLX's small row reduction uses,
  accumulated in the eager dtype (bf16 scores: bf16 accumulation), and the
  division is performed in the eager dtype.

``MLX_LM_MOE_ROUTER_KERNEL=0`` disables the kernels (eager chain).
"""

import os
from functools import lru_cache

import mlx.core as mx

_ENABLED = os.environ.get("MLX_LM_MOE_ROUTER_KERNEL", "1") != "0"

_N_READS = 4
_SIMD = 32

_HEADER = """
template <int E>
constexpr int router_threads() {
  // Same threadgroup size as MLX's softmax dispatch: ceil(E / 4) rounded up
  // to a multiple of the simd width.
  return ((E + 3) / 4 + 31) / 32 * 32;
}
"""

# Softmax router (Qwen3-MoE style): probs = softmax(logits) in fp32 rounded to
# T, inds = the k largest probs in ascending order (== argpartition(..)[-k:]),
# weights = probs[inds] (/ sum if NORM).
_SOFTMAX_SRC = """
    constexpr int N_READS = 4;
    constexpr int SIMD_SIZE = 32;
    constexpr int TPR = router_threads<E>();
    const uint row = threadgroup_position_in_grid.x;
    const int lid = thread_position_in_threadgroup.x;
    const uint simd_lane_id = thread_index_in_simdgroup;
    const uint simd_group_id = simdgroup_index_in_threadgroup;

    threadgroup float local_max[SIMD_SIZE];
    threadgroup float local_normalizer[SIMD_SIZE];
    threadgroup float probs[E];
    threadgroup float sel[K];

    const device T* in = logits + row * size_t(E) + lid * N_READS;
    float ld[N_READS];
    if (lid * N_READS + N_READS <= E) {
      for (int i = 0; i < N_READS; i++) {
        ld[i] = float(in[i]);
      }
    } else {
      for (int i = 0; i < N_READS; i++) {
        ld[i] = ((lid * N_READS + i) < E) ? float(in[i]) : Limits<float>::min;
      }
    }
    if (simd_group_id == 0) {
      local_max[simd_lane_id] = Limits<float>::min;
      local_normalizer[simd_lane_id] = 0;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // max (same two-level reduction as softmax_single_row)
    float maxval = Limits<float>::finite_min;
    for (int i = 0; i < N_READS; i++) {
      maxval = (maxval < ld[i]) ? ld[i] : maxval;
    }
    maxval = simd_max(maxval);
    if (simd_lane_id == 0) {
      local_max[simd_group_id] = maxval;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_group_id == 0) {
      maxval = simd_max(local_max[simd_lane_id]);
      if (simd_lane_id == 0) {
        local_max[0] = maxval;
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    maxval = local_max[0];

    // exp and normalizer
    float normalizer = 0;
    for (int i = 0; i < N_READS; i++) {
      float exp_x = metal::fast::exp(ld[i] - maxval);
      ld[i] = exp_x;
      normalizer += exp_x;
    }
    normalizer = simd_sum(normalizer);
    if (simd_lane_id == 0) {
      local_normalizer[simd_group_id] = normalizer;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_group_id == 0) {
      normalizer = simd_sum(local_normalizer[simd_lane_id]);
      if (simd_lane_id == 0) {
        local_normalizer[0] = normalizer;
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    normalizer = 1 / local_normalizer[0];

    // probabilities, rounded to T exactly like the softmax output
    for (int i = 0; i < N_READS; i++) {
      int gi = lid * N_READS + i;
      if (gi < E) {
        probs[gi] = float(T(ld[i] * normalizer));
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // top-k: rank = #elements sorting after this one in a stable ascending
    // sort (larger value, or equal value with larger index). The eager slice
    // argpartition(..)[-k:] is ascending, so rank 0 lands in slot K-1.
    for (int i = 0; i < N_READS; i++) {
      int gi = lid * N_READS + i;
      if (gi < E) {
        float v = probs[gi];
        int rank = 0;
        for (int j = 0; j < E; j++) {
          float vj = probs[j];
          rank += (vj > v) || ((vj == v) && (j > gi));
        }
        if (rank < K) {
          sel[K - 1 - rank] = v;
          inds[row * K + (K - 1 - rank)] = uint32_t(gi);
        }
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (lid == 0) {
      if (NORM) {
        // The eager `scores / scores.sum()`: MLX's row reduce accumulates a
        // T-typed sum sequentially IN T (bf16 rounds after every add), then
        // a T / T divide.
        T ts = T(0);
        for (int r = 0; r < K; r++) {
          ts = T(sel[r]) + ts;
        }
        for (int r = 0; r < K; r++) {
          weights[row * K + r] = T(sel[r]) / ts;
        }
      } else {
        for (int r = 0; r < K; r++) {
          weights[row * K + r] = T(sel[r]);
        }
      }
    }
"""

# Sigmoid + correction-bias router (DeepSeek-V3 / Laguna style): scores =
# sigmoid(fp32 logits [softcapped]), selection on scores + bias (descending,
# == argpartition(-sel)[:k]), weights = scores[inds] / sum.
_SIGMOID_SRC = """
    constexpr int N_READS = 4;
    const uint row = threadgroup_position_in_grid.x;
    const int lid = thread_position_in_threadgroup.x;

    threadgroup float scores[E];
    threadgroup float sels[E];
    threadgroup float sel[K];

    const device T* in = logits + row * size_t(E) + lid * N_READS;
    for (int i = 0; i < N_READS; i++) {
      int gi = lid * N_READS + i;
      if (gi < E) {
        float x = float(in[i]);
        if (SOFTCAP) {
          float c = softcap[0];
          x = metal::precise::tanh(x / c) * c;
        }
        // MLX's Sigmoid op. precise::exp is what mx.sigmoid evaluates to
        // (verified bit-exact over 2e5 inputs); plain metal::exp here is the
        // fast exp and differs by an ulp.
        float y = 1 / (1 + metal::precise::exp(metal::abs(x)));
        float sc = (x < 0) ? y : 1 - y;
        scores[gi] = sc;
        sels[gi] = sc + bias[gi];
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // rank = #elements sorting before this one in a stable ascending sort of
    // -sel (larger sel, or equal sel with smaller index); slot = rank.
    for (int i = 0; i < N_READS; i++) {
      int gi = lid * N_READS + i;
      if (gi < E) {
        float v = sels[gi];
        int rank = 0;
        for (int j = 0; j < E; j++) {
          float vj = sels[j];
          rank += (vj > v) || ((vj == v) && (j < gi));
        }
        if (rank < K) {
          sel[rank] = scores[gi];
          inds[row * K + rank] = uint32_t(gi);
        }
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (lid == 0) {
      float s = 0;
      for (int r = 0; r < K; r++) {
        s = sel[r] + s;
      }
      for (int r = 0; r < K; r++) {
        weights[row * K + r] = sel[r] / s;
      }
    }
"""


def enabled():
    return _ENABLED and mx.metal.is_available()


def _threads(num_experts):
    return ((num_experts + _N_READS - 1) // _N_READS + _SIMD - 1) // _SIMD * _SIMD


@lru_cache(maxsize=None)
def _softmax_kernel():
    return mx.fast.metal_kernel(
        name="moe_router_softmax_topk",
        input_names=["logits"],
        output_names=["inds", "weights"],
        header=_HEADER,
        source=_SOFTMAX_SRC,
    )


@lru_cache(maxsize=None)
def _sigmoid_kernel():
    return mx.fast.metal_kernel(
        name="moe_router_sigmoid_topk",
        input_names=["logits", "bias", "softcap"],
        output_names=["inds", "weights"],
        header=_HEADER,
        source=_SIGMOID_SRC,
    )


def softmax_topk(logits, k, norm):
    """Fused ``softmax(logits, precise=True)`` -> ``argpartition(kth=-k)[-k:]``
    -> ``take_along_axis`` -> (``/ sum``). Returns ``(inds, weights)`` with
    ``inds`` uint32 ``(..., k)`` and ``weights`` in ``logits.dtype``."""
    *batch, E = logits.shape
    rows = 1
    for b in batch:
        rows *= b
    tpr = _threads(E)
    inds, weights = _softmax_kernel()(
        inputs=[logits],
        template=[("T", logits.dtype), ("E", E), ("K", k), ("NORM", bool(norm))],
        grid=(rows * tpr, 1, 1),
        threadgroup=(tpr, 1, 1),
        output_shapes=[(*batch, k), (*batch, k)],
        output_dtypes=[mx.uint32, logits.dtype],
    )
    return inds, weights


def sigmoid_topk(logits, bias, k, softcap=0.0):
    """Fused ``sigmoid(logits.astype(f32))`` (after optional tanh softcap),
    ``argpartition(-(scores + bias), kth=k-1)[:k]``, ``take_along_axis``,
    ``/ sum``. Returns ``(inds, weights)``: uint32 ``(..., k)``, float32
    ``(..., k)``."""
    *batch, E = logits.shape
    rows = 1
    for b in batch:
        rows *= b
    tpr = _threads(E)
    inds, weights = _sigmoid_kernel()(
        inputs=[logits, bias, mx.array([float(softcap)], dtype=mx.float32)],
        template=[("T", logits.dtype), ("E", E), ("K", k), ("SOFTCAP", softcap > 0.0)],
        grid=(rows * tpr, 1, 1),
        threadgroup=(tpr, 1, 1),
        output_shapes=[(*batch, k), (*batch, k)],
        output_dtypes=[mx.uint32, mx.float32],
    )
    return inds, weights
