# Copyright © 2023-2024 Apple Inc.

import math
import os
from functools import partial, wraps

import mlx.core as mx
import mlx.nn as nn

from .activations import swiglu


# Kill switches (set to 0 to disable): lazy up/gate fusion in SwitchGLU, and
# the compiled decode path of MoE blocks decorated with ``compiled_decode``.
_FUSE_UP_GATE = os.environ.get("MLX_LM_FUSE_MOE_UP_GATE", "1") != "0"
_COMPILE_DECODE = os.environ.get("MLX_LM_COMPILE_MOE_DECODE", "1") != "0"


def compiled_decode(forward):
    """Decorator for an MoE block method ``forward(self, x, *args)``.

    At decode shapes (``x`` of sequence length 1, eval mode) the method runs
    as one shape-specialized ``mx.compile``d graph, traced once per batch
    size; any other shape (prefill, training) runs the plain Python method.
    At these shapes the block is dominated by per-kernel fixed cost, so
    replaying a pre-built graph from C++ (no op-by-op Python graph
    construction) is measurably faster. The block's parameters are compile
    *inputs*, not closure constants, so later weight updates are honored.

    Only decorate graphs that stay bit-identical under compile: adjacent
    elementwise ops get fused into JIT kernels whose transcendental math can
    differ from the standalone kernels by an ulp (``mx.sigmoid`` does), and a
    discrete top-k router turns an ulp into a different expert choice. Keep
    such routers eager and decorate the expert-mixing part instead.
    """
    key = "_compiled_" + forward.__name__

    @wraps(forward)
    def wrapper(self, x, *args):
        if self.training or x.shape[1] != 1 or not _COMPILE_DECODE:
            return forward(self, x, *args)
        fn = self.__dict__.get(key)
        if fn is None:
            # Finalize the module structure before tracing (the fused weights
            # must exist and be materialized outside the trace).
            if _FUSE_UP_GATE:
                for m in self.modules():
                    if isinstance(m, SwitchGLU):
                        m.fuse_up_gate()
            mx.eval(self.parameters())
            fn = mx.compile(partial(forward, self), inputs=[self])
            setattr(self, key, fn)
        return fn(x, *args)

    return wrapper


def _gather_sort(x, indices):
    *_, M = indices.shape
    indices = indices.flatten()
    order = mx.argsort(indices)
    inv_order = mx.argsort(order)
    return x.flatten(0, -3)[order // M], indices[order], inv_order


def _scatter_unsort(x, inv_order, shape=None):
    x = x[inv_order]
    if shape is not None:
        x = mx.unflatten(x, 0, shape)
    return x


class QuantizedSwitchLinear(nn.Module):
    def __init__(
        self,
        input_dims: int,
        output_dims: int,
        num_experts: int,
        bias: bool = True,
        group_size: int = 64,
        bits: int = 4,
        mode: str = "affine",
    ):
        super().__init__()

        scale = math.sqrt(1 / input_dims)
        self.weight, self.scales, *biases = mx.quantize(
            mx.random.uniform(
                low=-scale,
                high=scale,
                shape=(num_experts, output_dims, input_dims),
            ),
            group_size=group_size,
            bits=bits,
            mode=mode,
        )
        self.biases = biases[0] if biases else None

        if bias:
            self.bias = mx.zeros((num_experts, output_dims))

        self.group_size = group_size
        self.bits = bits
        self.mode = mode

        # Freeze this model's parameters
        self.freeze()

    @classmethod
    def from_arrays(cls, weight, scales, biases, bias, group_size, bits, mode):
        """Build a layer around already-quantized arrays without running the
        (expensive) random init + quantize of ``__init__``."""
        ql = cls.__new__(cls)
        nn.Module.__init__(ql)
        ql.weight, ql.scales = weight, scales
        ql.biases = biases
        if bias is not None:
            ql.bias = bias
        ql.group_size, ql.bits, ql.mode = group_size, bits, mode
        ql.freeze()
        return ql

    @property
    def input_dims(self):
        return self.scales.shape[2] * self.group_size

    @property
    def output_dims(self):
        return self.weight.shape[1]

    @property
    def num_experts(self):
        return self.weight.shape[0]

    def __call__(self, x, indices, sorted_indices=False):
        x = mx.gather_qmm(
            x,
            self["weight"],
            self["scales"],
            self.get("biases"),
            rhs_indices=indices,
            transpose=True,
            group_size=self.group_size,
            bits=self.bits,
            mode=self.mode,
            sorted_indices=sorted_indices,
        )
        if "bias" in self:
            x = x + mx.expand_dims(self["bias"][indices], -2)
        return x


class SwitchLinear(nn.Module):
    def __init__(
        self, input_dims: int, output_dims: int, num_experts: int, bias: bool = True
    ):
        super().__init__()
        scale = math.sqrt(1 / input_dims)
        self.weight = mx.random.uniform(
            low=-scale,
            high=scale,
            shape=(num_experts, output_dims, input_dims),
        )

        if bias:
            self.bias = mx.zeros((num_experts, output_dims))

    @property
    def input_dims(self):
        return self.weight.shape[2]

    @property
    def output_dims(self):
        return self.weight.shape[1]

    @property
    def num_experts(self):
        return self.weight.shape[0]

    def __call__(self, x, indices, sorted_indices=False):
        x = mx.gather_mm(
            x,
            self["weight"].swapaxes(-1, -2),
            rhs_indices=indices,
            sorted_indices=sorted_indices,
        )
        if "bias" in self:
            x = x + mx.expand_dims(self["bias"][indices], -2)
        return x

    def to_quantized(self, group_size: int = 64, bits: int = 4, mode: str = "affine"):
        num_experts, output_dims, input_dims = self.weight.shape
        ql = QuantizedSwitchLinear(
            input_dims,
            output_dims,
            num_experts,
            False,
            group_size,
            bits,
            mode=mode,
        )
        ql.weight, ql.scales, *biases = mx.quantize(
            self.weight, group_size, bits, mode=mode
        )
        ql.biases = biases[0] if biases else None

        if "bias" in self:
            ql.bias = self.bias
        return ql


class SwiGLU(nn.Module):
    def __init__(self):
        super().__init__()

    def __call__(self, x, gate):
        return swiglu(gate, x)


class SwitchGLU(nn.Module):
    def __init__(
        self,
        input_dims: int,
        hidden_dims: int,
        num_experts: int,
        activation=SwiGLU(),
        bias: bool = False,
    ):
        super().__init__()

        self.gate_proj = SwitchLinear(input_dims, hidden_dims, num_experts, bias=bias)
        self.up_proj = SwitchLinear(input_dims, hidden_dims, num_experts, bias=bias)
        self.down_proj = SwitchLinear(hidden_dims, input_dims, num_experts, bias=bias)
        self.activation = activation

    def fuse_up_gate(self):
        """Concatenate ``up_proj`` and ``gate_proj`` along the output dim into
        one ``up_gate_proj`` so a forward pass runs one gather-matmul instead
        of two (same bytes read, one kernel launch fewer; bit-identical
        output). Idempotent. A no-op when the two projections are not plain
        (Quantized)SwitchLinear of the same type (e.g. wrapped in LoRA)."""
        if "up_gate_proj" in self or "up_proj" not in self:
            return
        up, gate = self.up_proj, self.gate_proj
        if type(up) is not type(gate) or type(up) not in (
            SwitchLinear,
            QuantizedSwitchLinear,
        ):
            return
        bias = (
            mx.concatenate([up.bias, gate.bias], axis=-1) if "bias" in up else None
        )
        if isinstance(up, QuantizedSwitchLinear):
            fused = QuantizedSwitchLinear.from_arrays(
                mx.concatenate([up.weight, gate.weight], axis=1),
                mx.concatenate([up.scales, gate.scales], axis=1),
                (
                    mx.concatenate([up.biases, gate.biases], axis=1)
                    if up.get("biases") is not None
                    else None
                ),
                bias,
                up.group_size,
                up.bits,
                up.mode,
            )
        else:
            fused = SwitchLinear.__new__(SwitchLinear)
            nn.Module.__init__(fused)
            fused.weight = mx.concatenate([up.weight, gate.weight], axis=1)
            if bias is not None:
                fused.bias = bias
        self.up_gate_proj = fused
        del self.up_proj
        del self.gate_proj

    def __call__(self, x, indices) -> mx.array:
        if _FUSE_UP_GATE and not self.training and "up_gate_proj" not in self:
            self.fuse_up_gate()
        x = mx.expand_dims(x, (-2, -3))

        # When we have many tokens, then sort them to make sure that the access
        # of different experts is in order.
        do_sort = indices.size >= 64
        idx = indices
        inv_order = None
        if do_sort:
            x, idx, inv_order = _gather_sort(x, indices)
        if self.training:
            idx = mx.stop_gradient(idx)
        if "up_gate_proj" in self:
            x_up, x_gate = mx.split(
                self.up_gate_proj(x, idx, sorted_indices=do_sort), 2, axis=-1
            )
        else:
            x_up = self.up_proj(x, idx, sorted_indices=do_sort)
            x_gate = self.gate_proj(x, idx, sorted_indices=do_sort)
        x = self.down_proj(
            self.activation(x_up, x_gate),
            idx,
            sorted_indices=do_sort,
        )

        if do_sort:
            x = _scatter_unsort(x, inv_order, indices.shape)

        return x.squeeze(-2)


class SwitchMLP(nn.Module):
    def __init__(
        self,
        input_dims: int,
        hidden_dims: int,
        num_experts: int,
        activation=nn.GELU(approx="precise"),
        bias: bool = False,
    ):
        super().__init__()

        self.fc1 = SwitchLinear(input_dims, hidden_dims, num_experts, bias=bias)
        self.fc2 = SwitchLinear(hidden_dims, input_dims, num_experts, bias=bias)
        self.activation = activation

    def __call__(self, x, indices) -> mx.array:
        x = mx.expand_dims(x, (-2, -3))

        # When we have many tokens, then sort them to make sure that the access
        # of different experts is in order.
        do_sort = indices.size >= 64
        idx = indices
        inv_order = None
        if do_sort:
            x, idx, inv_order = _gather_sort(x, indices)
        if self.training:
            idx = mx.stop_gradient(idx)
        x = self.fc1(x, idx, sorted_indices=do_sort)
        x = self.activation(x)
        x = self.fc2(x, idx, sorted_indices=do_sort)

        if do_sort:
            x = _scatter_unsort(x, inv_order, indices.shape)

        return x.squeeze(-2)
