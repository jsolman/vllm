# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bit-level A/B of the two MXFP8 kernels wo_a can run on SM110.

DSv4.1's attention output projection (wo_a) is a grouped MXFP8 linear.
Which MXFP8 kernel runs it is decided by ``wo_a.is_bmm``:

  * is_bmm=True  -> the grouped-BMM kernel list -> EmulationMxfp8LinearKernel
  * is_bmm=False -> the ordinary list -> FlashInferCutlassMxfp8LinearKernel

The backport forces is_bmm=True on SM8x/SM110; rebase4 (before
4e894dbb8b6) inherited the ROCm parent's is_bmm=False and so ran the
FlashInfer-CUTLASS kernel instead. The two kernels are numerically
different implementations of the same math (bf16 dequant-and-linear
vs a CUTLASS MXFP8 GEMM with swizzled scales), so this test pins
whether they agree.

Run: .venv/bin/python -m pytest tests/kernels/test_wo_a_kernel_equivalence.py -v -s
"""

import pytest
import torch

from vllm.model_executor.kernels.linear import (
    EmulationMxfp8LinearKernel,
    FlashInferCutlassMxfp8LinearKernel,
)
from vllm.model_executor.kernels.linear.mxfp8 import Mxfp8LinearLayerConfig
from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
    MXFP8_BLOCK_SIZE,
    dequant_mxfp8_to_bf16,
    mxfp8_e4m3_quantize,
)
from vllm.platforms import current_platform

# DSv4.1 geometry: o_groups=8, head_dim=512, o_lora_rank=1024, 64 heads.
# wo_a is ColumnParallelLinear(n_heads*head_dim//n_groups, n_groups*o_lora_rank)
# = (64*512//8, 8*1024) = (4096, 8192); at TP=4 each rank holds a
# (4096, 2048) shard. The kernel sees the LOCAL shard.
N_GROUPS = 8
HEAD_DIM = 512
O_LORA_RANK = 1024
N_HEADS = 64
TP = 4

IN_FEATURES = N_HEADS * HEAD_DIM // N_GROUPS  # 4096
OUT_FEATURES = N_GROUPS * O_LORA_RANK  # 8192
LOCAL_OUT = OUT_FEATURES // TP  # 2048
LOCAL_GROUPS = N_GROUPS // TP  # 2

requires_cuda = pytest.mark.skipif(
    not current_platform.is_cuda(), reason="CUDA-only kernels"
)


def _make_mxfp8_layer(n_out: int, k: int, seed: int = 0):
    """Build a layer holding a valid MXFP8 weight + row-major block scales.

    Uses the codebase's own quantizer so the layout is exactly what the
    ModelOpt MXFP8 loader produces for DSv4.1's wo_a.
    """
    g = torch.Generator(device="cuda").manual_seed(seed)
    w = torch.randn((n_out, k), generator=g, device="cuda", dtype=torch.bfloat16)
    q, s = mxfp8_e4m3_quantize(w, is_sf_swizzled_layout=False)

    class _Layer(torch.nn.Module):
        pass

    layer = _Layer()
    layer.weight = torch.nn.Parameter(q, requires_grad=False)
    layer.weight_scale = torch.nn.Parameter(s, requires_grad=False)
    return layer


def _clone_layer(layer):
    class _Layer(torch.nn.Module):
        pass

    out = _Layer()
    out.weight = torch.nn.Parameter(layer.weight.data.clone(), requires_grad=False)
    out.weight_scale = torch.nn.Parameter(
        layer.weight_scale.data.clone(), requires_grad=False
    )
    return out


@requires_cuda
@pytest.mark.parametrize("m_tokens", [1, 8, 64])
def test_wo_a_emulation_vs_flashinfer_cutlass(m_tokens: int):
    """The two kernels must agree on wo_a's output for the same input."""
    if not FlashInferCutlassMxfp8LinearKernel.is_supported()[0]:
        pytest.skip("FlashInfer CUTLASS MXFP8 not supported on this arch")

    k, n = IN_FEATURES, LOCAL_OUT
    cfg = Mxfp8LinearLayerConfig(weight_shape=(n, k))

    emu = EmulationMxfp8LinearKernel(cfg)
    cut = FlashInferCutlassMxfp8LinearKernel(cfg)

    layer = _make_mxfp8_layer(n, k, seed=1234)
    emu_layer = _clone_layer(layer)
    cut_layer = _clone_layer(layer)

    emu.process_weights_after_loading(emu_layer)
    cut.process_weights_after_loading(cut_layer)

    g = torch.Generator(device="cuda").manual_seed(99)
    x = torch.randn((m_tokens, k), generator=g, device="cuda", dtype=torch.bfloat16)

    out_emu = emu.apply_weights(emu_layer, x)
    out_cut = cut.apply_weights(cut_layer, x)

    assert out_emu.shape == out_cut.shape, (out_emu.shape, out_cut.shape)
    assert out_emu.dtype == out_cut.dtype, (out_emu.dtype, out_cut.dtype)
    assert torch.isfinite(out_emu).all(), "emulation kernel produced non-finite"
    assert torch.isfinite(out_cut).all(), "cutlass kernel produced non-finite"

    exact = torch.equal(out_emu, out_cut)
    diff = (out_emu.float() - out_cut.float()).abs()
    max_abs = float(diff.max())
    denom = float(out_cut.float().abs().max().clamp(min=1e-6))
    rel = max_abs / denom

    print(
        f"\nm_tokens={m_tokens} shape={tuple(out_emu.shape)} "
        f"bit_exact={exact} max_abs_diff={max_abs:.6g} rel={rel:.3e} "
        f"emu[0,:3]={out_emu[0, :3].tolist()} "
        f"cut[0,:3]={out_cut[0, :3].tolist()}"
    )

    assert rel < 1e-2, (
        f"wo_a kernels disagree: max_abs_diff={max_abs:.6g} rel={rel:.3e}"
    )


@requires_cuda
def test_wo_a_emulation_matches_bf16_reference():
    """The emulation kernel must match a plain bf16 dequant-and-linear.

    Pins the emulation kernel itself (the one the backport's is_bmm=True
    path selects on Thor) against an independent reference.
    """
    k, n = IN_FEATURES, LOCAL_OUT
    cfg = Mxfp8LinearLayerConfig(weight_shape=(n, k))
    emu = EmulationMxfp8LinearKernel(cfg)

    layer = _make_mxfp8_layer(n, k, seed=7)
    g = torch.Generator(device="cuda").manual_seed(5)
    x = torch.randn((4, k), generator=g, device="cuda", dtype=torch.bfloat16)

    ref_w = dequant_mxfp8_to_bf16(layer.weight, layer.weight_scale).to(x.dtype)
    ref = torch.nn.functional.linear(x, ref_w)

    emu_layer = _clone_layer(layer)
    emu.process_weights_after_loading(emu_layer)
    out = emu.apply_weights(emu_layer, x)

    diff = (out.float() - ref.float()).abs()
    max_abs = float(diff.max())
    denom = float(ref.float().abs().max().clamp(min=1e-6))
    print(
        f"\nemulation vs bf16 reference: max_abs_diff={max_abs:.6g} "
        f"rel={max_abs / denom:.3e} exact={torch.equal(out, ref)}"
    )
    assert torch.equal(out, ref), (
        f"emulation kernel != bf16 dequant reference (max_abs={max_abs:.6g})"
    )
