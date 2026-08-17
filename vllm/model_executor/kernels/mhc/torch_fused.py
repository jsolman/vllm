# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pure-PyTorch MHC kernels for SM110 (Tegra).

The tilelang MHC kernels rely on PDL (programmatic dependent launch)
semantics that deadlock or corrupt on Tegra SM110 unified-memory systems
(see debugging notes 2026-08-16/17). These torch composites implement the
exact same math (validated against mhc_pre_torch/mhc_post_torch references
and the tilelang kernel source) using cuBLAS + eager ops only.

Enabled with DSV4_MHC_TORCH=1; selected by vllm/models/deepseek_v4/
nvidia/model.py at import time.
"""

from __future__ import annotations

import torch


def _sinkhorn(comb_logits: torch.Tensor, eps: float, repeat: int) -> torch.Tensor:
    # softmax over last dim with numerical stability
    cm = torch.softmax(comb_logits, dim=-1) + eps
    cm = cm / (cm.sum(dim=-2, keepdim=True) + eps)
    for _ in range(repeat - 1):
        cm = cm / (cm.sum(dim=-1, keepdim=True) + eps)
        cm = cm / (cm.sum(dim=-2, keepdim=True) + eps)
    return cm


def _mixes_from_gemm(
    gemm_out_mul: torch.Tensor,
    gemm_out_sqrsum: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_mult3: int,
) -> torch.Tensor:
    # sum split-k partials, then scale by rsqrt(mean square + eps)
    mixes = gemm_out_mul.sum(dim=0)  # (T, hc_mult3)
    rms = torch.rsqrt(gemm_out_sqrsum.sum(dim=0) / gemm_out_mul.shape[-1] + rms_eps)
    # NOTE: rms normalizes per token; multiply mixes by rms (broadcast over hc_mult3)
    return mixes * rms.unsqueeze(-1)


def mhc_pre_broadcast_torch(
    residual: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    n_splits: int = 1,
    norm_weight: torch.Tensor | None = None,
    norm_eps: float = 1e-6,
    fn_broadcast: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """First-layer mHC pre for a residual broadcast from (T, H).

    Mirrors mhc_pre_broadcast_tilelang: broadcast residual (T,H) ->
    (T, hc_mult, H), mixes from gemm with fn_broadcast (H dim), sinkhorn
    comb mix, fused RMSNorm on the pre-mix layer input.
    """
    assert residual.dtype == torch.bfloat16
    assert residual.dim() == 2
    assert fn_broadcast is not None and norm_weight is not None
    T, H = residual.shape
    hc_mult = fn.shape[1] // H
    hc_mult2 = hc_mult * hc_mult
    hc_mult3 = hc_mult * 2 + hc_mult2

    residual_out = residual.unsqueeze(1).expand(T, hc_mult, H).contiguous()

    x_float = residual.float()  # (T, H)
    # NOTE: broadcast variant uses fn_broadcast (hc_mult3, H); sqrsum over H
    gemm_mul = x_float @ fn_broadcast.t()  # (T, hc_mult3)
    gemm_sqr = x_float.square().sum(dim=-1)  # (T,)

    # rms over the ORIGINAL hidden dim (matches tilelang: rsqrt(sqrsum/H + eps))
    rms = torch.rsqrt(gemm_sqr / H + rms_eps)  # (T,)
    mixes = gemm_mul * rms.unsqueeze(-1)

    pre_logits = mixes[:, :hc_mult] * hc_scale[0] + hc_base[:hc_mult]
    pre_mix = torch.sigmoid(pre_logits) + hc_pre_eps

    post_logits = (
        mixes[:, hc_mult : 2 * hc_mult] * hc_scale[1] + hc_base[hc_mult : 2 * hc_mult]
    )
    post_mix = torch.sigmoid(post_logits) * hc_post_mult_value

    comb_logits = (
        mixes[:, 2 * hc_mult :]
        .view(T, hc_mult, hc_mult)
        * hc_scale[2]
        + hc_base[2 * hc_mult :].view(1, hc_mult, hc_mult)
    )
    comb_mix = _sinkhorn(comb_logits, hc_sinkhorn_eps, sinkhorn_repeat)

    # pre-mix layer input + fused RMSNorm (norm over H of the bf16 mixture)
    layer_input = torch.sum(
        pre_mix.unsqueeze(-1) * residual_out.float(), dim=1
    ).to(torch.bfloat16)
    _f = layer_input.float()
    layer_input = (
        _f * torch.rsqrt(_f.square().mean(dim=-1, keepdim=True) + norm_eps)
    ).to(torch.bfloat16) * norm_weight

    return (
        residual_out,
        post_mix.unsqueeze(-1),
        comb_mix,
        layer_input,
    )


def mhc_post_torch(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_layer_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
) -> torch.Tensor:
    """Post mapping: new_residual = comb @ residual + post_mix * x."""
    mixed = torch.einsum(
        "...ij,...ih->...jh", comb_res_mix.float(), residual.float()
    )
    post_term = post_layer_mix.float() * x.unsqueeze(-2).float()
    return (mixed + post_term).to(residual.dtype)


def mhc_fused_post_pre_torch(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_layer_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    n_splits: int = 1,
    tile_n: int = 1,
    norm_weight: torch.Tensor | None = None,
    norm_eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Post-mapping + pre block; mirrors mhc_fused_post_pre_tilelang.

    Returns (residual_cur, post_mix_cur, comb_mix_cur, layer_input_cur).
    """
    assert residual.dtype == torch.bfloat16
    assert x.dtype == torch.bfloat16
    hc_mult = residual.shape[-2]
    H = residual.shape[-1]
    hc_mult2 = hc_mult * hc_mult
    hc_mult3 = hc_mult * 2 + hc_mult2
    outer = residual.shape[:-2]

    rf = residual.view(-1, hc_mult, H)
    xf = x.view(-1, H)
    pf = post_layer_mix.reshape(-1, hc_mult).float()
    cf = comb_res_mix.reshape(-1, hc_mult, hc_mult).float()
    T = rf.shape[0]

    # post mapping
    residual_cur = torch.einsum("tij,tih->tjh", cf, rf.float()) + pf.unsqueeze(-1) * xf.float().unsqueeze(1)

    # pre: gemm with fn over flattened (hc_mult*H)
    x2 = residual_cur.reshape(T, hc_mult * H)
    mixes_raw = x2.float() @ fn.t()
    sqrsum = x2.float().square().sum(dim=-1)
    rms = torch.rsqrt(sqrsum / (hc_mult * H) + rms_eps)
    mixes = mixes_raw * rms.unsqueeze(-1)

    pre_logits = mixes[:, :hc_mult] * hc_scale[0] + hc_base[:hc_mult]
    pre_mix = torch.sigmoid(pre_logits) + hc_pre_eps

    post_logits = (
        mixes[:, hc_mult : 2 * hc_mult] * hc_scale[1] + hc_base[hc_mult : 2 * hc_mult]
    )
    post_mix_cur = torch.sigmoid(post_logits) * hc_post_mult_value

    comb_logits = (
        mixes[:, 2 * hc_mult :]
        .view(T, hc_mult, hc_mult)
        * hc_scale[2]
        + hc_base[2 * hc_mult :].view(1, hc_mult, hc_mult)
    )
    comb_mix_cur = _sinkhorn(comb_logits, hc_sinkhorn_eps, sinkhorn_repeat)

    layer_input = torch.sum(pre_mix.unsqueeze(-1) * residual_cur, dim=1).to(
        torch.bfloat16
    )
    if norm_weight is not None:
        _f = layer_input.float()
        layer_input = (
            _f
            * torch.rsqrt(_f.square().mean(dim=-1, keepdim=True) + norm_eps)
        ).to(torch.bfloat16) * norm_weight

    return (
        residual_cur.view(*outer, hc_mult, H).to(torch.bfloat16),
        post_mix_cur.view(*outer, hc_mult, 1),
        comb_mix_cur.view(*outer, hc_mult, hc_mult),
        layer_input.view(*outer, H),
    )


def mhc_pre_with_norm_torch(
    residual: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    n_splits: int = 1,
    norm_weight: torch.Tensor | None = None,
    norm_eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """mhc_pre_torch + optional fused RMSNorm on layer_input."""
    from vllm.model_executor.kernels.mhc.torch import mhc_pre_torch

    post_mix, comb_mix, layer_input = mhc_pre_torch(
        residual,
        fn,
        hc_scale,
        hc_base,
        rms_eps,
        hc_pre_eps,
        hc_sinkhorn_eps,
        hc_post_mult_value,
        sinkhorn_repeat,
        n_splits,
    )
    if norm_weight is not None:
        _f = layer_input.float()
        layer_input = (
            _f * torch.rsqrt(_f.square().mean(dim=-1, keepdim=True) + norm_eps)
        ).to(torch.bfloat16) * norm_weight
    return post_mix, comb_mix, layer_input
