# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""TileLang sparse attention for DeepSeek V4.1 on SM110 (Thor).

WHY THIS EXISTS
---------------
We originally chose the Triton sparse-MLA path for SM110 because we
believed the reference implementation's TileLang kernels would not
compile there. That was never tested. They do: all six reference kernels
compile and run natively with ``-arch=sm_110a`` (the only fix needed is
CUDA_HOME pointing at a CUDA whose nvcc matches its headers).

This backend runs the REFERENCE's own ``sparse_attn`` TileLang kernel -
the exact attention arithmetic DeepSeek ships - so we can A/B the
attention kernel against the Triton one with everything else in the
serving stack held constant.

THE INTERFACE GAP
-----------------
The reference kernel wants a DENSE bf16 KV tensor plus per-query top-k
indices. vLLM's DSv4.1 path keeps a PAGED fp8 cache in the 584-byte
record layout (448 B fp8 NoPE | 64 B bf16 RoPE raw, plus segregated
UE8M0 scales) and addresses it with ragged slot lists.

So this module:
  1. reads the packed paged rows and DEQUANTISES them into the dense
     bf16 layout the reference kernel expects,
  2. calls the reference's sparse_attn,
  3. writes the result back.

For DECODE the gather is cheap: topk x head_dim x 2 bytes per query
(512 x 512 x 2 = 512 KB at topk=512). This backend is therefore a
DIAGNOSTIC first and a performance path second.

Only the SWA (main) cache is handled here: the compressed (extra) cache
would need the same treatment with its own record width, and the
compressed path is a separate concern from the attention arithmetic.

TILELANG AVAILABILITY
---------------------
TileLang is an optional dependency. If it is missing (or cannot compile
for this arch) this module still imports, and the backend reports itself
unavailable so the caller can fall back.
"""

from __future__ import annotations

import functools
import os
from typing import Any

import torch

from vllm.logger import init_logger
from vllm.platforms.interface import DeviceCapability

logger = init_logger(__name__)

# The DSv4.1 SWA cache record: 448 B fp8 NoPE + 64 B bf16 RoPE (raw),
# with the scales segregated after the values within a page.
NOPE_DIM = 448
ROPE_DIM = 64
ROW_BYTES = NOPE_DIM + ROPE_DIM * 2      # 576 value bytes per token
SCALE_BYTES = 8                          # 8 UE8M0 scale bytes per token
# 7 real scales (one per 64-wide NoPE tile) + 1 zero padding byte.
NUM_SCALE_TILES = NOPE_DIM // 64


@functools.lru_cache(maxsize=1)
def _tilelang_available() -> bool:
    try:
        import tilelang  # noqa: F401
    except Exception:
        logger.warning(
            "TileLang is not installed: the DSv4.1 TileLang sparse attention "
            "backend is unavailable.")
        return False
    return True


@functools.lru_cache(maxsize=1)
def _reference_kernel_module():
    """Import the checkpoint's reference kernel module.

    The reference ships in the model directory as ``inference/kernel.py``.
    We load it by path so we do not need the model on disk at import time
    - the path is resolved when the backend is first used.
    """
    import importlib.util

    model_path = os.environ.get("VLLM_DSV41_REFERENCE_MODEL", "")
    if not model_path:
        logger.warning(
            "VLLM_DSV41_REFERENCE_MODEL is not set: the DSv4.1 TileLang "
            "backend cannot find the reference kernel module.")
        return None
    kernel_py = os.path.join(model_path, "inference", "kernel.py")
    if not os.path.exists(kernel_py):
        logger.warning("reference kernel module not found at %s", kernel_py)
        return None
    spec = importlib.util.spec_from_file_location("dsv41_reference_kernel",
                                                  kernel_py)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@functools.lru_cache(maxsize=8)
def _compiled_sparse_attn(num_heads: int, head_dim: int, scale: float):
    """Compile (and cache) the reference sparse_attn kernel."""
    mod = _reference_kernel_module()
    if mod is None:
        return None
    return mod.sparse_attn_kernel(num_heads, head_dim, scale)


def dequantise_swa_pages(
    swa_k_cache: torch.Tensor,
    slots: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Read packed fp8 SWA rows and dequantise them into dense bf16.

    Args:
        swa_k_cache: the packed paged cache, ``[blocks, block, bytes]``.
        slots: the flat slot ids to read, one per (query, slot).

    Returns:
        ``(rows, valid)`` where ``rows`` is ``[n_slots, head_dim]`` bf16
        (the dequantised K) and ``valid`` is ``[n_slots]`` bool.
    """
    blocks, block, _ = swa_k_cache.shape
    # The scale area starts after the values of the whole page.
    scale_base = block * ROW_BYTES

    # The kernel's own pointer arithmetic (ground truth):
    #   cache_block_ptr = main_cache_ptr + block_idx * main_cache_stride0
    #   token_data_ptr  = cache_block_ptr + pos_in_block * 576
    #   token_scale_ptr = cache_block_ptr + main_block_size * 576 + pos * 8
    # so a page holds [block * 576 value bytes][block * 8 scale bytes] and
    # the page stride is the tensor's own stride(0).
    page_stride = swa_k_cache.stride(0)
    flat = swa_k_cache.reshape(-1)

    block_ids = torch.div(slots, block, rounding_mode="floor").long()
    offsets = (slots % block).long()
    valid = (block_ids < blocks) & (slots >= 0)

    # Gather the value bytes and the scale bytes.
    value_off = block_ids * page_stride + offsets * ROW_BYTES
    vals = flat[value_off[:, None] + torch.arange(ROW_BYTES, device=flat.device)]
    scale_off = (block_ids * page_stride + scale_base + offsets * SCALE_BYTES)
    scales = flat[scale_off[:, None]
                  + torch.arange(SCALE_BYTES, device=flat.device)]

    vals = vals.reshape(-1, ROW_BYTES)
    scales = scales.reshape(-1, SCALE_BYTES)

    # 448 NoPE dims: fp8 e4m3 with one UE8M0 scale per 64-wide tile.
    nope_u8 = vals[:, :NOPE_DIM].contiguous().view(torch.float8_e4m3fn)
    nope = nope_u8.to(torch.float32)
    enc = scales[:, :NUM_SCALE_TILES].to(torch.float32)
    decoded = torch.pow(2.0, enc - 127.0)          # exp2(enc - 127)
    tile = decoded.repeat_interleave(64, dim=1)     # [n, 448]
    nope = nope * tile

    # 64 RoPE dims: stored raw bf16.
    rope = vals[:, NOPE_DIM:].contiguous().view(torch.bfloat16).to(torch.float32)

    rows = torch.cat([nope, rope], dim=1).to(torch.bfloat16)
    rows = torch.where(valid[:, None], rows, torch.zeros_like(rows))
    return rows, valid


def tilelang_sparse_attn_decode(
    q: torch.Tensor,
    swa_k_cache: torch.Tensor,
    swa_slots: torch.Tensor,
    attn_sink: torch.Tensor | None,
    scale: float,
) -> torch.Tensor:
    """Run the reference TileLang sparse_attn over the packed SWA cache.

    Args:
        q: ``[num_queries, num_heads, head_dim]`` bf16.
        swa_k_cache: the packed paged cache ``[blocks, block, bytes]``.
        swa_slots: flat slot ids, ``[num_queries, topk]`` int32.
        attn_sink: per-head sink logits, ``[num_heads]`` fp32, or None.
        scale: the softmax scale.

    Returns:
        ``[num_queries, num_heads, head_dim]`` bf16.
    """
    num_queries, num_heads, head_dim = q.shape
    topk = swa_slots.shape[1]

    # The reference kernel's KV is the CONCATENATED (window, compressed)
    # sequence and its topk_idxs index into that. For the SWA-only case
    # the window is the whole KV, and each query's index list selects its
    # own window slots. So we pass the dequantised window as the dense KV
    # and express each query's slots as positions within that window.
    #
    # Build the dense window from the UNIQUE slots the queries reference,
    # then map each query's slot to its position in that window.
    flat_slots = swa_slots.reshape(-1).long()
    uniq, inv = torch.unique(flat_slots, return_inverse=True)
    dense_rows, dense_valid = dequantise_swa_pages(swa_k_cache, uniq)

    # torch.unique sorts, so a slot that is out of range still appears in
    # `uniq`; dense_valid marks those.
    pos_in_uniq = inv.reshape(num_queries, topk)
    slot_valid = dense_valid[pos_in_uniq]

    # The reference treats an index of -1 as "no valid position": it
    # writes -inf into the scores for that slot.
    topk_idxs = torch.where(slot_valid,
                             pos_in_uniq.to(torch.int32),
                             torch.full((num_queries, topk), -1,
                                        device=q.device, dtype=torch.int32))

    num_heads_padded = 16 if num_heads < 16 else num_heads
    q_p = q
    sink = attn_sink
    if num_heads < 16:
        q_p = torch.cat([q, q.new_zeros(num_queries, 16 - num_heads, head_dim)],
                        dim=1)
        sink = (torch.cat([attn_sink, attn_sink.new_zeros(16 - num_heads)])
                if attn_sink is not None else None)

    kernel = _compiled_sparse_attn(num_heads_padded, head_dim, scale)
    if kernel is None:
        raise RuntimeError("the DSv4.1 reference TileLang kernel is unavailable")

    # The reference kernel wants q as [b, m, h, d] and kv as [b, n, d], and
    # takes the output buffer as an explicit argument (its own wrapper
    # allocates it with torch.empty_like(q)).
    out = torch.empty_like(q_p).unsqueeze(0)
    kernel(q_p.unsqueeze(0), dense_rows.unsqueeze(0), out,
           sink if sink is not None else torch.zeros(num_heads_padded,
                                                     device=q.device,
                                                     dtype=torch.float32),
           topk_idxs.unsqueeze(0))
    out = out[0]
    if num_heads < 16:
        out = out[:, :num_heads].contiguous()
    return out


class DeepseekV41TileLangSparseBackend:
    """Backend descriptor for the DSv4.1 TileLang sparse attention."""

    @staticmethod
    def get_name() -> str:
        return "TILELANG_MLA_SPARSE_DSV41"

    @classmethod
    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:
        return capability.major in [8, 11] and _tilelang_available()
