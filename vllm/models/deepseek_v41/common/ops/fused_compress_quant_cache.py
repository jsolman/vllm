# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""V4.1 state saving/compression and independently schedulable cache insertion."""

import torch

from vllm.models.deepseek_v4.common.ops.fused_indexer_q import _fp32x2_to_fp4x2
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton


def _quant_e2m1_enabled() -> bool:
    """Use the reference's compressed-cache format (e2m1 with one e4m3
    scale per 16 dims) instead of our fp8/UE8M0-per-64 one.

    The reference (`inference/model.py`) quantises the whole 512-dim
    latent to e2m1 with an e4m3 scale per 16 dims and stores the
    DEQUANTISED values. Our format is more accurate, but the model was
    trained on the reference's, so the difference is a real difference
    in the numbers the attention sees (~10% of the attention score's
    spread, with 31 of the 512 selected positions changing). See
    notes/DSV41-LOOPING-ROOTCAUSE.md (2aw).

    Read from the env OR from /tmp/dsv41-flags (one name per line) - the
    engine process does not inherit every env var, so the file is the
    reliable channel for a live A/B.
    """
    import os

    name = "VLLM_DSV41_QUANT_E2M1_CACHE"
    if os.environ.get(name, "0") == "1":
        return True
    try:
        with open("/tmp/dsv41-flags") as f:
            for line in f:
                if line.strip() == name:
                    return True
    except OSError:
        pass
    return False

if current_platform.is_rocm():
    from vllm.platforms.rocm import _ON_GFX950
else:
    _ON_GFX950 = False

# The ring and raw rows a ratio-2 request program pools are not adjacent, so
# reading them as one tile means joining two pointers. ROCm's Triton fails to
# legalize `tt.join` on pointers, so there the two rows are loaded separately.
_JOIN_ROW_PTRS = not current_platform.is_rocm()


def fused_save_compress_norm(
    kv_score: torch.Tensor,
    positions: torch.Tensor,
    state_cache: torch.Tensor | None,
    slot_mapping: torch.Tensor,
    query_start_loc: torch.Tensor | None,
    token_to_req_indices: torch.Tensor | None,
    rms_norm_weight: torch.Tensor,
    rms_norm_eps: float,
    compress_ratio: int,
    latent_out: torch.Tensor,
    prev_rows: torch.Tensor | None = None,
    prev_row_indices: torch.Tensor | None = None,
) -> None:
    """Pool each closed group into a normalized BF16 latent; save FP32 states.

    The latent feeds the main-cache insert and the indexer K path, which the
    attention layer schedules on separate streams.

    Ratio 2 keeps one ring block per request holding the open group's rows:
    position ``p`` lives in row ``p % capacity`` and ``slot_mapping`` encodes
    ``block * capacity + p % capacity``. The grid has one program per request
    followed by one per pair of packed tokens. A request program handles the
    group that the chunk's first token closes with its predecessor's ring row,
    then stores the chunk's last ``capacity`` rows to the ring; because the
    same program does both, ring reads and writes never race. A pair program
    handles the group that ends inside its pair, reading both rows from the
    raw input. Ratio 1 has no ring and one program per token; ``slot_mapping``
    then only marks valid tokens.

    Args:
        kv_score: FP32 [tokens, 512] for CR1, [tokens, 1024] for CR2.
        positions: Absolute positions of the packed request tokens.
        state_cache: Ring FP32 [blocks, capacity, 1024] KV/score states (CR2).
        slot_mapping: Ring slots (CR2) or main-cache slots (CR1).
        query_start_loc: [num_reqs + 1] token offsets of each request's chunk.
        token_to_req_indices: Request indices for the packed token rows.
        rms_norm_weight: BF16 [512] normalization weight.
        rms_norm_eps: RMSNorm epsilon.
        compress_ratio: Group size, either 1 or 2.
        latent_out: BF16 [tokens, 512], written only at valid group boundaries.
        prev_rows: FP32 [rows, 1024] rows gathered from all PCP ranks (CR2). A
            chunk reads its predecessor here where ``prev_row_indices`` is set,
            and the ring is left to ``save_ring_rows``.
        prev_row_indices: [num_reqs] row in ``prev_rows`` holding the token
            before each chunk's first token, or -1 to read the ring.

    """
    assert compress_ratio in (1, 2)
    assert kv_score.dtype == torch.float32
    assert kv_score.shape[1] == 512 * compress_ratio and kv_score.stride(1) == 1
    assert latent_out.shape == (kv_score.shape[0], 512)
    assert latent_out.is_contiguous() and latent_out.dtype == torch.bfloat16
    assert positions.is_contiguous() and slot_mapping.is_contiguous()
    # Rows stay 64-byte aligned, as the kernel's tl.multiple_of hint promises.
    assert kv_score.stride(0) % 16 == 0
    if compress_ratio == 2:
        assert state_cache is not None and query_start_loc is not None
        assert token_to_req_indices is not None
        assert query_start_loc.is_contiguous()
        assert token_to_req_indices.is_contiguous()
        assert state_cache.dtype == torch.float32
        assert state_cache.shape[2] == 1024 and state_cache.stride(2) == 1
        assert state_cache.stride(1) % 16 == 0
        state_stride, state_row_stride, state_block = (
            state_cache.stride(0),
            state_cache.stride(1),
            state_cache.shape[1],
        )
        num_reqs = query_start_loc.numel() - 1
        if prev_rows is not None:
            assert prev_row_indices is not None
            assert prev_row_indices.numel() >= num_reqs
            assert prev_rows.dtype == torch.float32 and prev_rows.shape[1] == 1024
            assert prev_rows.stride(1) == 1 and prev_rows.stride(0) % 16 == 0
    else:
        state_cache = query_start_loc = token_to_req_indices = None
        state_stride = state_row_stride = state_block = 1
        num_reqs = 0
    num_tokens = slot_mapping.numel()
    assert num_tokens <= min(kv_score.shape[0], positions.numel())
    if num_tokens == 0:
        return
    has_prev_rows = prev_rows is not None
    prev_row_stride = prev_rows.stride(0) if prev_rows is not None else 1
    grid = num_reqs + triton.cdiv(num_tokens, compress_ratio)
    _fused_save_compress_norm_kernel[(grid,)](
        kv_score,
        positions,
        state_cache,
        slot_mapping,
        query_start_loc,
        token_to_req_indices,
        rms_norm_weight,
        latent_out,
        prev_rows if has_prev_rows else kv_score,
        prev_row_indices if has_prev_rows else slot_mapping,
        num_tokens,
        num_reqs,
        RAW_STRIDE=kv_score.stride(0),
        PREV_ROW_STRIDE=prev_row_stride,
        STATE_STRIDE=state_stride,
        STATE_ROW_STRIDE=state_row_stride,
        STATE_BLOCK=state_block,
        COMPRESS_RATIO=compress_ratio,
        EPS=rms_norm_eps,
        JOIN_ROW_PTRS=_JOIN_ROW_PTRS,
        HAS_PREV_ROWS=has_prev_rows,
        num_warps=4,
        **({"launch_pdl": False} if current_platform.is_cuda() else {}),
    )


@triton.jit(do_not_specialize=["num_tokens", "num_reqs"])
def _fused_save_compress_norm_kernel(
    raw,
    positions,
    state,
    state_slots,
    query_start_loc,
    req_ids,
    norm_weight,
    latent,
    prev_rows,
    prev_row_indices,
    num_tokens,
    num_reqs,
    RAW_STRIDE: tl.constexpr,
    PREV_ROW_STRIDE: tl.constexpr,
    STATE_STRIDE: tl.constexpr,
    STATE_ROW_STRIDE: tl.constexpr,
    STATE_BLOCK: tl.constexpr,
    COMPRESS_RATIO: tl.constexpr,
    EPS: tl.constexpr,
    JOIN_ROW_PTRS: tl.constexpr,
    HAS_PREV_ROWS: tl.constexpr,
):
    pid = tl.program_id(0)
    d = tl.arange(0, 512)

    if COMPRESS_RATIO == 2:  # noqa: SIM102 (constexpr branch, then runtime)
        if pid < num_reqs:
            # Request program: the group closed by the chunk's first token,
            # then the ring tail store.
            start = tl.load(query_start_loc + pid)
            end = tl.load(query_start_loc + pid + 1)
            if start >= end:
                return
            state_slot = tl.load(state_slots + start)
            if state_slot < 0:
                return
            position = tl.load(positions + start)
            if (position + 1) % 2 == 0:
                ring = state + (state_slot // STATE_BLOCK).to(tl.int64) * STATE_STRIDE
                prev = ring + ((position - 1) % STATE_BLOCK) * STATE_ROW_STRIDE
                if HAS_PREV_ROWS:
                    prev_idx = tl.load(prev_row_indices + pid)
                    if prev_idx >= 0:
                        prev = prev_rows + prev_idx.to(tl.int64) * PREV_ROW_STRIDE
                current = raw + start.to(tl.int64) * RAW_STRIDE
                if JOIN_ROW_PTRS:
                    # Joining pointers hides row alignment from Triton; restate it.
                    rows = tl.multiple_of(tl.join(prev, current)[:, None], (16, 16))
                    kv = tl.load(rows + d[None, :])
                    score = tl.load(rows + 512 + d[None, :])
                    # Every lane has read the ring before the tail store below
                    # writes it.
                    tl.debug_barrier()
                    pooled = tl.sum(kv * tl.softmax(score, 0), 0)
                else:
                    kv_prev = tl.load(prev + d)
                    score_prev = tl.load(prev + 512 + d)
                    kv_current = tl.load(current + d)
                    score_current = tl.load(current + 512 + d)
                    # Every lane has read the ring before the tail store below
                    # writes it.
                    tl.debug_barrier()
                    peak = tl.maximum(score_prev, score_current)
                    weight_prev = tl.exp(score_prev - peak)
                    weight_current = tl.exp(score_current - peak)
                    pooled = (kv_prev * weight_prev + kv_current * weight_current) / (
                        weight_prev + weight_current
                    )
                _store_latent(pooled, start, norm_weight, latent, EPS)
            if not HAS_PREV_ROWS:
                num_rows = tl.minimum(end - start, STATE_BLOCK)
                for k in tl.range(0, num_rows):
                    token = end - num_rows + k
                    slot = tl.load(state_slots + token)
                    if slot >= 0:
                        row = (
                            state
                            + (slot // STATE_BLOCK).to(tl.int64) * STATE_STRIDE
                            + (slot % STATE_BLOCK) * STATE_ROW_STRIDE
                        )
                        src = raw + token.to(tl.int64) * RAW_STRIDE
                        tl.store(row + d, tl.load(src + d))
                        tl.store(row + 512 + d, tl.load(src + 512 + d))
            return

    # Group program: one token (ratio 1) or the token of the pair that ends
    # a group (ratio 2). A chunk's first token belongs to its request program.
    t = (pid - num_reqs) * COMPRESS_RATIO
    if COMPRESS_RATIO == 2:
        t += (tl.load(positions + t) + 1) % 2
    state_slot = tl.load(state_slots + t, t < num_tokens, other=-1)
    if state_slot < 0:
        return
    if COMPRESS_RATIO == 1:
        pooled = tl.load(raw + t.to(tl.int64) * RAW_STRIDE + d)
    else:
        if t == tl.load(query_start_loc + tl.load(req_ids + t)):
            return
        rows = raw + (t.to(tl.int64) - 1 + tl.arange(0, 2)[:, None]) * RAW_STRIDE
        kv = tl.load(rows + d[None, :])
        score = tl.load(rows + 512 + d[None, :])
        pooled = tl.sum(kv * tl.softmax(score, 0), 0)
    _store_latent(pooled, t, norm_weight, latent, EPS)


def save_ring_rows(
    rows: torch.Tensor, slot_mapping: torch.Tensor, state_cache: torch.Tensor
) -> None:
    """Store FP32 [kv, score] rows to their ring slots, skipping slot -1."""
    assert rows.dtype == torch.float32 and rows.shape[1] == 1024
    assert rows.stride(1) == 1 and state_cache.stride(2) == 1
    assert slot_mapping.numel() == rows.shape[0]
    if rows.shape[0] == 0:
        return
    _save_ring_rows_kernel[(rows.shape[0],)](
        rows,
        slot_mapping,
        state_cache,
        ROW_STRIDE=rows.stride(0),
        STATE_STRIDE=state_cache.stride(0),
        STATE_ROW_STRIDE=state_cache.stride(1),
        STATE_BLOCK=state_cache.shape[1],
        num_warps=4,
    )


@triton.jit
def _save_ring_rows_kernel(
    rows,
    slot_mapping,
    state,
    ROW_STRIDE: tl.constexpr,
    STATE_STRIDE: tl.constexpr,
    STATE_ROW_STRIDE: tl.constexpr,
    STATE_BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    slot = tl.load(slot_mapping + pid)
    if slot < 0:
        return
    d = tl.arange(0, 1024)
    dst = (
        state
        + (slot // STATE_BLOCK).to(tl.int64) * STATE_STRIDE
        + (slot % STATE_BLOCK) * STATE_ROW_STRIDE
    )
    tl.store(dst + d, tl.load(rows + pid.to(tl.int64) * ROW_STRIDE + d))


@triton.jit
def _store_latent(pooled, t, norm_weight, latent, EPS: tl.constexpr):
    d = tl.arange(0, 512)
    weight = tl.load(norm_weight + d).to(tl.float32)
    variance = tl.sum(pooled * pooled, 0) / 512
    normed = pooled * tl.rsqrt(variance + EPS) * weight
    tl.store(latent + t.to(tl.int64) * 512 + d, normed.to(tl.bfloat16))


def rope_quant_insert(
    latent: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    compress_ratio: int,
    fp8_scale: torch.Tensor | None = None,
) -> None:
    """Apply GPT-J RoPE and publish a latent to the compressed KV cache.

    The BF16 latent supplies both NoPE quantization and RoPE input. It is read
    only for valid slots at group boundaries. The cache dtype selects the
    layout: ``uint8`` is a paged FlashMLA layout, whose record the per-token
    byte width names -- 584 B for V4 (576 value bytes and eight segregated
    UE8M0 scale bytes, including one zero padding scale), 528 B for V4.1
    (512 MXFP8 value bytes covering the RoPE dims too, then 16 UE8M0 scales of
    32 dims each), or 288 B for V4.1 NVFP4 (256 bytes of e2m1 pairs then 32
    e4m3 scales of 16 dims each), which only the compressed cache uses. The
    two V4.1 records keep each token's scales right after its own values.
    ``bfloat16`` and ``float8_e4m3fn`` are the plain [448 NoPE | 64 RoPE] rows
    read by FlashInfer, the latter scaled by the per-tensor ``fp8_scale``.
    """
    assert compress_ratio in (1, 2)
    assert latent.shape[1] == 512 and latent.dtype == torch.bfloat16
    assert latent.is_contiguous()
    num_tokens = slot_mapping.numel()
    assert num_tokens <= min(latent.shape[0], positions.numel())
    if num_tokens == 0:
        return
    launch_kwargs = {"launch_pdl": False} if current_platform.is_cuda() else {}
    quant_e2m1 = _quant_e2m1_enabled()
    if kv_cache.dtype == torch.uint8:
        kernel = {
            584: _rope_quant_insert_kernel,
            528: _rope_quant_insert_mxfp8_kernel,
            288: _rope_quant_insert_nvfp4_kernel,
        }.get(kv_cache.shape[-1])
        assert kernel is not None, (
            f"unsupported paged KV record width {kv_cache.shape[-1]}"
        )
        if kernel is _rope_quant_insert_kernel and quant_e2m1:
            # The reference's compressed-cache format (e2m1 with one e4m3
            # scale per 16 dims), stored in our 584 B record.
            kernel = _rope_quant_insert_e2m1_kernel
        kernel[(num_tokens,)](
            latent,
            positions,
            cos_sin_cache,
            kv_cache,
            slot_mapping,
            COS_STRIDE=cos_sin_cache.stride(0),
            CACHE_STRIDE=kv_cache.stride(0),
            CACHE_BLOCK=kv_cache.shape[1],
            COMPRESS_RATIO=compress_ratio,
            SANITIZE_CACHE_NANS=_ON_GFX950,
            num_warps=4,
            **launch_kwargs,
        )
        return

    assert kv_cache.dtype in (torch.bfloat16, torch.float8_e4m3fn)
    assert kv_cache.shape[-1] == 512 and kv_cache.stride(-1) == 1
    store_fp8 = kv_cache.dtype == torch.float8_e4m3fn
    if store_fp8:
        assert fp8_scale is not None and fp8_scale.numel() == 1
        assert fp8_scale.dtype == torch.float32
    _rope_plain_insert_kernel[(num_tokens,)](
        latent,
        positions,
        cos_sin_cache,
        kv_cache,
        slot_mapping,
        fp8_scale if store_fp8 else None,
        COS_STRIDE=cos_sin_cache.stride(0),
        CACHE_STRIDE=kv_cache.stride(0),
        ROW_STRIDE=kv_cache.stride(1),
        CACHE_BLOCK=kv_cache.shape[1],
        COMPRESS_RATIO=compress_ratio,
        STORE_FP8=store_fp8,
        num_warps=4,
        **launch_kwargs,
    )


@triton.jit
def _round_e2m1(scaled):
    """Round to the nearest e2m1 level, with the midpoints going to the
    even level.

    The e2m1 grid: {0, 0.5, 1, 1.5, 2, 3, 4, 6} x the sign. The
    reference's tilelang kernel lowers its FP4 cast to the hardware's
    `cvt.rn.satfinite.e2m1x2.f32`, which rounds to nearest with the
    ties to even - so the midpoints (0.25, 0.75, 1.25, 1.75, 2.5, 3.5,
    5.0) go to the even level (0, 1, 2, 4, 6). The arithmetic is used
    instead of the PTX cast because the FP4 conversion instruction is
    not available on every architecture (SM110 here).
    """
    mag = tl.abs(scaled)
    q = tl.zeros_like(mag)
    q = tl.where(mag >= 5.0, 6.0, q)
    q = tl.where((mag >= 3.5) & (mag < 5.0), 4.0, q)
    q = tl.where((mag >= 2.5) & (mag < 3.5), 3.0, q)
    q = tl.where((mag >= 1.75) & (mag < 2.5), 2.0, q)
    q = tl.where((mag >= 1.25) & (mag < 1.75), 1.5, q)
    q = tl.where((mag >= 0.75) & (mag < 1.25), 1.0, q)
    q = tl.where((mag >= 0.25) & (mag < 0.75), 0.5, q)
    # The exact midpoints go to the even level.
    q = tl.where(mag == 0.25, 0.0, q)
    q = tl.where(mag == 0.75, 1.0, q)
    q = tl.where(mag == 1.25, 1.0, q)
    q = tl.where(mag == 1.75, 2.0, q)
    q = tl.where(mag == 2.5, 2.0, q)
    q = tl.where(mag == 3.5, 4.0, q)
    q = tl.where(mag == 5.0, 4.0, q)
    return q * tl.where(scaled < 0, -1.0, 1.0)


@triton.jit
def _rope_quant_insert_e2m1_kernel(
    latent,
    positions,
    cos_sin,
    cache,
    cache_slots,
    COS_STRIDE: tl.constexpr,
    CACHE_STRIDE: tl.constexpr,
    CACHE_BLOCK: tl.constexpr,
    COMPRESS_RATIO: tl.constexpr,
    SANITIZE_CACHE_NANS: tl.constexpr,
):
    """The reference's compressed-cache format, in our record layout.

    The reference (`inference/model.py`):
        apply_rotary_emb(latent[..., -rope_head_dim:], freqs)
        fp4_act_quant(latent, 16, True, scale_dtype=torch.float8_e4m3fn)
    quantises the WHOLE 512-dim latent to e2m1 with one e4m3 scale per
    16 dims and stores the DEQUANTISED values (the reference's cache is
    a bf16 tensor).

    Our record (the 584 B plain-row record) holds the NoPE dims as fp8
    e4m3 with a UE8M0 scale per 64 dims and the RoPE tail as bf16. So
    the reference's dequantised values are stored there instead - and
    since the attention reads the record back through its own
    dequantisation, the numbers it sees are the reference's.

    **WHY:** the reference's cache is 4-bit and ours is 8-bit, and the
    model was trained on the reference's. Our more accurate cache is a
    DIFFERENT cache: a ~10% perturbation of the attention score, with
    31 of the 512 selected positions changing. See
    notes/DSV41-LOOPING-ROOTCAUSE.md (2aw).

    The NoPE dims need no rotation (their cos/sin are (1, 0)), so they
    are read straight from the latent; only the tail is rotated.
    """
    t = tl.program_id(0)
    slot = tl.load(cache_slots + t)
    if slot < 0:
        return
    position = tl.load(positions + t)
    if (position + 1) % COMPRESS_RATIO != 0:
        return

    page = cache + (slot // CACHE_BLOCK).to(tl.int64) * CACHE_STRIDE
    values = page + (slot % CACHE_BLOCK) * 576
    scales = page + CACHE_BLOCK * 576 + (slot % CACHE_BLOCK) * 8
    base = position // COMPRESS_RATIO * COMPRESS_RATIO

    # ---- the latent as a [32, 16] tile: the 32 groups of 16 (the
    # reference's e2m1 block size). The NoPE dims are the first 28 rows
    # (448 dims) and the tail is the last 4 rows (64 dims).
    row = tl.arange(0, 32)
    col = tl.arange(0, 16)
    loff = row[:, None] * 16 + col[None, :]
    lat = tl.load(latent + t.to(tl.int64) * 512 + loff).to(tl.float32)
    # The rotation partner: the pair partner of dim d is d^1, and the
    # pairs never cross a 16-dim row (the rows start at even dims).
    part = tl.load(latent + t.to(tl.int64) * 512 + (loff ^ 1)).to(tl.float32)
    # The cos/sin: the tail's 32 pairs (the NoPE dims' cos/sin are
    # (1, 0), so they are not rotated).
    in_tail = row >= 28
    # The tail's 32 pairs: the pair p covers dims 448+2p and 448+2p+1.
    # The NoPE dims' pairs have cos/sin (1, 0), so the identity.
    pidx = tl.where(in_tail[:, None],
                    (row[:, None] - 28) * 8 + col[None, :] // 2,
                    0)
    cs = cos_sin + base * COS_STRIDE
    cc = tl.load(cs + pidx).to(tl.float32)
    ss = tl.load(cs + 32 + pidx).to(tl.float32)
    is_even = (col % 2) == 0
    rot = tl.where(is_even, lat * cc - part * ss, lat * cc + part * ss)
    rot = tl.where(in_tail[:, None], rot, lat)
    if SANITIZE_CACHE_NANS:
        rot = tl.where(rot == rot, rot, 0.0)

    # The reference's scale: e4m3(amax/6) per group of 16, over the
    # WHOLE 512 dims.
    amax16 = tl.maximum(tl.max(tl.abs(rot), 1), 6.0 * (2.0**-9))
    sc16 = (amax16 * (1.0 / 6.0)).to(tl.float8e4nv).to(tl.float32)
    scaled = tl.clamp(tl.math.div_rn(rot, sc16[:, None]), -6.0, 6.0)
    deq = _round_e2m1(scaled) * sc16[:, None]

    # ---- store: the NoPE dims as fp8 (our record), with the UE8M0
    # scale per 64 dims. The [8, 64] view gives the 7 groups of 64
    # NoPE dims (the 8th is the tail, which is stored as bf16).
    d8 = tl.reshape(deq, (8, 64))
    amax64 = tl.maximum(tl.max(tl.abs(d8), 1), 1e-4)
    exponent = tl.ceil(tl.log2(amax64 * (1.0 / 448.0)))
    fp8 = tl.clamp(d8 * tl.exp2(-exponent)[:, None], -448.0, 448.0)
    fp8 = fp8.to(tl.float8e4nv)
    n8 = tl.arange(0, 8)
    c64 = tl.arange(0, 64)
    nidx8 = n8[:, None] * 64 + c64[None, :]
    tl.store(values + nidx8, fp8.to(tl.uint8, bitcast=True), nidx8 < 448)
    enc = tl.minimum(tl.maximum(exponent + 127.0, 0.0), 255.0).to(tl.uint8)
    tl.store(scales + n8, enc, n8 < 7)
    tl.store(scales + 7, tl.full((), 0, tl.uint8))

    # ---- store: the tail as bf16.
    rope_dst = (values + 448).to(tl.pointer_type(tl.bfloat16))
    tl.store(rope_dst + loff - 448, deq.to(tl.bfloat16), in_tail[:, None])


@triton.jit
def _rope_quant_insert_kernel(
    latent,
    positions,
    cos_sin,
    cache,
    cache_slots,
    COS_STRIDE: tl.constexpr,
    CACHE_STRIDE: tl.constexpr,
    CACHE_BLOCK: tl.constexpr,
    COMPRESS_RATIO: tl.constexpr,
    SANITIZE_CACHE_NANS: tl.constexpr,
):
    t = tl.program_id(0)
    slot = tl.load(cache_slots + t)
    if slot < 0:
        return
    position = tl.load(positions + t)
    if (position + 1) % COMPRESS_RATIO != 0:
        return
    d = tl.arange(0, 512)
    normed = tl.load(latent + t.to(tl.int64) * 512 + d).to(tl.float32)
    page = cache + (slot // CACHE_BLOCK).to(tl.int64) * CACHE_STRIDE
    values = page + (slot % CACHE_BLOCK) * 576
    scales = page + CACHE_BLOCK * 576 + (slot % CACHE_BLOCK) * 8

    even, odd = tl.split(tl.reshape(normed, (256, 2)))
    pair = tl.arange(0, 256) - 224
    cs = cos_sin + (position // COMPRESS_RATIO * COMPRESS_RATIO) * COS_STRIDE
    c = tl.load(cs + tl.maximum(pair, 0), pair >= 0, other=1.0).to(tl.float32)
    s = tl.load(cs + 32 + tl.maximum(pair, 0), pair >= 0, other=0.0).to(tl.float32)
    rotated = tl.interleave(even * c - odd * s, odd * c + even * s)
    if SANITIZE_CACHE_NANS:
        rotated = tl.where(rotated == rotated, rotated, 0.0)

    # The NoPE dims (the first 448 of the latent): our record stores
    # them as fp8 e4m3 with a UE8M0 scale per 64 dims.
    quant = tl.reshape(normed, (8, 64))
    amax = tl.maximum(tl.max(tl.abs(quant), 1), 1e-4)
    exponent = tl.ceil(tl.log2(amax * (1.0 / 448.0)))
    scaled = quant * tl.reshape(tl.exp2(-exponent), (8, 1))
    fp8 = tl.clamp(scaled, -448.0, 448.0).to(tl.float8e4nv)
    packed = tl.reshape(fp8.to(tl.uint8, bitcast=True), (512,))
    tl.store(values + d, packed, d < 448)
    s = tl.arange(0, 8)
    max_encoded: tl.constexpr = 254.0 if SANITIZE_CACHE_NANS else 255.0
    encoded = tl.minimum(tl.maximum(exponent + 127.0, 0.0), max_encoded)
    tl.store(scales + s, encoded.to(tl.uint8), s < 7)
    tl.store(scales + 7, tl.full((), 0, tl.uint8))

    rope_dst = (values + 448).to(tl.pointer_type(tl.bfloat16))
    tl.store(rope_dst + d - 448, rotated.to(tl.bfloat16), d >= 448)


@triton.jit
def _rope_quant_insert_mxfp8_kernel(
    latent,
    positions,
    cos_sin,
    cache,
    cache_slots,
    COS_STRIDE: tl.constexpr,
    CACHE_STRIDE: tl.constexpr,
    CACHE_BLOCK: tl.constexpr,
    COMPRESS_RATIO: tl.constexpr,
    SANITIZE_CACHE_NANS: tl.constexpr,
):
    """V4.1 record: RoPE first, then MXFP8-quantize all 512 dims.

    The RoPE dims are quantized here too, so unlike the V4 kernel the rotation
    has to happen before the scales are picked.
    """
    t = tl.program_id(0)
    slot = tl.load(cache_slots + t)
    if slot < 0:
        return
    position = tl.load(positions + t)
    if (position + 1) % COMPRESS_RATIO != 0:
        return
    d = tl.arange(0, 512)
    normed = tl.load(latent + t.to(tl.int64) * 512 + d).to(tl.float32)

    # NoPE pairs load (cos, sin) = (1, 0), so the rotation is the identity there.
    even, odd = tl.split(tl.reshape(normed, (256, 2)))
    pair = tl.arange(0, 256) - 224
    cs = cos_sin + (position // COMPRESS_RATIO * COMPRESS_RATIO) * COS_STRIDE
    c = tl.load(cs + tl.maximum(pair, 0), pair >= 0, other=1.0).to(tl.float32)
    s = tl.load(cs + 32 + tl.maximum(pair, 0), pair >= 0, other=0.0).to(tl.float32)
    rotated = tl.interleave(even * c - odd * s, odd * c + even * s)
    if SANITIZE_CACHE_NANS:
        rotated = tl.where(rotated == rotated, rotated, 0.0)

    page = cache + (slot // CACHE_BLOCK).to(tl.int64) * CACHE_STRIDE
    values = page + (slot % CACHE_BLOCK) * 528
    scales = values + 512

    quant = tl.reshape(rotated, (16, 32))
    amax = tl.maximum(tl.max(tl.abs(quant), 1), 1e-4)
    exponent = tl.ceil(tl.log2(amax * (1.0 / 448.0)))
    scaled = quant * tl.reshape(tl.exp2(-exponent), (16, 1))
    fp8 = tl.clamp(scaled, -448.0, 448.0).to(tl.float8e4nv)
    tl.store(values + d, tl.reshape(fp8.to(tl.uint8, bitcast=True), (512,)))

    max_encoded: tl.constexpr = 254.0 if SANITIZE_CACHE_NANS else 255.0
    encoded = tl.minimum(tl.maximum(exponent + 127.0, 0.0), max_encoded)
    tl.store(scales + tl.arange(0, 16), encoded.to(tl.uint8))


@triton.jit
def _rope_quant_insert_nvfp4_kernel(
    latent,
    positions,
    cos_sin,
    cache,
    cache_slots,
    COS_STRIDE: tl.constexpr,
    CACHE_STRIDE: tl.constexpr,
    CACHE_BLOCK: tl.constexpr,
    COMPRESS_RATIO: tl.constexpr,
    SANITIZE_CACHE_NANS: tl.constexpr,
):
    """V4.1 NVFP4 record: RoPE, then e2m1 with one e4m3 scale per 16 dims.

    The scale is ``amax / 6`` (6 is e2m1's largest magnitude) clamped to the
    e4m3 range, with no per-tensor scale on top.
    """
    t = tl.program_id(0)
    slot = tl.load(cache_slots + t)
    if slot < 0:
        return
    position = tl.load(positions + t)
    if (position + 1) % COMPRESS_RATIO != 0:
        return
    d = tl.arange(0, 512)
    normed = tl.load(latent + t.to(tl.int64) * 512 + d).to(tl.float32)

    # NoPE pairs load (cos, sin) = (1, 0), so the rotation is the identity there.
    even, odd = tl.split(tl.reshape(normed, (256, 2)))
    pair = tl.arange(0, 256) - 224
    cs = cos_sin + (position // COMPRESS_RATIO * COMPRESS_RATIO) * COS_STRIDE
    c = tl.load(cs + tl.maximum(pair, 0), pair >= 0, other=1.0).to(tl.float32)
    s = tl.load(cs + 32 + tl.maximum(pair, 0), pair >= 0, other=0.0).to(tl.float32)
    rotated = tl.interleave(even * c - odd * s, odd * c + even * s)
    if SANITIZE_CACHE_NANS:
        rotated = tl.where(rotated == rotated, rotated, 0.0)

    tiles = tl.reshape(rotated, (32, 16))
    amax = tl.max(tl.abs(tiles), 1)
    # 2**-9 is the smallest normal e4m3 magnitude; 448 the largest.
    scale = tl.clamp(amax * (1.0 / 6.0), 0.001953125, 448.0).to(tl.float8e4nv)
    # Round-to-nearest division: Triton's default div.full misplaces values
    # that land exactly on an e2m1 tie.
    scaled = tl.math.div_rn(tiles, tl.reshape(scale.to(tl.float32), (32, 1)))
    lo, hi = tl.split(tl.reshape(tl.reshape(scaled, (512,)), (256, 2)))

    page = cache + (slot // CACHE_BLOCK).to(tl.int64) * CACHE_STRIDE
    values = page + (slot % CACHE_BLOCK) * 288
    tl.store(values + tl.arange(0, 256), _fp32x2_to_fp4x2(lo, hi))
    tl.store(values + 256 + tl.arange(0, 32), scale.to(tl.uint8, bitcast=True))


@triton.jit
def _rope_plain_insert_kernel(
    latent,
    positions,
    cos_sin,
    cache,
    cache_slots,
    fp8_scale,
    COS_STRIDE: tl.constexpr,
    CACHE_STRIDE: tl.constexpr,
    ROW_STRIDE: tl.constexpr,
    CACHE_BLOCK: tl.constexpr,
    COMPRESS_RATIO: tl.constexpr,
    STORE_FP8: tl.constexpr,
):
    t = tl.program_id(0)
    slot = tl.load(cache_slots + t)
    if slot < 0:
        return
    position = tl.load(positions + t)
    if (position + 1) % COMPRESS_RATIO != 0:
        return
    d = tl.arange(0, 512)
    normed = tl.load(latent + t.to(tl.int64) * 512 + d).to(tl.float32)

    # NoPE pairs load (cos, sin) = (1, 0), so the rotation is the identity there.
    even, odd = tl.split(tl.reshape(normed, (256, 2)))
    pair = tl.arange(0, 256) - 224
    cs = cos_sin + (position // COMPRESS_RATIO * COMPRESS_RATIO) * COS_STRIDE
    c = tl.load(cs + tl.maximum(pair, 0), pair >= 0, other=1.0).to(tl.float32)
    s = tl.load(cs + 32 + tl.maximum(pair, 0), pair >= 0, other=0.0).to(tl.float32)
    row = tl.interleave(even * c - odd * s, odd * c + even * s).to(tl.bfloat16)

    dst = (
        cache
        + (slot // CACHE_BLOCK).to(tl.int64) * CACHE_STRIDE
        + (slot % CACHE_BLOCK) * ROW_STRIDE
    )
    if STORE_FP8:
        scaled = row.to(tl.float32) * (1.0 / tl.load(fp8_scale))
        tl.store(dst + d, tl.clamp(scaled, -448.0, 448.0).to(tl.float8e4nv))
    else:
        tl.store(dst + d, row)
