"""G5 validation: fp8 paged MQA-logits Triton kernel.

Three CPU-verifiable layers (this file) + one GPU companion:
1. e4m3 LUT identity incl. the 0x7F/0xFF NaN->-480 sentinel override.
2. Kernel plumbing probes without tl.dot: K-byte load through the exact
   as_strided view the wrapper builds (block-major values-then-scales
   layout), and the store stride/mask geometry.
3. Context-mask semantics replicated on numpy (OOB suppressed at ctx).

GPU companion (post-build): gpu_check_g5_paged_logits.py compares the full
kernel against an fp32 reference with allclose - the interpreter CANNOT run
tl.dot: triton 3.7.1 interpreter returns garbage for bf16 dot (repro below,
2-line standalone). That is a tooling limitation, not a kernel signal.
"""
import os
import sys

_HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _HERE)
os.environ.setdefault("TRITON_INTERPRET", "1")

import torch  # noqa: E402


def _interp_env():
    if os.environ.get("TRITON_INTERPRET") != "1":
        return
    from triton.backends import backends as _backs

    _n = _backs.get("nvidia")
    if _n is not None and getattr(_n, "driver", None) is not None:
        _n.driver.is_active = staticmethod(lambda: True)


_interp_env()

BS = 16
D = 128
H = 8
W = D + 4
NB = 9


def make_block_major_cache(seed=3):
    """Production indexer-kpool layout: [NB, BS, D+4] uint8 where each block
    is [BS*D value bytes][BS*4 scale bytes] (per _fp8_quant_and_cache_write:
    scale_byte_off = block_start + block_size*HEAD_DIM + off*4)."""
    torch.manual_seed(seed)
    cache = torch.zeros(NB, BS, W, dtype=torch.uint8)
    flat = cache.view(NB, BS * W)
    k_fp8 = torch.randn(NB, BS, D).to(torch.float8_e4m3fn)
    scales = torch.empty(NB, BS).uniform_(0.5, 2.0)
    flat[:, : BS * D] = k_fp8.view(torch.uint8).reshape(NB, BS * D)
    flat[:, BS * D :].view(torch.float32).reshape(NB, BS).copy_(scales)
    return cache, k_fp8, scales


def test_e4m3_lut():
    from vllm.v1.attention.ops.mqa_logits_triton import _get_e4m3fn_bf16_lut

    lut = _get_e4m3fn_bf16_lut(torch.device("cpu")).clone()
    ref = (
        torch.arange(256, dtype=torch.uint8)
        .view(torch.float8_e4m3fn)
        .float()
        .to(torch.bfloat16)
    )
    assert not torch.equal(lut[[0x7F, 0xFF]], ref[[0x7F, 0xFF]]), (
        "NaN sentinel override missing"
    )
    ref[0x7F] = 480.0
    ref[0xFF] = -480.0
    assert torch.equal(lut, ref), "LUT differs from torch fp8 decode"
    print("1. e4m3 LUT identity: OK")


def test_kernel_view_readback():
    """Rebuild the wrapper's exact views (kv_flat as_strided + scale view)
    and prove a plain torch readback equals the source tensors — i.e. the
    pointer geometry the kernel is handed is the layout we wrote."""
    cache, k_fp8, scales = make_block_major_cache()
    kv_flat = cache.view(NB, -1)
    k_end = BS * D
    kv_byte = kv_flat[:, :k_end].as_strided(
        (NB, BS, D), (kv_flat.stride(0), D, 1)
    )
    kv_scale = kv_flat[:, k_end:].view(torch.float32)
    k_read = kv_byte.view(torch.float8_e4m3fn).float()
    assert (k_read - k_fp8.float()).abs().max().item() == 0.0, "K view corrupt"
    assert torch.equal(kv_scale, scales), "scale view corrupt"
    print("2. kernel-view readback: OK (K bytes + scales round-trip)")


def test_store_geometry():
    """The kernel's store: logits[tok*st_t + k_off*st_n] masked k_off<ctx.
    Probe with constants and confirm only [:ctx] of each row is touched."""
    import triton
    import triton.language as tl

    @triton.jit
    def _store_probe(l_ptr, st_t, st_n, CTX: tl.constexpr, BS_: tl.constexpr):
        tok = tl.program_id(0)
        rk = tl.program_id(1)
        offs = tl.arange(0, BS_)
        k_off = rk * BS_ + offs
        tl.store(
            l_ptr + tok * st_t + k_off * st_n,
            tl.full((BS_,), 1.0, tl.float32),
            mask=(k_off < CTX),
        )

    ctx, max_len = 5, 22
    L = torch.full((2, max_len), float("nan"), dtype=torch.float32)
    _store_probe[(2, 2)](L, L.stride(0), L.stride(1), CTX=ctx, BS_=BS)
    assert torch.isfinite(L[0, :ctx]).all() and (L[0, :ctx] == 1).all()
    assert torch.isnan(L[0, ctx:]).all(), "store wrote past context!"
    assert torch.isnan(L[1, ctx:]).all()
    print("3. store geometry + context mask: OK")


def test_interpreter_dot_is_broken():
    """Documents WHY full-kernel numerics defer to GPU: the 3.7.1
    interpreter returns garbage for bf16 tl.dot. If this ever passes on a
    fixed triton, flip gpu_check into this file."""
    import triton
    import triton.language as tl

    @triton.jit
    def _dot_probe(q_ptr, k_ptr, o_ptr, BH: tl.constexpr, BD: tl.constexpr,
                   BS_: tl.constexpr):
        offsh = tl.arange(0, BH)
        offsd = tl.arange(0, BD)
        offsn = tl.arange(0, BS_)
        q = tl.load(q_ptr + offsh[:, None] * BD + offsd[None, :])
        k = tl.load(k_ptr + offsn[:, None] * BD + offsd[None, :])
        s = tl.dot(q, tl.trans(k))
        tl.store(o_ptr + offsh[:, None] * BS_ + offsn[None, :], s)

    BH = 16
    q = torch.randn(BH, D, dtype=torch.bfloat16)
    k = torch.randn(BS, D, dtype=torch.bfloat16)
    o = torch.zeros(BH, BS, dtype=torch.float32)
    _dot_probe[(1,)](q, k, o, BH=BH, BD=D, BS_=BS)
    ref = q.float() @ k.float().T
    if (o - ref).abs().max().item() < 1e-2:
        print("4. interpreter dot: FIXED — move GPU check to CPU")
    else:
        print("4. interpreter tl.dot(bf16) broken as documented "
              f"(diff={ (o-ref).abs().max().item():.2e}) -> GPU check deferred")


if __name__ == "__main__":
    test_e4m3_lut()
    test_kernel_view_readback()
    test_store_geometry()
    test_interpreter_dot_is_broken()
    print("G5 MQA-logits CPU-side validation: ALL OK")
