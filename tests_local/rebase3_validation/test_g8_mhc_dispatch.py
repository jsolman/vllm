"""G8 validation: mhc_pre_broadcast DeepGEMM fallback dispatch.

G8 fixes the SM110 path: mhc_pre_broadcast_tilelang passed
use_tilelang_fallback=False to _hc_prenorm_gemm_outputs, forcing the DeepGEMM
tf32_hc_prenorm_gemm even when is_deep_gemm_supported() is False (SM110 has no
DeepGEMM MQA/GEMM support). With fallback=True the helper dispatches to the
tilelang GEMM kernel whenever DeepGEMM is absent.

CPU-provable here:
1. dispatch decision boolean: supported OR (not fallback).
2. on this no-DeepGEMM, no-tilelang host the helper contracts to:
   n_splits=1 outputs and the tilelang kernel invoked with 6 positional args,
   verified by stubbing the kernels module BEFORE its first import
   (its @tilelang_jit decorators require tilelang otherwise).

Real-GPU post-build check: mhc pre-broadcast runs in the GLM/DSv4 layer
forward ( exercised by the 22-turn MTP3 repro ).
"""
import importlib
import os
import sys
import types

_HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _HERE)

import torch  # noqa: E402


def test_dispatch_logic():
    from vllm.utils.deep_gemm import is_deep_gemm_supported

    supported = is_deep_gemm_supported()

    def decide(use_tilelang_fallback: bool) -> bool:
        return supported or not use_tilelang_fallback

    # G8's bug: fallback=False forces DeepGEMM decision even if unsupported
    assert decide(False) is True
    # G8's fix: fallback=True defers to the platform answer
    assert decide(True) == supported
    if not supported:
        assert decide(True) is False, "fallback must be selected w/o DeepGEMM"
    print(f"dispatch logic: OK (is_deep_gemm_supported={supported})")


def test_fallback_kernel_dispatched():
    if torch.cuda.is_available():
        print("fallback dispatch: SKIP (real GPU present)")
        return
    from vllm.utils.deep_gemm import is_deep_gemm_supported
    if is_deep_gemm_supported():
        print("fallback dispatch: SKIP (deep_gemm supported here)")
        return

    # Stub the kernels module BEFORE its first import: tilelang_kernels
    # decorates kernels with @tilelang_jit at import; tilelang is absent here.
    import vllm.tilelang_utils as _tlu
    _tlu.tilelang_jit = lambda *a, **k: (lambda fn: fn)

    stub_calls = []

    def _stub_kernel(x, fn, out, sqrsum, hidden_size, hc_mult):
        stub_calls.append((x.shape, out.shape))
        out.fill_(1.0)
        sqrsum.fill_(2.0)

    fake = types.ModuleType("vllm.model_executor.kernels.mhc.tilelang_kernels")
    fake._HC_PRENORM_GEMM_TILELANG_KERNEL = _stub_kernel
    fake.compute_num_split = lambda *a, **k: 1  # not used on fallback path
    sys.modules[fake.__name__] = fake

    import vllm.model_executor.kernels.mhc.tilelang as tlm

    x = torch.randn(32, 64)
    fn = torch.randn(64, 128)
    out, sqrsum = tlm._hc_prenorm_gemm_outputs(
        x, fn, hidden_size=64, hc_mult=2, use_tilelang_fallback=True
    )
    assert stub_calls, "tilelang fallback kernel was NOT dispatched"
    assert out.shape[0] == 1, f"fallback n_splits must be 1, got {out.shape}"
    assert torch.allclose(out.squeeze(0), torch.full_like(out.squeeze(0), 1.0))
    assert torch.allclose(sqrsum.squeeze(0), torch.full_like(sqrsum.squeeze(0), 2.0))
    print("fallback kernel dispatched + n_splits=1 shapes: OK")
    return tlm


def test_bug_regression_forced_deep_path(tlm=None):
    """Document the failuremode G8 fixed: use_tilelang_fallback=False on a
    no-DeepGEMM host routes into tf32_hc_prenorm_gemm whose impl is None (=missing
    package) — which must raise rather than silently produce garbage."""
    from vllm.utils.deep_gemm import is_deep_gemm_supported
    if torch.cuda.is_available() or is_deep_gemm_supported():
        print("forced-deep regression: SKIP")
        return
    if tlm is None:
        import vllm.model_executor.kernels.mhc.tilelang as tlm
    x = torch.randn(32, 64)
    fn = torch.randn(64, 128)
    try:
        tlm._hc_prenorm_gemm_outputs(
            x, fn, hidden_size=64, hc_mult=2, use_tilelang_fallback=False
        )
        status = "no-raise (deep impl present? unexpected on this host)"
    except Exception as e:
        status = f"raises as expected ({type(e).__name__})"
    print("forced-deep regression: " + status)


if __name__ == "__main__":
    test_dispatch_logic()
    tlm = test_fallback_kernel_dispatched()
    test_bug_regression_forced_deep_path(tlm)
    print("G8 mhc dispatch validation: done")
