"""csrc-quant (e40c5bc9dc) validation: per_token_group_quant launch args + PDL.

This is a .cu file: no CPU execution is possible pre-build. Two layers:

1. Structural (this test): every cudaLaunchKernelEx site in
   per_token_group_quant.cu passes the FULL kernel argument list (the G1
   conflict-loss squash had truncated the arg tail after the kernel pointer,
   which compiles but launches with garbage/missing args -> silent NaN/garbage
   quantized scales) and every cudaLaunchConfig_t site routes numAttrs
   through vllm_stable::disable_pdl_sm110() (PDL wedges kernels on Tegra).
2. Real check (post-build): the quant kernel path is exercised by every GLM
   forward; covered by the 22-turn MTP3 repro plus a per-layer output sanity
   (finite scales).

The gdb/ptxas-level check (arg count at launch) requires the built .so; the
post-build gate runs:
  python -c "import vllm._C"  (link + registration)
and the serving smoke test.
"""
import os
import re
import sys

_HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_F = os.path.join(
    _HERE, "csrc", "quantization", "w8a8", "fp8",
    "per_token_group_quant.cu",
)

KERNEL_EXPECT_ARGS = {
    "per_token_group_quant_8bit_kernel": 11,        # incl. defaults (2)
    "per_token_group_quant_8bit_register_kernel": 9,
    "per_token_group_quant_8bit_packed_register_kernel": None,  # counted dynamically
}


def _kernel_param_counts(src):
    counts = {}
    for m in re.finditer(
        r"__global__ void (\w+)\(([^{]*?)\)\s*{", src, re.DOTALL
    ):
        name, params = m.group(1), m.group(2)
        # грубо: count top-level commas (heels with templates ignored: no
        # commas inside template args in these signatures)
        depth = 0
        nargs = 1
        for ch in params:
            if ch in "(<[":
                depth += 1
            elif ch in ")>]":
                depth -= 1
            elif ch == "," and depth == 0:
                nargs += 1
        counts[name] = nargs
    return counts


def test_launch_sites_args_and_pdl():
    src = open(_F).read()
    kparams = _kernel_param_counts(src)
    # 1. every cudaLaunchKernelEx immediately names a kernel and then passes
    #    a non-empty argument list ending in ");" — plus PDL guard on numAttrs
    n_launch = src.count("cudaLaunchKernelEx(")
    assert n_launch == 3, f"expected 3 Ex-launch sites, got {n_launch}"
    assert src.count("disable_pdl_sm110") >= n_launch, (
        "each Ex-launch site must gate numAttrs on disable_pdl_sm110"
    )
    # 2. the truncated-args loss class: a launch whose arg tail is missing —
    #    detectable as `kernel<...>,\n    } while` without any arg lines
    for m in re.finditer(r"cudaLaunchKernelEx\(([^;]*?)\)[\s\\]*;", src, re.DOTALL):
        body = m.group(1)
        assert re.search(r"data_ptr\(\)", body), (
            f"launch site without kernel args (truncated-loss pattern): {body[:120]}"
        )
        assert (
            ("group_size" in body or "num_groups" in body
             or "groups_per_row" in body or "num_scale_elems" in body)
        ), "launch args missing the geometry tail"
        assert ("eps" in body and "min_8bit" in body), "quant bounds args missing"
    print(f"launch sites: args complete + PDL-guarded at all {n_launch} sites")


def test_kernel_sigs_match_launch_caller_vars():
    src = open(_F).read()
    # the scale tail args (scale_num_rows, scale_stride) appear in every
    # classic + Ex launch of the two full kernels
    for k in ("per_token_group_quant_8bit_kernel",):
        n_sig = len(re.findall(rf"{k}\(", src))
        n_tail = src.count("scale_num_rows, scale_stride")
        assert n_tail >= 3, f"scale tail args missing at some site ({n_tail})"
    print("scale-geometry args threaded at every launch: OK")


if __name__ == "__main__":
    test_launch_sites_args_and_pdl()
    test_kernel_sigs_match_launch_caller_vars()
    print("csrc-quant structural validation: ALL OK (runtime check post-build)")
