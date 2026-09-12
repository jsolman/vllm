"""MTP-f2 (7b7f1463ee) validation — UPDATED for upstream-parity revert (10/19).

HISTORY: against the rebase-3-era base (2026-09-18), the align-mode padding
math derived num_skipped_blocks from the lookahead-inflated num_tokens,
emitting rows like [null, page1, spec]; the worker's running-state index is
cdiv(main_len, bs) - 1 → the NULL slot → next chunk's state pre-copy chained
from the null slot → recurrent state garbage → spec-decode acceptance
collapsed to 0 permanently. The fix tracked num_skipped_blocks on the MAIN
length and materialized the reserve page (budget +1).

POST-REBASE: upstream's align-mode manager evolved (scratch/speculative
column relocation _relocate_speculative_block, checkpoint handling) and its
admission/allocation IGNORE lookahead for mamba ("current draft models don't
have mamba layers"). The upstream pytest tier (manager + chunk-split suites,
95 tests) encodes that economy; our two mirror hunks broke it (10 failures:
block economy off by one at boundaries). REVERTED in the rebase to upstream
parity; the original 9/18 bug does not reproduce on the evolved upstream
manager (the scratch column covers the boundary write).

CPU-provable here: (a) the historical bug arithmetic still demonstrates the
displacement mechanism on the OLD math (kept for the record); (b) the current
tree carries NO residual mirror hunks (grep-level); (c) the manager suites
must pass at exact upstream parity (asserted in run_all on the manager files).
"""
import os
import re
import sys

_HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _HERE)

BS = 32  # mamba page/block size
SPEC_BLOCKS = 1
_MGR = os.path.join(
    _HERE, "vllm", "v1", "core", "single_type_kv_cache_manager.py"
)


def cdiv(a, b):
    return -(-a // b)


def old_skip_blocks(num_tokens):
    """Pre-fix (rebase-3 era): num_skipped from (possibly inflated) num_tokens."""
    num_required = cdiv(num_tokens, BS) + SPEC_BLOCKS
    return num_required - SPEC_BLOCKS - 1


def fixed_skip_blocks(num_tokens, num_tokens_main_model):
    """The 9/18 fix (kept for the historical record of the mechanism)."""
    if num_tokens != num_tokens_main_model:
        return max(cdiv(num_tokens_main_model, BS) - 1, 0)
    return old_skip_blocks(num_tokens)


def test_historical_mechanism_documented():
    """The displacement mechanism the original fix addressed (old math)."""
    main_len = 2176  # 68 * 32 → exact page boundary (the repro junction)
    inflated = main_len + BS
    old = old_skip_blocks(inflated)
    worker_rows_main = cdiv(main_len, BS) - 1
    assert old == worker_rows_main + 1, (old, worker_rows_main)
    assert fixed_skip_blocks(inflated, main_len) == worker_rows_main
    print(f"historical mechanism: inflated math displaces the running-state "
          f"column by 1 ({old} vs {worker_rows_main}): OK")


def test_tree_at_upstream_parity():
    """The rebased tree must carry NO residual mirror/fixup hunks in the
    align-mode allocation (upstream parity, verified by grep)."""
    src = open(_MGR).read()
    assert "_gl53_la" not in src, "G6 admission/allocation mirror still present"
    assert src.count("materialized as a real") == 0, (
        "MTP-f2 reserve-budget hunk still present"
    )
    assert "num_skipped_blocks = max(" not in src, (
        "MTP-f2 skip-correction hunk still present"
    )
    # and upstream's own ignore-lookahead line must be back at both sites
    n = src.count("num_tokens = num_tokens_main_model")
    assert n >= 2, f"upstream ignore-lookahead sites missing ({n})"
    print("tree at upstream parity: no residual mirror hunks: OK")


if __name__ == "__main__":
    test_historical_mechanism_documented()
    test_tree_at_upstream_parity()
    print("MTP-f2 lookahead-reserve validation: ALL OK")
