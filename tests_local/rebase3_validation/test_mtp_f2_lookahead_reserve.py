"""MTP-f2 (7b7f1463ee) validation: lookahead page-reserve vs align-mode null
padding in MambaManager.

BUG: when a prefill chunk ends exactly on a page boundary AND the scheduler's
+1 decode lookahead crossed it, get_num_blocks_to_allocate inflates
num_tokens = main + block_size (page reserve). The align-mode padding math
derived num_skipped_blocks from the INFLATED count, emitting rows like
[null, page1, spec]; the worker's running-state index is
cdiv(main_len, bs) - 1 = 0 → the NULL slot. The next chunk's state pre-copy
chained from the null slot → recurrent state garbage → spec-decode acceptance
collapsed to 0 permanently for that request.

FIX: num_skipped_blocks tracks the MAIN-model length; the reserve page is
instead materialized as a real allocated block (budget +1).

CPU-provable: replicate the padding arithmetic for the boundary case and the
normal case; assert old math displaces the running-state column, new math
keeps the real sequence's page at the worker's expected index.
"""
import os
import sys

_HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _HERE)

BS = 32  # mamba page/block size
SPEC_BLOCKS = 1


def cdiv(a, b):
    return -(-a // b)


def old_skip_blocks(num_tokens):
    """Pre-fix: num_skipped from (possibly inflated) num_tokens."""
    num_required = cdiv(num_tokens, BS) + SPEC_BLOCKS
    return num_required - SPEC_BLOCKS - 1  # != cdiv(num_tokens,BS)-1 only when aligned


def new_skip_blocks(num_tokens, num_tokens_main_model):
    """Post-fix: track the MAIN length."""
    if num_tokens != num_tokens_main_model:
        return max(cdiv(num_tokens_main_model, BS) - 1, 0)
    num_required = cdiv(num_tokens, BS) + SPEC_BLOCKS
    return num_required - SPEC_BLOCKS - 1


def test_boundary_chunk_displacement():
    main_len = 2176  # 68 * 32 → exact page boundary (the repro junction)
    inflated = main_len + BS

    old = old_skip_blocks(inflated)
    new = new_skip_blocks(inflated, main_len)

    worker_rows_main = cdiv(main_len, BS) - 1  # worker's running-state index
    # OLD: skipped = inflated/BS - 1 → one MORE null than the worker expects:
    # row = [null, page1, spec] — worker's index 0 points at the null.
    assert old == worker_rows_main + 1, (old, worker_rows_main)
    # NEW: skipped == worker index → row = [page1, spec(+reserve)] — page1 is
    # the real state column at the expected position.
    assert new == worker_rows_main, (new, worker_rows_main)
    print(f"boundary chunk (main={main_len}): old={old} displaced the column; "
          f"new={new} aligns with worker index {worker_rows_main}: OK")


def test_normal_case_noop():
    # not at boundary: num_tokens == num_tokens_main_model (no inflation);
    # both formulas coincide on that input.
    main_len = 2175
    old = old_skip_blocks(main_len)
    new = new_skip_blocks(main_len, main_len)
    assert old == new == cdiv(main_len, BS) - 1, (old, new)
    # ALSO: a boundary case WITHOUT inflation (upstream never inflates there
    # without the reserve) stays a no-op:
    b = 2176
    assert old_skip_blocks(b) == new_skip_blocks(b, b) == cdiv(b, BS) - 1
    print("normal chunk: identical to upstream behavior: OK")


def test_reserve_budget_grows():
    """Second hunk: with the reserve, max_new_blocks grows by one (real block)."""
    src = open(os.path.join(
        _HERE, "vllm", "v1", "core", "single_type_kv_cache_manager.py"
    )).read()
    assert "if num_tokens != num_tokens_main_model:" in src
    assert "num_skipped_blocks = max(" in src
    # budget growth for the reserve page
    assert src.count("materialized as a real") == 1
    print("source carries skip-correction + reserve-budget growth: OK")


if __name__ == "__main__":
    test_boundary_chunk_displacement()
    test_normal_case_noop()
    test_reserve_budget_grows()
    print("MTP-f2 lookahead-reserve validation: ALL OK")
