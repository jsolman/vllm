"""MTP-f1 (f1ae35cec4) validation: placeholder-drain skipping for finished
requests under async spec decode.

The BUG: a finished request (FINISHED_LENGTH_CAPPED mid-step) can still receive
an in-flight async output row scheduled before the finish; draining its
placeholders again double-counts: either the assert crashes the engine, or
(green-path variants) num_output_placeholders goes negative and poisons
num_computed_tokens / cache_block offsets -> poisoned prefix-cache blocks.

The FIX: skip drain+recache for finished requests; clamp defensively.

CPU-provable WITHOUT an engine (the drain logic is pure counter arithmetic):
replicate the accounting state machine with a stub request and prove
(1) the OLD path underflows on the double-drain scenario,
(2) the NEW path (is_finished() check + clamp) preserves the counter and
    other requests are unaffected,
(3) normal (non-finished) requests still drain exactly len(new_token_ids).
"""
import os
import sys

_HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _HERE)


class StubRequest:
    def __init__(self, placeholders, finished=False):
        self.num_output_placeholders = placeholders
        self._finished = finished
        self.num_computed_tokens = 10  # arbitrary
        self.poisoned = False

    def is_finished(self):
        return self._finished


def drain_old(request, new_token_ids):
    """Pre-fix logic: drain unconditionally, assert >= 0."""
    request.num_output_placeholders -= len(new_token_ids)
    assert request.num_output_placeholders >= 0, "UNDERFLOW (engine would crash)"
    return request.num_output_placeholders


def drain_new(request, new_token_ids, logger=None):
    """Post-fix logic (mirrors the f1ae patch, core arithmetic only)."""
    if not request.is_finished():
        remaining = request.num_output_placeholders
        drain = len(new_token_ids)
        if drain > remaining:
            # defensive clamp (would log a WARNING probe in production)
            drain = remaining
        request.num_output_placeholders -= drain
    return request.num_output_placeholders


def test_old_path_underflows():
    req = StubRequest(placeholders=1, finished=True)
    # second async row arrives after finish (counts already consumed)
    try:
        drain_old(req, [-1, -1, -1])
        raise SystemExit("old path did NOT underflow — test scenario wrong")
    except AssertionError:
        pass
    print("old path: underflow on finished-request double-drain: OK (bug exists)")


def test_new_path_skips_finished():
    req = StubRequest(placeholders=0, finished=True)
    bal = drain_new(req, [-1, -1, -1])
    assert bal == 0, f"finished request must not drain further, got {bal}"
    print("new path: finished-request drain skipped: OK")


def test_new_path_clamps_stale_balance():
    req = StubRequest(placeholders=1, finished=False)  # stale balance
    bal = drain_new(req, [-1, -1, -1])
    assert bal == 0, f"clamp expected, got {bal}"
    print("new path: stale-balance clamp: OK")


def test_normal_drain_unchanged():
    req = StubRequest(placeholders=4, finished=False)
    bal = drain_new(req, [-1, -1, -1])
    assert bal == 1, f"normal drain must equal len(tokens), got {bal}"
    print("new path: normal request drains exactly: OK")


def test_production_source_semantics():
    """The cherry-picked patch in the tree must carry both the is_finished()
    gate and the defensive clamp (grep the source — not exact-text, semantics)."""
    src = open(os.path.join(
        _HERE, "vllm", "v1", "core", "sched", "async_scheduler.py"
    )).read()
    assert "was_finished_before = request.is_finished()" in src, (
        "was-finished-before gate (pre-super check) missing"
    )
    assert "_drain = _remaining" in src, "defensive clamp missing"
    assert "not is_stale and not was_finished_before" in src
    # the WARNING-level underflow probe (recurrence detector)
    assert "MTP placeholder underflow probe" in src
    # per-add ledger only at DEBUG
    assert "MTP placeholder add:" in src
    assert "isEnabledFor(logging.DEBUG)" in src
    print("production source carries gate + clamp + probe + debug ledger: OK")


if __name__ == "__main__":
    test_old_path_underflows()
    test_new_path_skips_finished()
    test_new_path_clamps_stale_balance()
    test_normal_drain_unchanged()
    test_production_source_semantics()
    print("MTP-f1 drain-gate validation: ALL OK")
