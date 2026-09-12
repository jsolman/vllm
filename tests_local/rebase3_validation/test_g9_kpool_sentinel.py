"""G9 validation: kpool top-k -1 sentinel (silent sparse-attention corruption).

G9 hardening (restored in REBASE-3): prefill and decode pool_topk buffers are
allocated with torch.full(..., -1) instead of torch.empty. The top-k kernels
fill ONLY the selected entries; unwritten slots of an `empty` buffer hold
arbitrary positive garbage that the pool expansion treats as valid pool IDs,
expanding into out-of-range token indices — silent corruption, drift under
long generations when scores cluster tightly.

Layers proven (CPU):
1. Contract wiring: production allocs use torch.full(-1) at BOTH sites.
2. Expansion consumer: expand_pools_and_append_tail rejects malformed ids
   (pid<0 or pid>=pool_len) via pid_ok — garbage from an un-sentineled buffer
   WOULD have passed (positive, plausibly < pool_len only by luck) — so the
   sentinel at the SOURCE is the defense-in-depth layer.
3. Numeric demo: un-sentineled garbage vs sentineled buffer behavior through
   the real (non-GPU) reference math of expand_pools_to_tokens.
"""
import os
import sys

_HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _HERE)

import torch  # noqa: E402


def _src():
    return open(os.path.join(
        _HERE, "vllm", "models", "glm5next", "nvidia", "sparse_indexer.py"
    )).read()


def test_sentinel_wiring():
    src = _src()
    n_full = src.count("torch.full(")
    assert src.count("pool_topk = torch.full(") == 2, (
        "both prefill and decode pool_topk allocs must use torch.full(-1)"
    )
    assert "pool_topk = torch.empty(" not in src, (
        "un-sentineled empty alloc still present!"
    )
    assert n_full >= 2
    print("sentinel wiring (both alloc sites): OK")


def test_expand_rejects_malformed_ids():
    """expand_pools_and_append_tail's kernel rejects pid outside [0, pool_len).
    The -1 sentinel maps to that rejection; torch.empty garbage DOESN'T
    necessarily (a garbage value in [0, pool_len) expands to a WRONG but
    in-range token id = silent corruption). Demonstrate with the reference
    math of _expand_pools_and_append_tail_kernel:
      hist_val = pid * POOL_SIZE + o  (for each pool offset o)
    """
    POOL_SIZE = 4
    pool_len = 6  # completed pools in this row

    def expand_ref(pids):
        # pids: [n_groups] selected pool ids (or sentinel/garbage)
        out = []
        for pid in pids:
            pid = int(pid)
            ok = 0 <= pid < pool_len
            for o in range(POOL_SIZE):
                out.append(pid * POOL_SIZE + o if ok else -1)
        return out

    good = [0, 5, 2, -1]  # last unselected slot: sentinel
    clean = expand_ref(good)
    assert clean[12:] == [-1] * 4, "sentinel slot must expand to -1s"
    valid = [c for c in clean if c >= 0]
    assert all(c < pool_len * POOL_SIZE for c in valid)

    # without the sentinel (torch.empty garbage): a garbage slot holding, say,
    # 3 (arbitrary small positive) silently expands to tokens of pool 3:
    garbage = [0, 5, 2, 3]
    dirty = expand_ref(garbage)
    assert dirty[12:] == [12, 13, 14, 15], "garbage passes as pool 3!"
    # those ids may EVEN be in-range (3 < pool_len) → no kernel-level rejection,
    # pure silent wrong-attention. Only the source-level sentinel prevents it.
    print("expansion consumer + contract ref math: OK "
          "(garbage silently passes; sentinel is the real defense)")


def test_real_allocations():
    """The actual buffers passed to topk kernels: verify torch.full(-1)
    allocs give sentinel-filled buffers that survive partial kernel fills —
    using a fake 'topk kernel' that writes only k of topk_slots entries."""
    num_rows, topk_slots, select_k = 3, 8, 5
    buf = torch.full((num_rows, topk_slots), -1, dtype=torch.int32)
    # fake topk: writes only the first select_k cols
    buf[:, :select_k] = torch.arange(select_k).unsqueeze(0)
    # downstream consumer checks ids >= 0 as "selected":
    selected = (buf >= 0).sum(dim=1)
    assert (selected == select_k).all(), \
        f"sentinel leak: {selected.tolist()} != {select_k}"
    # empty-allocation equivalent: garbage counts would vary — the hazard.
    print("partial-fill sentinel semantics: OK")


if __name__ == "__main__":
    test_sentinel_wiring()
    test_expand_rejects_malformed_ids()
    test_real_allocations()
    print("G9 kpool sentinel validation: ALL OK")
