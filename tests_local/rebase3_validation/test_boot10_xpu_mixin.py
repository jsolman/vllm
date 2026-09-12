"""Boot10 (4d005f9e2e) validation: XPUMLASparseImpl mixin + hook.

This is an XPU-path class (not exercised on our CUDA/Thor serving), so the
CPU-provable layers are structural:
1. XPUMLASparseImpl inherits SharedTopkIndicesBuffer and provides
   init_topk_indices_buffer + record_logical_topk_ready (the hook the
   SparseMLA glue calls; missing hook = AttributeError at first sparse-MLA
   forward on any platform that routes through the common path).
2. AST-safe: file parses; the hook signature is () -> None.
(The class-level regression this squash fixed: upstream refactor dropped the
mixin from the bases during an earlier rebase — an import-time-silent loss
that only surfaces at runtime; enforced here structurally.)
"""
import ast as ast_mod
import os
import sys

_HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _HERE)

_FILE = os.path.join(
    _HERE, "vllm", "v1", "attention", "backends", "mla", "xpu_mla_sparse.py"
)


def test_structure():
    src = open(_FILE).read()
    tree = ast_mod.parse(src)

    cls = None
    for node in ast_mod.walk(tree):
        if isinstance(node, ast_mod.ClassDef) and node.name == "XPUMLASparseImpl":
            cls = node
            break
    assert cls is not None, "XPUMLASparseImpl class missing"
    bases = {
        (b.attr if isinstance(b, ast_mod.Attribute) else
         (b.value.attr if isinstance(b, ast_mod.Attribute) else getattr(b, "id", b)))
        for b in cls.bases
    } if False else {
        getattr(b, "attr", None) or getattr(b, "id", str(b)) for b in cls.bases
    }
    assert any("SharedTopkIndicesBuffer" in str(b) for b in bases), (
        f"mixin lost from bases: {bases}"
    )
    methods = {n.name for n in cls.body if isinstance(n, ast_mod.FunctionDef)}
    # init_topk_indices_buffer lives on SharedTopkIndicesBuffer (sparse_mla_
    # attention.py) — the mixin mix-in is what supplies it; only the CALL must
    # exist in __init__ plus our no-op hook override.
    assert "record_logical_topk_ready" in methods, "hook missing"
    assert any(
        isinstance(n, ast_mod.Expr) and "init_topk_indices_buffer" in ast_mod.dump(n)
        for n in cls.body
    ) or any(
        isinstance(n, ast_mod.FunctionDef)
        and "init_topk_indices_buffer" in ast_mod.dump(n)
        for n in cls.body
    ), "init_topk_indices_buffer call missing from __init__"
    for n in cls.body:
        if isinstance(n, ast_mod.FunctionDef) and n.name == "record_logical_topk_ready":
            args = [a.arg for a in n.args.args]
            assert args == ["self"], f"hook signature changed: {args}"
            assert n.returns is not None and "None" in ast_mod.dump(n.returns)
    print("class bases + init_topk_indices_buffer + no-op hook: OK")


def test_full_parse():
    ast_mod.parse(open(_FILE).read())
    print("module parses: OK")


if __name__ == "__main__":
    test_full_parse()
    test_structure()
    print("Boot10 mixin/hook validation: ALL OK")
