# Thor SM110 Carried Patches (glm52-plus-dsv4-thor-sm110)

This branch carries the patch stack needed to stably serve GLM-5.2 (+Vision) and
DeepSeek V4 on Jetson Thor (SM110, aarch64, 4-node cluster). It tracks upstream
vLLM `main` and is maintained with a linear rebase workflow.

## Table of contents

- [Branch layout](#branch-layout)
- [Vendored upstream PRs (drop when merged)](#vendored-upstream-prs-drop-when-merged)
- [Carried fixes by area](#carried-fixes-by-area)
- [Debug branch](#debug-branch)
- [Maintenance](#maintenance)

## Branch layout

Linear stack on top of upstream `main`:

```
fix: shm_broadcast copies out-of-band buffers at enqueue   <- CARRY-TEMP (upstream #53217)
fix: disable PDL launch attribute on SM110 (4 commits)     <- platform hardening
fix/test: scheduler + routing fixes (4 commits)            <- DSV4/GLM serving stability
fix: DSV4 decode corruption + SM110 kernel gates (13)      <- core SM110 enablement
feat: GLM-5.2-Vision port (4 commits)                      <- vision model
port: GLM-5.2 AGX Thor + FP8/DCP/MQA kernels (5 commits)   <- platform foundation
vendor: 4 upstream PRs (below)                             <- DSV4 parser/tokenizer fixes
```

## Vendored upstream PRs (drop when merged)

These four upstream vLLM PRs fix DSV4 serving bugs but are **not yet merged** and
not maintained by their submitters. They are vendored as single commits with the
original author preserved; each is dropped on the first rebase after it merges
upstream (monitored via the vendor re-check).

| PR | Subject | Author | Bundled commits | Status |
|----|--------|--------|-----------------|--------|
| [#51856](https://github.com/vllm-project/vllm/pull/51856) | Attach request-level tools to existing system message in DSV4 renderer | thegoldenflow | 1 | OPEN |
| [#50684](https://github.com/vllm-project/vllm/pull/50684) | DSV4 reasoning_effort "high" + message-level tools preservation | prasanna-gyde | 2 | OPEN |
| [#51262](https://github.com/vllm-project/vllm/pull/51262) | Handle trailing system messages in DSV4 prompt rendering | jiahaoliang | 1 | OPEN |
| [#52645](https://github.com/vllm-project/vllm/pull/52645) | Recover malformed DSV4 DSML wrappers (parser series) | jinbagi | 5 | OPEN |

Drop-when-merged check (run at each rebase or via cron):

```bash
for n in 51856 50684 51262 52645 53217; do
  gh pr view $n --repo vllm-project/vllm --json number,state,mergedAt,title \
    --jq '"\(.number) \(.state) \(.mergedAt // "unmerged") \(.title)"'
done
```

When a PR shows `state=MERGED`, drop its vendor commit on the next
`git rebase --onto upstream/main` (the upstream version supersedes it).

## Carried fixes by area

### Stability (CARRY-TEMP)
- **shm_broadcast zero-copy send race** (`a68528c57d` → upstream [#53217]): ZMQ
  `send_multipart(copy=False)` raced post-enqueue tensor mutation; remote workers
  got truncated pickles, crashed, and ranks hung in collectives. Fixed by copying
  out-of-band buffers at enqueue. **Drop when #53217 merges.**

### Platform hardening (SM110)
- PDL (programmatic dependent launch) disabled on SM110 across python platform
  gate, all csrc launch sites, dsv3 router GEMM, fp8 per-token-group quant.
  SM110 has only 20 SMs; PDL was observed hanging kernels
  (`cudaGridDependencySynchronize` spin at 98% SM / idle power).

### DSV4 / GLM scheduler + routing
- scheduled-spec-slot accounting on empty output rows
- non-spec slot refund (extends upstream #47928)
- sigmoid grouped-routing metadata preservation
- test alignment for vendored PR-51856 behavior change

### DSV4-on-SM110 core
- Triton sparse MLA backend for SM110 (`ampere_sparse`)
- MHC tilelang DeepGEMM fallback + n_splits fix
- DSV4 decode corruption fix (multi-stream execution on non-DeepGemm hosts)
- ll_bf16 / cutedsl gating, indexer platform checks, cutlass fp8 e8m0 upcast,
  sparse_swa metadata builder + FlashMLA tile-scheduler skip, gpt_oss Triton
  MoE device gate
- sparse-MLA attention impl for XPUMLASparse metadata
- hybrid expert weight remap, nvfp4_aqlm_hybrid / tp_hybrid_moe restoration

### GLM-5.2-Vision
- GLM-5.2-Vision model (MoonViT + K2VL projector)
- glm5v quant-config fix, config attr delegation, safetensors.index loader,
  cute.experimental ImportError stub

## Excluded from this branch

- 6 debug/instrumentation commits (MoEDBG ×2 + reverses, env-gated DSV4
  instrumentation, DSV4_LOG_BODIES middleware) — preserved on
  `debug/thor-sm110-diagnostics`.
- `Dockerfile.glm52-sm120` — SM120 desktop build artifact, not source; not
  applicable to the SM110 cluster target.
- The 4 upstream PR *merges* themselves (their content is vendored above).

## Debug branch

`debug/thor-sm110-diagnostics` = `glm52-plus-dsv4-thor-sm110` + the 6 excluded
debug commits, kept for instrumenting the same areas again (MoE op timing,
request-body role structure, DSV4 SM110 internals).

## Maintenance

See `PLAN-2026-08-dsv4-thor-sm110.md` (repo root of `~/vllm-upstream`) for the
full procedure. TL;DR: weekly `git fetch upstream main` +
`git rebase --onto upstream/main <tagged-base>`, renovate conflicts onto the new
upstream shape (never merge-main), re-run py_compile + `import vllm` smoke +
DSV4 tokenizer/parser tests, force-push with `--force-with-lease`. Deployment
tree re-sync stays manual (cluster kill/build/launch is human-triggered).
