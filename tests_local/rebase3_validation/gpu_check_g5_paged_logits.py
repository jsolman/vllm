"""GPU companion to test_g5_mqa_logits.py — post-build numerics check.

Compares fp8_paged_mqa_logits_triton against an fp32 reference on the real
GPU (the triton 3.7.1 CPU interpreter cannot execute bf16 tl.dot).
Run on any CUDA node after the build:
  vllm env: python tests_local/rebase3_validation/gpu_check_g5_paged_logits.py
"""
import os
import sys

_HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _HERE)

import torch  # noqa: E402

BS = 16
D = 128
H = 8
W = D + 4


def main():
    assert torch.cuda.is_available(), "run on a CUDA node"
    from vllm.v1.attention.ops.mqa_logits_triton import (
        fp8_paged_mqa_logits_triton,
    )

    dev = torch.device("cuda")
    for seed in (3, 11, 29):
        torch.manual_seed(seed)
        next_n_seq = (1, 2)
        for next_n in next_n_seq:
            B = 4
            ctxs = torch.randint(1, 40, (B,), device=dev) + (next_n - 1) * 0
            max_ctx = int(ctxs.max())
            bps = (max_ctx + BS - 1) // BS
            block_table = torch.stack(
                [
                    torch.arange(
                        b * bps + 7, b * bps + 7 + bps, device=dev
                    )
                    for b in range(B)
                ]
            ).to(torch.int32)
            NB = int(block_table.max()) + 1
            cache = torch.zeros(NB, BS, W, dtype=torch.uint8, device=dev)
            flat = cache.view(NB, BS * W)
            k_fp8 = torch.randn(NB, BS, D, device=dev).to(torch.float8_e4m3fn)
            scales = torch.empty(NB, BS, device=dev).uniform_(0.5, 2.0)
            flat[:, : BS * D] = k_fp8.view(torch.uint8).reshape(NB, BS * D)
            flat[:, BS * D :].view(torch.float32).reshape(NB, BS).copy_(scales)

            q_fp8 = torch.randn(B, next_n, H, D, device=dev).to(torch.float8_e4m3fn)
            weights = torch.rand(B * next_n, H, device=dev) + 0.1

            logits = fp8_paged_mqa_logits_triton(
                q_fp8,
                cache.unsqueeze(2),
                weights,
                ctxs,
                block_table,
                max_model_len=max_ctx,
                clean_logits=True,
            )

            q_f = q_fp8.float()
            worst = 0.0
            for b in range(B):
                for j in range(next_n):
                    ctx = int(ctxs[b])
                    ctx_j = ctx - next_n + j + 1
                    if ctx_j <= 0:
                        continue
                    for n in range(ctx_j):
                        blk = int(block_table[b, n // BS])
                        off = n % BS
                        s = (
                            q_f[b, j].float() * k_fp8[blk, off].float()
                        ).sum(-1) * float(scales[blk, off])
                        ref = (torch.relu(s) * weights[b * next_n + j]).sum()
                        g = logits[b * next_n + j, n].item()
                        worst = max(worst, abs(g - ref.item()))
            # bf16 pre-decode: relative error ~2^-8 on 128-dim dots
            assert worst < 1.0, f"seed {seed} next_n {next_n}: max diff {worst}"
            print(f"seed {seed} next_n {next_n}: OK (worst |diff| {worst:.4f})")
    print("G5 GPU paged-logits numerics: ALL OK")


if __name__ == "__main__":
    main()
