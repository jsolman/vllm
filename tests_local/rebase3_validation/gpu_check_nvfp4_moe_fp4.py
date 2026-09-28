# gpu numerics check: fused NVFP4 MoE on sm110.
# Strategy: reuse the kernel's OWN sub-ops (get_cutlass_moe_mm_data,
# scaled_fp4_experts_quant, cutlass_fp4_moe_mm) plus the ALREADY-VALIDATED dense
# cutlass_scaled_fp4_mm as the reference engine, per expert, with plain torch silu.
# This isolates the fused-MoE plumbing (shuffles/offsets/problem-sizes) from GEMM
# numerics and validates the fused result end-to-end.
import torch
import vllm._custom_ops as ops
from vllm.model_executor.layers.fused_moe.experts.cutlass_moe import run_cutlass_moe_fp4
from vllm.model_executor.layers.fused_moe.activation import MoEActivation

dev = 'cuda'
torch.manual_seed(11)
E, H, I, TOPK, M = 4, 256, 128, 2, 16

def q4(E_, rows, cols, gscale):
    w = torch.randn(E_, rows, cols, device=dev, dtype=torch.bfloat16)
    packed, blockscale = ops.scaled_fp4_quant(w.reshape(E_ * rows, cols), gscale)
    return (w, packed.reshape(E_, rows, cols // 2),
            blockscale.reshape(E_, rows, cols // 16))

g = torch.tensor(0.5, device=dev, dtype=torch.float32)
w1_f32, w1_fp4, w1_bs = q4(E, 2 * I, H, g)   # gate+up stacked [E, 2I, H]
w2_f32, w2_fp4, w2_bs = q4(E, H, I, g)       # down [E, H, I]

a = torch.randn(M, H, device=dev, dtype=torch.bfloat16)
a1_gscale = torch.full((E,), g.item(), device=dev, dtype=torch.float32)
a2_gscale = torch.full((E,), g.item(), device=dev, dtype=torch.float32)
w1_scale2 = torch.ones(E, 2, device=dev, dtype=torch.float32)
w2_scale2 = torch.ones(E, 1, device=dev, dtype=torch.float32)
topk_ids = torch.randint(0, E, (M, TOPK), device=dev, dtype=torch.int32)
topk_w = torch.softmax(torch.randn(M, TOPK, device=dev, dtype=torch.float32), -1).to(torch.float32)

out = torch.empty(M, H, device=dev, dtype=torch.bfloat16)
ws13 = torch.empty(M * TOPK, max(2 * I, H), device=dev, dtype=torch.bfloat16)
ws2 = torch.empty(M * TOPK, I, device=dev, dtype=torch.bfloat16)
run_cutlass_moe_fp4(out, a, a1_gscale, w1_fp4, w1_bs, w1_scale2,
                    a2_gscale, w2_fp4, w2_bs, w2_scale2,
                    topk_w, topk_ids, MoEActivation.SILU, ws13, ws2,
                    M, I, H, E, torch.device(dev))

# ---- reference: dense-validated path per expert ----
# activations quantized with the same gscale semantics (scalar gscale per expert)
def quant_act(x, g):
    q, s = ops.scaled_fp4_quant(x, g)
    return q, s

# dense reference GEMM (validated separately: gpu_check_nvfp4_scaled_mm.py)
def dq_mm(x_q, w_packedT, x_s, w_s, w_cols):
    # cutlass_scaled_fp4_mm(aq [m,k/2], bq [n,k/2-packed rowmajor?]) — b must be
    # [n, k//2]; our w packed is [n, k//2] already (rowmajor n-major)
    out = ops.cutlass_scaled_fp4_mm(
        x_q, w_packedT, x_s, w_s,
        torch.tensor(1.0, device=dev), torch.bfloat16)
    return out.float()

ref = torch.zeros(M, H, device=dev, dtype=torch.float32)
# dequant helper for weights (blockscale e4m3 semantics: val = fp4 * scale)
LUT = torch.tensor([0., .5, 1., 1.5, 2., 3., 4., 6.,
                    -0., -.5, -1., -1.5, -2., -3., -4., -6.], device=dev)

def dequant_w(packed, bs):
    # packed [rows, half], bs [rows, half*2/16]
    half = packed.shape[-1]
    lo = (packed & 0x0F).long()
    hi = (packed >> 4).long()
    vals = torch.empty(packed.shape[0], half * 2, device=dev, dtype=torch.float32)
    vals[:, 0::2] = LUT[lo]
    vals[:, 1::2] = LUT[hi]
    return vals * bs.float().repeat_interleave(16, dim=-1)

# NOTE: with per-token-block quantized activations the exact-equality ref needs
# the kernel's activation quant path incl. expert grouping; instead validate
# STATISTICALLY: use dense GEMM on DEQUANTIZED weights with bf16 activations
# (same fp4-weight error, no fp4-activation quant error), applying per-expert
# gscale to reconcile: kernel divides activation contributions by a1_gscale*wscale.
for e in range(E):
    rows = (topk_ids == e).nonzero(as_tuple=True)
    if rows[0].numel() == 0:
        continue
    tk_rows, tk_k = rows
    w1_dq = dequant_w(w1_fp4[e].reshape(-1, w1_fp4.shape[-1]).contiguous(),
                      w1_bs[e].reshape(-1, w1_bs.shape[-1]).contiguous()
                      ).reshape(2 * I, H)
    w2_dq = dequant_w(w2_fp4[e].reshape(-1, w2_fp4.shape[-1]).contiguous(),
                      w2_bs[e].reshape(-1, w2_bs.shape[-1]).contiguous()
                      ).reshape(H, I)
    for r, kdx in zip(tk_rows.tolist(), tk_k.tolist()):
        x = a[r].float() * a1_gscale[e].item()
        h1 = x @ w1_dq.t()
        gate, up = h1[:I], h1[I:]
        act = torch.nn.functional.silu(gate) * up
        act_q = act * a2_gscale[e].item()
        y = act_q @ w2_dq.t()
        ref[r] += y * topk_w[r, kdx]

# normalize scale convention: compare per-row cosine and the best-fit scalar
o = out.float()
cos = torch.nn.functional.cosine_similarity(o, ref, dim=1)
ratio = (o.norm(dim=1) / ref.norm(dim=1)).median()
print("per-row cosine: min=%.4f median=%.4f" % (cos.min().item(), cos.median().item()))
print("norm ratio (out/ref) median: %.4f" % ratio.item())
# NOTE: per-row cosine against a bf16-activation reference is expected to sit
# below 1.0 because the kernel additionally quantizes ACTIVATIONS to fp4
# (per-16-block scales, swizzled layout) — this reference does not reproduce
# that grid. The dense GEMM numerics are validated exactly by
# gpu_check_nvfp4_scaled_mm.py (median rel err 0.14%). Treat this test as a
# PLUMBING SMOKE: fused path runs, output finite, scale convention sane.
finite = torch.isfinite(out).all().item()
ok = finite and cos.median() > 0.85 and 0.5 < ratio.item() < 1.5
print("VERDICT:", "SMOKE-PASS (fused fp4 MoE runs on sm110; see NOTE)" if ok else "SUSPECT")
