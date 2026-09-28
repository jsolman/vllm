# gpu numerics check: NVFP4 — cutlass_scaled_fp4_mm on sm110 vs dequant reference
import torch
import vllm._custom_ops as ops

dev = 'cuda'
torch.manual_seed(7)

m, k, n = 128, 4096, 256     # realistic tile: hidden 4096
a = torch.randn(m, k, device=dev, dtype=torch.bfloat16)
b_f32 = torch.randn(n, k, device=dev, dtype=torch.bfloat16)

# sane global scales (typical: amax/6 for fp4)
a_g = (a.float().abs().amax() / 6.0).clamp_min(1e-8)
b_g = (b_f32.float().abs().amax() / 6.0).clamp_min(1e-8)
aq, ascale = ops.scaled_fp4_quant(a, a_g)
bq, bscale = ops.scaled_fp4_quant(b_f32, b_g)

out = ops.cutlass_scaled_fp4_mm(
    aq, bq, ascale, bscale, torch.tensor(1.0, device=dev), torch.bfloat16)

# reference path: vllm's own reference quantizer, then dequant by LUT and GEMM in fp32
from vllm.model_executor.layers.quantization.utils.nvfp4_emulation_utils import (
    ref_nvfp4_quant,
)
a_ref_q, a_ref_s = ref_nvfp4_quant(a.float(), a_g, 16)
b_ref_q, b_ref_s = ref_nvfp4_quant(b_f32.float(), b_g, 16)

LUT = torch.tensor([0., .5, 1., 1.5, 2., 3., 4., 6.,
                    -0., -.5, -1., -1.5, -2., -3., -4., -6.], device=dev)

def unpack(q, s):
    # ref_nvfp4_quant returns unpacked float32 values in [-6,6] —  use directly
    return q * s.float().repeat_interleave(16, dim=1)

a_deq = unpack(a_ref_q, a_ref_s)
b_deq = unpack(b_ref_q, b_ref_s)
# The kernel consumes e4m3 block scales already divided by the global scale
# (scaled_fp4_quant bakes 1/g into them), so no extra a_g*b_g factor here.
ref = (a_deq @ b_deq.t())

diff = (out.float() - ref).abs()
rel = (diff / ref.abs().clamp_min(1e-3))
print(f"out absmean={out.float().abs().mean():.4f} ref absmean={ref.abs().mean():.4f}")
print(f"MAE={diff.mean():.4f}  median_rel={rel.median():.4%}  "
      f"p99_rel={rel.flatten().kthvalue(int(rel.numel()*0.99)).values:.4%}")
print("VERDICT:", "PASS" if rel.median() < 0.02 else "SUSPECT")
