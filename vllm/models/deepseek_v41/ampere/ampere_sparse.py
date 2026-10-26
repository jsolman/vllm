# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Ampere (SM8x) and Thor (SM110) sparse MLA for DeepSeek V4.1.

Reuses the ROCm Triton sparse-MLA implementation wholesale: its kernels,
metadata builders, and dequantized o_proj reference path are plain
Triton/torch (the aiter-only dispatches self-disable off ROCm).

On SM8x (A100/RTX 3080), Triton refuses native fp8e4nv converts below
SM89, so the ROCm kernels' software fp8 paths supply bit-exact
equivalents.

On SM110 (Thor/Blackwell), native fp8e4nv converts are available, so
the same Triton kernels run on the fast hardware path.

The CUDA selector declines every native V4.1 backend here (FlashMLA
mega/sparse need SM90+/SM100, FlashInfer is SM120), so this module is
the portable fallback for SM8x and SM110.
"""

from vllm.models.deepseek_v41.amd.rocm import (
    DeepseekV41ROCMAiterMLAAttention,
    DeepseekV41ROCMAiterMLASparseBackend,
    DeepseekV41ROCMAiterSparseSWABackend,
)
from vllm.models.deepseek_v41.ampere.woa_dequant import (
    DequantizedBf16LinearMethod,
)
from vllm.platforms.interface import DeviceCapability

class DeepseekV41AmpereMLASparseBackend(
    DeepseekV41ROCMAiterMLASparseBackend
):
    @staticmethod
    def get_name() -> str:
        return "TRITON_MLA_SPARSE_DSV41"

    @classmethod
    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:
        # SM8x (Ampere) and SM110 (Thor/Blackwell consumer).
        return capability.major in [8, 11]


class DeepseekV41AmpereSparseSWABackend(
    DeepseekV41ROCMAiterSparseSWABackend
):
    @staticmethod
    def get_name() -> str:
        return "DEEPSEEK_SPARSE_SWA"


class DeepseekV41AmpereMLAAttention(
    DeepseekV41ROCMAiterMLAAttention
):
    """SM8x/SM110 DeepSeek V4.1 attention: ROCm Triton path on CUDA."""

    backend_cls = DeepseekV41AmpereMLASparseBackend
    swa_backend_cls = DeepseekV41AmpereSparseSWABackend

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # The einsum/dequant o_proj path reads wo_a's retained FP8 block
        # scale, which the Marlin repack on SM8x/SM110 destroys. Dequantize
        # wo_a to BF16 at load instead (this runs before post-load
        # processing); the helper's BF16 branch consumes it directly.
        # wo_b keeps its quantized runtime (plain linear apply).
        if getattr(self.wo_a, "weight_block_size", None):
            self.wo_a.quant_method = DequantizedBf16LinearMethod()
