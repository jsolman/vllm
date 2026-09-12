# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek V4 sparse MLA attention for SM8x (Ampere) and SM110 (Thor).

Reuses the ROCm Triton sparse-MLA implementation wholesale: its kernels,
ragged metadata builders, and bf16 o_proj reference path are plain
Triton/torch (the aiter-only preshuffle GEMMs self-disable off ROCm).

On SM8x (A100/RTX 3080), Triton refuses native fp8e4nv converts below
SM89, so the ROCm kernels' software fp8 paths supply bit-exact
equivalents.

On SM110 (Thor/Blackwell), native fp8e4nv converts are available, so
the same Triton kernels run on the fast hardware path.

The base ``DeepseekV4ROCMAiterMLAAttention`` is platform-neutral once
its fp8 conversions are abstracted; only the aiter dispatches and gfx9
tuning are ROCm-specific and self-disable on CUDA.
"""

from vllm.models.deepseek_v4.amd.rocm import (
    DeepseekV4ROCMAiterMLAAttention,
    DeepseekV4ROCMAiterMLASparseBackend,
)
from vllm.platforms.interface import DeviceCapability


class DeepseekV4AmpereMLASparseBackend(DeepseekV4ROCMAiterMLASparseBackend):
    @staticmethod
    def get_name() -> str:
        return "TRITON_MLA_SPARSE_DSV4"

    @classmethod
    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:
        # SM8x (Ampere: A100 SM80, RTX 3080 SM86) and SM110 (Thor/Blackwell).
        # The parent restricts to Hopper/Blackwell (9/10); this backend uses
        # the portable Triton sparse-MLA kernels instead of FlashMLA/cutedsl.
        return capability.major in [8, 11]


class DeepseekV4AmpereMLAAttention(DeepseekV4ROCMAiterMLAAttention):
    """SM8x/SM110 DeepSeek V4 attention: ROCm Triton path on CUDA."""

    backend_cls = DeepseekV4AmpereMLASparseBackend
