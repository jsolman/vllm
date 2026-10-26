"""WoA dequant helper for the DeepSeek V4.1 CUDA (SM8x/SM110) path.

The ROCm-derived attention runs wo_a through a dequantized-BF16 einsum
(``_get_cached_wo_a_bf16``), which reads the layer's retained block scale.
On SM110 the FP8-block linear resolves to the Marlin kernel, whose
post-load repack permutes ``weight_scale`` into an opaque 1-D blob the
helper cannot interpret (view crash: 'shape [2, -1, 262144] is invalid').

Fix: swap wo_a/wo_b to a load-time-dequantize method BEFORE
``process_weights_after_loading`` runs — the weights stay plain BF16 at
runtime, matching what the einsum path wants anyway. Two small
per-layer GEMMs; the decode bottleneck is MoE weight streaming.
"""

import torch
from torch.nn import Parameter

from vllm.model_executor.layers.linear import LinearMethodBase


class DequantizedBf16LinearMethod(LinearMethodBase):
    """Convert an FP8-block-quantized weight to BF16 at load time."""

    def create_weights(
        self, layer, input_size_per_partition, output_partition_sizes,
        input_size, output_size, params_dtype, **extra_weight_attrs,
    ):
        raise NotImplementedError(
            "DequantizedBf16LinearMethod wraps an already-created "
            "FP8-block layer; it must not create weights."
        )

    def process_weights_after_loading(self, layer) -> None:
        weight = layer.weight.data
        scale = getattr(layer, "weight_scale_inv", None)
        if scale is None:
            scale = getattr(layer, "weight_scale", None)
        assert scale is not None, "FP8-block layer lost its weight scale"
        block = getattr(layer, "weight_block_size", None) or [32, 32]
        block_n, block_k = block
        n, k = weight.shape
        assert scale.numel() == (n // block_n) * (k // block_k), (
            f"unexpected scale numel {scale.numel()} for weight {tuple(weight.shape)} "
            f"with blocks {block}"
        )
        scale = scale.reshape(n // block_n, k // block_k).to(torch.float32)
        expanded = (
            scale.repeat_interleave(block_n, dim=0)[:, :n]
            .T.repeat_interleave(block_k, dim=0)[:, :k]
            .T
        )
        dequant = weight.to(torch.float32) * expanded
        layer.weight = Parameter(
            dequant.to(layer.params_dtype).contiguous(), requires_grad=False
        )
        layer.weight_scale_inv = None
        layer.weight_scale = None
        layer.weight_block_size = None

    def apply(self, layer, x: torch.Tensor, bias=None) -> torch.Tensor:
        return torch.nn.functional.linear(x, layer.weight, bias)
