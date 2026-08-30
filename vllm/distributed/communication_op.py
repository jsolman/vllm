# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any

import torch
import torch.distributed

from vllm.compilation.breakable_cudagraph import eager_break_during_capture

from .parallel_state import get_tp_group


# Cross-node NCCL collectives on transports without GPUDirect RDMA (e.g.
# host-staged socket/IB paths) are not CUDA-graph-capturable: the replay
# cannot re-execute the host-side progress that the collective depends on,
# which silently corrupts data at decode replay. Make the TP collectives
# eager break points so they re-execute outside the captured segments on
# every replay (upstream vllm-project/vllm#46372, fixes #46253).
@eager_break_during_capture
def tensor_model_parallel_all_reduce(input_: torch.Tensor) -> torch.Tensor:
    """All-reduce the input tensor across model parallel group."""
    result = get_tp_group().all_reduce(input_)
    # Write the result back in place so the captured segments on both
    # sides of the break share the same static buffer.
    input_.copy_(result)
    return input_


@eager_break_during_capture
def tensor_model_parallel_all_gather(
    input_: torch.Tensor, dim: int = -1
) -> torch.Tensor:
    """All-gather the input tensor across model parallel group."""
    result = get_tp_group().all_gather(input_, dim)
    return result


@eager_break_during_capture
def tensor_model_parallel_reduce_scatter(
    input_: torch.Tensor, dim: int = -1
) -> torch.Tensor:
    """Reduce-Scatter the input tensor across model parallel group."""
    result = get_tp_group().reduce_scatter(input_, dim)
    return result


def tensor_model_parallel_gather(
    input_: torch.Tensor, dst: int = 0, dim: int = -1
) -> torch.Tensor | None:
    """Gather the input tensor across model parallel group."""
    return get_tp_group().gather(input_, dst, dim)


def broadcast_tensor_dict(
    tensor_dict: dict[Any, torch.Tensor | Any] | None = None, src: int = 0
):
    if not torch.distributed.is_initialized():
        return tensor_dict
    return get_tp_group().broadcast_tensor_dict(tensor_dict, src)
