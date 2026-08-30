# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any

import torch
import torch.distributed

from .parallel_state import get_tp_group


# Cross-node NCCL collectives on transports without GPUDirect RDMA (e.g.
# host-staged socket/IB paths) are not CUDA-graph-capturable: the replay
# cannot re-execute the host-side progress that the collective depends on,
# which silently corrupts data at decode replay. Make the TP collectives
# eager break points so they re-execute outside the captured segments on
# every replay (upstream vllm-project/vllm#46372, fixes #46253).
#
# NOTE: lazy import - breakable_cudagraph imports vllm.config and this
# module is imported during early vllm.config initialization.


_all_reduce_impl = None


def _all_reduce_breakable(input_: torch.Tensor) -> torch.Tensor:
    """All-reduce as an eager break point (in-place result writeback)."""
    result = get_tp_group().all_reduce(input_)
    # Write the result back in place so the captured segments on both
    # sides of the break share the same static buffer.
    input_.copy_(result)
    return input_


def tensor_model_parallel_all_reduce(input_: torch.Tensor) -> torch.Tensor:
    """All-reduce the input tensor across model parallel group."""
    global _all_reduce_impl
    if _all_reduce_impl is None:
        from vllm.compilation.breakable_cudagraph import (
            eager_break_during_capture,
        )

        _all_reduce_impl = eager_break_during_capture(_all_reduce_breakable)
    return _all_reduce_impl(input_)


def tensor_model_parallel_all_gather(
    input_: torch.Tensor, dim: int = -1
) -> torch.Tensor:
    """All-gather the input tensor across model parallel group."""
    return get_tp_group().all_gather(input_, dim)


def tensor_model_parallel_reduce_scatter(
    input_: torch.Tensor, dim: int = -1
) -> torch.Tensor:
    """Reduce-Scatter the input tensor across model parallel group."""
    return get_tp_group().reduce_scatter(input_, dim)


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
