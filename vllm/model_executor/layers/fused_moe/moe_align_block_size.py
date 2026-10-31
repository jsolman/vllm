# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
from typing import Literal, overload

import torch

from vllm import _custom_ops as ops
from vllm.triton_utils import triton
from vllm.utils.math_utils import round_up

# Debug/fix switch: use the deterministic (stable-sorted) token-to-expert
# block assignment instead of the CUDA op. The CUDA op hands out the
# slots with an atomicAdd, so the order of the tokens inside an expert's
# block is not deterministic; the expert GEMM then accumulates them in
# that order, and fp32 addition is not associative, so the same request
# gave different results run to run.
_DETERMINISTIC_ALIGN = os.environ.get("VLLM_DSV41_DETERMINISTIC_ALIGN", "0") == "1"


def _deterministic_align(
    topk_ids: torch.Tensor,
    block_size: int,
    num_experts: int,
    sorted_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_pad: torch.Tensor,
    scatter_idx: torch.Tensor | None = None,
    expert_map: torch.Tensor | None = None,
    ignore_invalid_experts: bool = False,
) -> None:
    """The stable (expert_id, token_index) block assignment.

    Same layout as ``ops.moe_align_block_size``, but the tokens of each
    expert are placed in increasing token-index order, so the result does
    not depend on the order the atomic increments happened in.
    """
    numel = topk_ids.numel()
    device = topk_ids.device
    flat_expert = topk_ids.reshape(-1).to(torch.int64)
    flat_index = torch.arange(numel, device=device, dtype=torch.int64)

    # The CUDA op's get_local_expert_id applies the expert_map when the
    # caller passes it (which the expert-parallel callers do, with
    # ignore_invalid_experts=True), and the grouping is then over the
    # LOCAL expert ids. When the caller does NOT pass it, the grouping
    # is over the GLOBAL ids and the caller maps the block ids itself.
    # The helper must follow the same rule.
    if expert_map is not None:
        valid = flat_expert >= 0
        _mapped = expert_map[flat_expert.clamp(min=0)].to(torch.int64)
        local_expert = torch.where(valid, _mapped, torch.full_like(flat_expert, -1))
    else:
        valid = flat_expert >= 0
        local_expert = flat_expert

    # the count of tokens per expert, and the padded base of each expert
    # bincount needs non-negative values, and the invalid entries are -1
    _sel = local_expert[valid]
    _sel = _sel[_sel >= 0]
    counts = torch.bincount(_sel, minlength=num_experts)
    padded = ((counts + block_size - 1) // block_size) * block_size
    bases = torch.cumsum(padded, 0) - padded
    total = int(padded.sum().item())

    # a STABLE sort by (expert, index): the index is the tie-break, so the
    # tokens of an expert keep their relative order.
    order = torch.argsort(local_expert, stable=True)
    sorted_expert = local_expert[order]
    starts = torch.cumsum(counts, 0) - counts
    rank_in_expert = torch.empty(numel, dtype=torch.int64, device=device)
    _pos = torch.arange(numel, device=device, dtype=torch.int64)
    # the invalid entries (-1) sort first; their rank is meaningless but
    # they are never used (the slot is masked out below).
    _base_of = torch.where(sorted_expert >= 0,
                           starts[sorted_expert.clamp(min=0)],
                           torch.zeros_like(sorted_expert))
    rank_in_expert[order] = _pos - _base_of

    slot = torch.where(valid,
                       bases[local_expert.clamp(min=0)] + rank_in_expert,
                       torch.zeros_like(flat_index))
    # the caller's buffer; the tail is the padding
    n = min(total, sorted_ids.numel())
    # the WHOLE buffer is the padding, so fill all of it (the CUDA op
    # fills it with numel, the "this row does not exist" sentinel)
    sorted_ids.fill_(numel)
    # the slots and the token indices must be masked TOGETHER: taking
    # the first n of each separately would pair a slot with the wrong
    # token.
    _slots = slot[valid]
    _tokens = flat_index[valid]
    _m = _slots < n
    sorted_ids[:n].scatter_(0, _slots[_m].to(torch.int64),
                            _tokens[_m].to(torch.int32))

    # the expert id of every block
    num_blocks = expert_ids.numel()
    expert_ids.fill_(-1)
    _b = torch.arange(num_blocks, device=device, dtype=torch.int64)
    # the block of each expert, and the expert of each block
    _blk_of_expert = bases // block_size
    for e in range(num_experts):
        _b0 = int(_blk_of_expert[e].item())
        _nb = int((int(padded[e].item()) + block_size - 1) // block_size)
        if _nb > 0:
            expert_ids[_b0:min(_b0 + _nb, num_blocks)] = e
    # The block ids hold the LOCAL expert ids; the CALLER maps them
    # through expert_map, exactly as it does for the CUDA op.
    num_tokens_post_pad[0] = min(total, sorted_ids.numel())
    if scatter_idx is not None:
        scatter_idx.copy_(
            torch.where(valid, flat_index, torch.full_like(flat_index, -1))
            .to(torch.int32)
            .unsqueeze(1)
        )



@overload

def moe_align_block_size(
    topk_ids: torch.Tensor,
    block_size: int,
    num_experts: int,
    expert_map: torch.Tensor | None = None,
    pad_sorted_ids: bool = False,
    ignore_invalid_experts: bool = False,
    *,
    return_scatter_idx: Literal[False] = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]: ...


@overload
def moe_align_block_size(
    topk_ids: torch.Tensor,
    block_size: int,
    num_experts: int,
    expert_map: torch.Tensor | None = None,
    pad_sorted_ids: bool = False,
    ignore_invalid_experts: bool = False,
    *,
    return_scatter_idx: Literal[True],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: ...


@overload
def moe_align_block_size(
    topk_ids: torch.Tensor,
    block_size: int,
    num_experts: int,
    expert_map: torch.Tensor | None = None,
    pad_sorted_ids: bool = False,
    ignore_invalid_experts: bool = False,
    *,
    return_scatter_idx: bool,
) -> (
    tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    | tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
): ...


def moe_align_block_size(
    topk_ids: torch.Tensor,
    block_size: int,
    num_experts: int,
    expert_map: torch.Tensor | None = None,
    pad_sorted_ids: bool = False,
    ignore_invalid_experts: bool = False,
    return_scatter_idx: bool = False,
) -> (
    tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    | tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
):
    """Aligns the token distribution across experts to be compatible with block
    size for matrix multiplication.

    Note: In the case of expert_parallel, moe_align_block_size initially
    considers all experts as valid and aligns all tokens appropriately.
    Before the function returns it marks the experts_ids that are not in
    the current GPU rank as -1 so the MoE matmuls could skip those blocks.
    This requires the num_experts input arg to be the num global experts.

    Args:
        topk_ids: A tensor of shape [total_tokens, top_k] representing the
            top-k expert indices for each token.
        block_size: The block size used in block matrix multiplication.
        num_experts: The total number of experts.
        expert_map: A tensor of shape [num_experts] that maps the expert index
            from the global space to the local index space of the current
            expert parallel shard. If the expert is not in the current expert
            parallel shard, the mapping is set to -1.
        pad_sorted_ids: A flag indicating whether the sorted_token_ids length
            should be padded to a multiple of block_size,
        ignore_invalid_experts: A flag indicating whether to ignore invalid
            experts. When False, all expert_ids in topk_ids will participate in
            counting and ranking, but invalid experts in expert_ids will be marked
            as -1. When True, all invalid expert_ids in topk_ids will be ignored
            and will not participate in counting or ranking, and there will be no
            -1 in expert_ids.
        return_scatter_idx: Whether to additionally return a masked identity
            mapping for the original routed rows.

    Returns:
    - sorted_token_ids: A tensor containing the sorted token indices according
        to their allocated expert.
    - expert_ids: A tensor indicating the assigned expert index for each block.
    - num_tokens_post_padded: The total number of tokens after padding,
        ensuring divisibility by block_size.
    - scatter_idx: Only returned when return_scatter_idx=True. An int32 tensor
        shaped [topk_ids.numel(), 1], containing the original routed row index
        for valid routes and -1 otherwise. With expert_map, this requires
        ignore_invalid_experts=True.

    This function pads the number of tokens that each expert needs to process
    so that it is divisible by block_size.
    Padding ensures that during block matrix multiplication, the dimensions
    align correctly.

    Example:
    Given topk_ids = [[2, 3, 4], [1, 2, 4], [1, 3, 4], [1, 2, 3]],
    block_size = 4, and num_experts = 4:
    - We initially have 12 tokens (after repeating 'top_k' times) and 4 experts,
        with each expert needing to process 3 tokens.
    - As block_size is 4, we pad 1 token for each expert.
    - First, flatten topk_ids to [2, 3, 4, 1, 2, 4, 1, 3, 4, 1, 2, 3].
    - Then append padding tokens [12, 12, 12, 12] for each block.
    - After sorting by expert index, we obtain token_ids
        [3, 6, 9, 12, 0, 4, 10, 12, 1, 7, 11, 12, 2, 5, 8, 12].
        Tokens 12 are non-existent (padding) and are ignored in
        the subsequent matrix multiplication.
    - The padding ensures that the total number of tokens is now divisible
        by block_size for proper block matrix operations.

    """
    if return_scatter_idx and expert_map is not None and not ignore_invalid_experts:
        raise ValueError("scatter_idx with expert_map requires ignore_invalid_experts")
    scatter_idx = None
    if return_scatter_idx:
        scatter_idx = torch.empty(
            (topk_ids.numel(), 1),
            dtype=torch.int32,
            device=topk_ids.device,
        )
    max_num_tokens_padded = topk_ids.numel() + num_experts * (block_size - 1)
    if pad_sorted_ids:
        max_num_tokens_padded = round_up(max_num_tokens_padded, block_size)
    if topk_ids.numel() < num_experts:
        max_num_tokens_padded = min(
            topk_ids.numel() * block_size, max_num_tokens_padded
        )
    sorted_ids = torch.empty(
        (max_num_tokens_padded,), dtype=torch.int32, device=topk_ids.device
    )
    max_num_m_blocks = triton.cdiv(max_num_tokens_padded, block_size)
    expert_ids = torch.empty(
        (max_num_m_blocks,), dtype=torch.int32, device=topk_ids.device
    )
    num_tokens_post_pad = torch.empty((1), dtype=torch.int32, device=topk_ids.device)

    if _DETERMINISTIC_ALIGN:
        # The CUDA op hands each token its slot within its expert's block
        # with an atomicAdd, so the ORDER of the tokens inside a block is
        # whatever order the atomics happened in. The expert GEMM then
        # accumulates the block's tokens in that order, and fp32 addition
        # is not associative, so the SAME request produced different
        # results run to run (and the difference grew with the layer
        # depth, ending in a repetition loop at temperature 1.0).
        #
        # This computes the same layout with a STABLE sort by
        # (expert_id, token_index), which is both deterministic and the
        # ordering the reference implementation uses.
        _deterministic_align(
            topk_ids,
            block_size,
            num_experts,
            sorted_ids,
            expert_ids,
            num_tokens_post_pad,
            scatter_idx,
            # the op is given the map only when the caller wants the
            # LOCAL ids (ignore_invalid_experts); otherwise the caller
            # maps the block ids itself.
            expert_map if ignore_invalid_experts else None,
            ignore_invalid_experts,
        )
    else:
        ops.moe_align_block_size(
            topk_ids,
            num_experts,
            block_size,
            sorted_ids,
            expert_ids,
            num_tokens_post_pad,
            expert_map if ignore_invalid_experts else None,
            scatter_idx,
        )

    if expert_map is not None and not ignore_invalid_experts:
        # the block ids hold the LOCAL expert ids (the CUDA op's
        # get_local_expert_id applies the map); the caller maps them
        # back to global ids here - exactly as for the CUDA path.
        expert_ids = expert_map[expert_ids]

    if scatter_idx is not None:
        return sorted_ids, expert_ids, num_tokens_post_pad, scatter_idx
    return sorted_ids, expert_ids, num_tokens_post_pad


def batched_moe_align_block_size(
    max_tokens_per_batch: int, block_size: int, expert_num_tokens: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Given num_batches, max_tokens_per_batch, block_size and the number of
    valid-tokens in each batch, prepare sorted_token_ids, expert_ids and
    num_tokens_post_pad. sorted_token_ids, expert_ids and num_tokens_post_pad
    have the same semantics as in moe_align_block_size.

    This function is intended to be a drop in replacement for
    moe_align_batch_size for the batched case.

    Args:
        max_tokens_per_batch (int): Number of tokens in each batch (both
            valid and invalid).
        block_size (int): block_size to align the data to.
        expert_num_tokens (torch.Tensor): expert_num_tokens[i], indicates
            the number of valid tokens in batch i.

    Returns:
    - sorted_token_ids (torch.Tensor): Torch tensor of size
        (num_batches * max_tokens_per_batch) indicating the token indices for
        that block.
    - expert_ids (torch.Tensor): Torch tensor of size
        ceil((num_batches * max_tokens_per_batch) / block_size) indicating
        what expert to use for each block.
    - num_tokens_post_pad (torch.Tensor): Torch tensor of size 1
        indicating the number of valid blocks with actual data to
        process. This is represented in terms of num tokens.

    Example:
    Let num_batches=5, max_tokens_per_batch=8, block_size=4, and
    expert_num_tokens=[2, 3, 0, 6, 8]. This expert_num_tokens tensor
    indicates that,
     - The first 2 tokens in the 0th batch are valid and the rest 6 are
     invalid (i.e. in the 2D hidden_states tensor of shape,
     [num_batches * max_tokens_per_batch, K], indices 0, 1 are valid)
     - The first 3 tokens in the 1st batch are valid. i.e. indices 8, 9, 10
     - 0 tokens in the 2nd batch are valid
     - first 6 tokens in the  3rd batch are valid. i.e. indices,
     24, 25, 26, 27, 28, 29
     - so on ...

     In this case,
      sorted_token_ids will be [0, 1, 40, 40,
                                8, 9, 10, 40,
                                24, 25, 26, 27,
                                28, 29, 40, 40,
                                32, 33, 34, 35,
                                36, 37, 38, 39,
                                40, 40, 40, 40,
                                (rest all 40, 40, 40, 40)
                                ...]
      Here, 40 represents an invalid index. as there is no token index 40.
      The gemm kernel using this sorted_token_ids is expected to skip the
      gemm computation when it encounters this invalid index.

      expert_ids will be [0, 1, 3, 3, 4, 5, 5, -1, -1, (rest all -1) ...]
      Here, -1 represents an invalid expert. The gemm kernel using this
      expert_ids is expected to skip the gemm computation when it encounters
      an expert of id -1.

      num_tokens_post_pad will be 24 as sorted_token_ids has valid entries
      until 24.

    """
    B = expert_num_tokens.size(0)
    device = expert_num_tokens.device

    # Round up so each batch can be split to blocks evenly.
    max_num_tokens_padded = B * round_up(max_tokens_per_batch, block_size)

    sorted_ids = torch.empty((max_num_tokens_padded,), dtype=torch.int32, device=device)
    assert max_num_tokens_padded % block_size == 0
    max_num_m_blocks = max_num_tokens_padded // block_size
    expert_ids = torch.empty((max_num_m_blocks,), dtype=torch.int32, device=device)
    num_tokens_post_pad = torch.empty((1), dtype=torch.int32, device=device)

    ops.batched_moe_align_block_size(
        max_tokens_per_batch,
        block_size,
        expert_num_tokens,
        sorted_ids,
        expert_ids,
        num_tokens_post_pad,
    )

    return sorted_ids, expert_ids, num_tokens_post_pad
