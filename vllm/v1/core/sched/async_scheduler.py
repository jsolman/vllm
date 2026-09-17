# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import logging

from vllm.logger import init_logger
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.request import Request, RequestStatus

logger = init_logger(__name__)


class AsyncScheduler(Scheduler):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # reusable read-only placeholder list for speculative decoding.
        self._spec_token_placeholders: list[int] = [-1] * self.num_spec_tokens
        self.pp_size = self.parallel_config.pipeline_parallel_size

    def _update_after_schedule(self, scheduler_output: SchedulerOutput) -> None:
        super()._update_after_schedule(scheduler_output)
        spec_decode_tokens = scheduler_output.scheduled_spec_decode_tokens
        # Use the latest num of scheduled draft tokens in next step as placeholder.
        self._spec_token_placeholders = [
            -1
        ] * scheduler_output.num_spec_tokens_to_schedule
        for req_id in scheduler_output.num_scheduled_tokens:
            request = self.requests[req_id]
            if request.is_prefill_chunk:
                continue

            scheduler_output.pending_structured_output_tokens |= (
                request.use_structured_output and request.num_output_placeholders > 0
            )
            # The request will generate num_sampled_tokens_per_step new tokens
            # plus num_spec_tokens in this scheduling step. Diffusion has no AR
            # bonus token (num_sampled_tokens_per_step == 0) — only the canvas
            # (spec) tokens.
            cur_num_spec_tokens = len(spec_decode_tokens.get(req_id, ()))
            # Debug-level ledger of every placeholder add (the underflow
            # detector below stays at WARNING: it is the recurrence
            # detector for the finished-request double-drain bug).
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "MTP placeholder add: req=%s add=%d (sampled=%d spec=%d) "
                    "pre_ph=%d post_ph=%d computed=%d chunk=%s out_toks=%d",
                    req_id,
                    self.num_sampled_tokens_per_step + cur_num_spec_tokens,
                    self.num_sampled_tokens_per_step,
                    cur_num_spec_tokens,
                    request.num_output_placeholders,
                    request.num_output_placeholders
                    + self.num_sampled_tokens_per_step
                    + cur_num_spec_tokens,
                    request.num_computed_tokens,
                    request.is_prefill_chunk,
                    len(request._output_token_ids),
                )
            request.num_output_placeholders += (
                self.num_sampled_tokens_per_step + cur_num_spec_tokens
            )
            # Add placeholders for the new draft/spec tokens.
            # We will update the actual spec token ids in the worker process.
            request.spec_token_ids = self._spec_token_placeholders

            if self.use_v2_model_runner:
                # Set the next step index in which this request is eligible to be
                # scheduled for decode (for PP microbatching).
                request.next_decode_eligible_step = self.current_step + self.pp_size

    def _update_request_with_output(
        self, request: Request, new_token_ids: list[int], is_stale: bool = False
    ) -> tuple[list[int], bool]:
        status_before_update = request.status
        new_token_ids, stopped = super()._update_request_with_output(
            request, new_token_ids
        )

        # Placeholders were zeroed at preemption; a stale delivery must not
        # decrement them (it would underflow).
        #
        # FIX (MTP placeholder underflow, 9/17): a finished request (e.g.
        # FINISHED_LENGTH_CAPPED hit mid-step) can still receive an in-flight
        # async output row from a schedule issued before the finish was
        # detected. That row's placeholders were already consumed by the
        # finish-time accounting; draining again double-counts, asserts (or,
        # clamped, corrupts num_computed/placeholders -> wrong cache_blocks
        # offset -> poisoned prefix-cache blocks). Skip both the drain and the
        # re-cache for finished requests. Also clamp defensively: a negative
        # balance here must never assert-crash the engine.
        if not is_stale and not request.is_finished():
            _remaining = request.num_output_placeholders
            _drain = len(new_token_ids)
            if _drain > _remaining:
                logger.warning(
                    "MTP placeholder underflow probe: request %s would drain "
                    "%d placeholder(s) but only %d remain "
                    "(num_computed_tokens=%d, status=%s, "
                    "output_token_ids=%d). Clamping to %d; this delivery is "
                    "double-accounted \u2014 investigate spec-reject/stop trim "
                    "interaction.",
                    request.request_id,
                    _drain,
                    _remaining,
                    request.num_computed_tokens,
                    request.status,
                    len(request._output_token_ids),
                    _remaining,
                )
                _drain = _remaining
            request.num_output_placeholders -= _drain

        # Cache the new tokens. Preempted requests should be skipped.
        if status_before_update == RequestStatus.RUNNING:
            self.kv_cache_manager.cache_blocks(
                request, request.num_computed_tokens - request.num_output_placeholders
            )
        return new_token_ids, stopped
