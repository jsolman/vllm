# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
from vllm.config import VllmConfig
from vllm.renderers.chat_utils import (
    ChatCompletionMessageParam,
    ConversationMessage,
    parse_chat_messages,
    parse_chat_messages_async,
)
from vllm.tokenizers.deepseek_v4 import DeepseekV4Tokenizer
from vllm.utils.async_utils import make_async

from .base import BaseRenderer
from .inputs import DictPrompt
from .inputs.preprocess import parse_dec_only_prompt
from .params import ChatParams

_FIM_BEGIN = "<｜fim▁begin｜>"
_FIM_HOLE = "<｜fim▁hole｜>"
_FIM_END = "<｜fim▁end｜>"


def _dsv41_flag(name: str) -> bool:
    """Read a debug flag from the env OR from /tmp/dsv41-flags."""
    if os.environ.get(name, "0") == "1":
        return True
    try:
        with open("/tmp/dsv41-flags") as f:
            for line in f:
                if line.strip() == name:
                    return True
    except OSError:
        pass
    return False


class DeepseekV4Renderer(BaseRenderer[DeepseekV4Tokenizer]):
    def __init__(
        self,
        config: VllmConfig,
        tokenizer: DeepseekV4Tokenizer | None,
    ) -> None:
        super().__init__(config, tokenizer)

        self._apply_chat_template_async = make_async(
            self._apply_chat_template, executor=self._executor
        )

    def _apply_chat_template(self, *args, **kwargs):
        rendered = self.get_tokenizer().apply_chat_template(*args, **kwargs)
        if _dsv41_flag("VLLM_DSV41_LOG_PROMPT"):
            # Debug: the server's ACTUAL rendered prompt. The encoder was
            # verified to match the model's own reference byte for byte,
            # but that was the ENCODER in isolation - this is what the
            # server really sends, including the thinking mode and the
            # reasoning effort it chose.
            try:
                prompt_raw = rendered[0] if isinstance(rendered, tuple) else rendered
                print(f"DSV41 PROMPT len={len(prompt_raw)} "
                      f"hash={hash(tuple(prompt_raw[:400]))} "
                      f"head={prompt_raw[:160]!r}", flush=True)
            except Exception as e:
                print(f"DSV41 PROMPT err {e}", flush=True)
        return rendered, None

    def _build_prompt(self, rendered, mm_data, mm_uuids) -> DictPrompt:
        prompt_raw, image_order = rendered
        prompt = parse_dec_only_prompt(prompt_raw)
        if image_order is not None:
            for items in (mm_data, mm_uuids):
                if items is not None and "image" in items:
                    if image_order:
                        items["image"] = [items["image"][i] for i in image_order]
                    else:
                        del items["image"]
        if mm_data:
            prompt["multi_modal_data"] = mm_data
        if mm_uuids:
            prompt["multi_modal_uuids"] = mm_uuids
        return prompt

    def render_completion_suffix(self, prompt: str, suffix: str) -> str | None:
        return f"{_FIM_BEGIN}{prompt}{_FIM_HOLE}{suffix}{_FIM_END}"

    def render_messages(
        self,
        messages: list[ChatCompletionMessageParam],
        params: ChatParams,
    ) -> tuple[list[ConversationMessage], DictPrompt]:
        conversation, mm_data, mm_uuids = parse_chat_messages(
            messages,
            self.model_config,
            content_format="openai",
            media_io_kwargs=params.media_io_kwargs,
            mm_processor_kwargs=params.mm_processor_kwargs,
        )

        rendered = self._apply_chat_template(
            conversation=conversation,
            messages=messages,
            **params.get_apply_chat_template_kwargs(),
        )

        return conversation, self._build_prompt(rendered, mm_data, mm_uuids)

    async def render_messages_async(
        self,
        messages: list[ChatCompletionMessageParam],
        params: ChatParams,
    ) -> tuple[list[ConversationMessage], DictPrompt]:
        conversation, mm_data, mm_uuids = await parse_chat_messages_async(
            messages,
            self.model_config,
            content_format="openai",
            media_io_kwargs=params.media_io_kwargs,
            mm_processor_kwargs=params.mm_processor_kwargs,
        )

        rendered = await self._apply_chat_template_async(
            conversation=conversation,
            messages=messages,
            **params.get_apply_chat_template_kwargs(),
        )

        return conversation, self._build_prompt(rendered, mm_data, mm_uuids)


class DeepseekV41Renderer(DeepseekV4Renderer):
    def _apply_chat_template(self, *args, **kwargs):
        image_order: list[int] = []
        kwargs["image_order"] = image_order
        prompt_raw = self.get_tokenizer().apply_chat_template(*args, **kwargs)
        if _dsv41_flag("VLLM_DSV41_LOG_PROMPT"):
            # Debug: the server's ACTUAL rendered prompt. The encoder was
            # verified to match the model's own reference byte for byte,
            # but that was the ENCODER in isolation - this is what the
            # server really sends, including the thinking mode and the
            # reasoning effort it chose.
            try:
                import hashlib
                _p = prompt_raw[0] if isinstance(prompt_raw, tuple) else prompt_raw
                # a STABLE digest: Python's hash() on strings is
                # randomised per process, so it is not comparable
                # between the server and an offline reference.
                _sha = hashlib.sha256(_p.encode()).hexdigest()
                print(f"DSV41 PROMPT len={len(_p)} sha256={_sha}", flush=True)
                # dump the full prompt so it can be diffed byte for byte
                # against the model's own reference rendering
                _path = "/tmp/dsv41-server-prompt.txt"
                try:
                    with open(_path, "w") as _f:
                        _f.write(_p)
                except OSError:
                    pass
            except Exception as e:
                print(f"DSV41 PROMPT err {e}", flush=True)
        return prompt_raw, image_order
