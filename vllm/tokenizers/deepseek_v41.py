# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import copy
import os
from typing import Any

from transformers import TokenizersBackend

from vllm.renderers.chat_utils import ChatCompletionMessageParam

from .deepseek_v41_encoding import (
    IMAGE_PLACEHOLDER,
    REASONING_EFFORT_MAPPINGS,
    encode_messages,
)
from .hf import HfTokenizer, get_cached_tokenizer
from .protocol import TokenizerLike


def _reject_image_placeholder(text: Any) -> None:
    if isinstance(text, str) and IMAGE_PLACEHOLDER in text:
        raise ValueError(
            f"Message text contains image special token {IMAGE_PLACEHOLDER!r}; "
            "provide images as image content parts."
        )


def _normalize_messages(
    messages: list[ChatCompletionMessageParam],
) -> list[dict[str, Any]]:
    # vLLM additions to the reference: Responses API part types, image_pil,
    # and ValueError (400) instead of the reference's asserts.
    result = [dict(message) for message in copy.deepcopy(messages)]
    image_idx = 0
    for message in result:
        image_indices = message["_image_indices"] = []
        role = message.get("role")
        if role not in ("system", "developer", "user", "assistant", "tool"):
            raise ValueError(f"Invalid role: {role}")
        if "reasoning" in message:
            message["reasoning_content"] = message["reasoning"]
        _reject_image_placeholder(message.get("reasoning_content"))
        content = message.get("content")
        if isinstance(content, str):
            _reject_image_placeholder(content)
        elif isinstance(content, list):
            parts = []
            for block in content:
                part_type = block.get("type")
                if part_type in ("text", "input_text", "output_text"):
                    _reject_image_placeholder(block.get("text"))
                    parts.append(block.get("text", ""))
                elif part_type in ("image_url", "input_image", "image_pil"):
                    parts.append(IMAGE_PLACEHOLDER)
                    image_indices.append(image_idx)
                    image_idx += 1
                else:
                    raise ValueError(
                        "DeepSeek V4.1 supports text and image content only; "
                        f"got {part_type!r}"
                    )
            message["content"] = "\n\n".join(parts)
    return result


def get_deepseek_v41_tokenizer(tokenizer: HfTokenizer) -> HfTokenizer:
    """Wrap an HF tokenizer with the V4.1 prompt encoder."""
    wrapped = copy.copy(tokenizer)
    added_vocab = tokenizer.get_added_vocab()

    class _DeepseekV41Tokenizer(tokenizer.__class__):  # type: ignore
        def apply_chat_template(
            self,
            messages: list[ChatCompletionMessageParam],
            tools: list[dict[str, Any]] | None = None,
            image_order: list[int] | None = None,
            **kwargs,
        ) -> str | list[int]:
            # The generic renderer's conversation has already flattened text
            # parts with '\n'. V4.1 encodes the original messages with '\n\n'.
            conversation = _normalize_messages(messages)
            if tools:
                system = next((m for m in conversation if m["role"] == "system"), None)
                if system is None:
                    system = {"role": "system", "content": ""}
                    conversation.insert(0, system)
                system["tools"] = tools

            thinking = bool(kwargs.get("thinking") or kwargs.get("enable_thinking"))
            # An EXPLICIT reasoning effort implies the thinking mode: the
            # effort prefix is only rendered in the thinking mode, so
            # asking for an effort and getting the chat mode would
            # silently drop it. This also matches the parser, which
            # treats an effort other than "none" as the thinking mode
            # (vllm/parser/deepseek_v4.py) - the tokenizer and the
            # parser must agree, or the reasoning and the content get
            # swapped.
            if "thinking" not in kwargs and "enable_thinking" not in kwargs:
                if kwargs.get("reasoning_effort") is not None:
                    thinking = kwargs.get("reasoning_effort") != "none"
                else:
                    # The DEFAULT thinking mode: the THINKING mode.
                    #
                    # The reference's STANDALONE inference script
                    # defaults to the chat mode (inference/generate.py:
                    # --thinking-mode default="chat"), but the SERVED
                    # deployment uses the THINKING mode, and the token
                    # counts prove it:
                    #
                    #   the single-turn sermon request:
                    #     the cloud 6996, the thinking mode 6996, the chat mode 6971
                    #   the multi-turn (2 tool results):
                    #     the cloud 7849, the thinking mode 7849, the chat mode 7822
                    #   the multi-turn (6 tool results):
                    #     the cloud 8018, the thinking mode 8018, the chat mode 7986
                    #
                    # The cloud's counts match the THINKING mode EXACTLY,
                    # so the thinking mode is what the reference
                    # deployment serves - and matching it is what makes
                    # the local's first-token distribution match the
                    # cloud's.
                    #
                    # Set VLLM_DSV41_THINKING_MODE=chat for the
                    # standalone script's behaviour (no effort prefix,
                    # no <think>).
                    thinking = os.environ.get(
                        "VLLM_DSV41_THINKING_MODE", "thinking"
                    ).strip().lower() not in ("chat", "0", "false", "off")
            effort = kwargs.get("reasoning_effort")
            if effort == "none":
                thinking = False
                effort = None
            if effort is None:
                effort = "high"
            if not (
                (type(effort) is int and 1 <= effort <= 100)
                or (isinstance(effort, str) and effort in REASONING_EFFORT_MAPPINGS)
            ):
                raise ValueError(
                    "DeepSeek V4.1 reasoning_effort must be low, high, xhigh, max, "
                    "or an integer within [1, 100] in chat_template_kwargs"
                )

            prompt = encode_messages(
                conversation,
                thinking_mode="thinking" if thinking else "chat",
                drop_thinking=kwargs.get("drop_thinking", True),
                reasoning_effort=effort,
                image_order=image_order,
            )
            if kwargs.get("tokenize", True):
                tokenizer_kwargs = {
                    key: kwargs[key]
                    for key in ("truncation", "max_length")
                    if key in kwargs
                }
                return self.encode(prompt, add_special_tokens=False, **tokenizer_kwargs)
            return prompt

        def num_special_tokens_to_add(self) -> int:
            return len(self.encode(""))

        def get_added_vocab(self) -> dict[str, int]:
            return added_vocab.copy()

        def __reduce__(self):
            return get_deepseek_v41_tokenizer, (tokenizer,)

    _DeepseekV41Tokenizer.__name__ = f"DSV41{tokenizer.__class__.__name__}"
    wrapped.__class__ = _DeepseekV41Tokenizer
    return wrapped


class DeepseekV41Tokenizer(TokenizerLike):
    @classmethod
    def from_pretrained(cls, *args, **kwargs) -> HfTokenizer:
        tokenizer = TokenizersBackend.from_pretrained(*args, **kwargs)
        return get_cached_tokenizer(get_deepseek_v41_tokenizer(tokenizer))
