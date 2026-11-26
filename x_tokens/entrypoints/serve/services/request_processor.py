"""Convert OpenAI API requests to engine prompts and sampling parameters."""

from __future__ import annotations

from typing import Any

from x_tokens.engine.protocol import SamplingParams
from x_tokens.entrypoints.serve.openai.protocol import (
    ChatCompletionRequest,
    CompletionRequest,
)


def process_completion_request(
    request: CompletionRequest,
) -> tuple[str, SamplingParams]:
    sampling_params = SamplingParams(
        max_tokens=request.max_tokens,
        temperature=request.temperature,
        top_p=request.top_p,
        top_k=request.top_k,
        stop=tuple(request.stop or ()),
        ignore_eos=request.ignore_eos,
    )
    return request.prompt, sampling_params


def process_chat_request(
    request: ChatCompletionRequest, *, tokenizer: Any | None = None
) -> tuple[str | tuple[int, ...], SamplingParams]:
    if getattr(tokenizer, "chat_template", None):
        prompt = tuple(
            tokenizer.apply_chat_template(
                [message.model_dump() for message in request.messages],
                add_generation_prompt=True,
                tokenize=True,
                return_dict=False,
            )
        )
    else:
        prompt = "\n".join(
            f"{message.role}: {message.content}" for message in request.messages
        )
    sampling_params = SamplingParams(
        max_tokens=request.max_completion_tokens or request.max_tokens or 512,
        temperature=request.temperature,
        top_p=request.top_p,
        top_k=request.top_k,
        stop=tuple(request.stop or ()),
        ignore_eos=request.ignore_eos,
    )
    return prompt, sampling_params
