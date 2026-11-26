"""Worker-side CUDA timings for opt-in model benchmarks."""

from __future__ import annotations

import functools
import importlib.abc
import importlib.machinery
import json
import os
import sys
from pathlib import Path

CONTROL = os.environ.get("XTOKENS_BENCH_CONTROL")
AIPERF = os.environ.get("XTOKENS_BENCH_AIPERF")
TARGETS = {"x_tokens.worker.model_runner.gpu_model_runner"}
if AIPERF:
    TARGETS.add("aiperf.endpoints.openai_completions")


def control():
    if CONTROL:
        path = Path(CONTROL)
        if path.exists():
            return json.loads(path.read_text())
    return None


def record(operation, details, call, *args, **kwargs):
    import torch

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    result = call(*args, **kwargs)
    end.record()
    end.synchronize()
    output = Path(details["trace"])
    row = {key: value for key, value in details.items() if key != "trace"}
    row.update(operation=operation, gpu_ms=start.elapsed_time(end))
    with output.open("a") as stream:
        stream.write(json.dumps(row) + "\n")
    return result


def install_xtokens(module):
    owner = module.GPUModelRunner
    forward = owner._forward
    sample = owner.sample_tokens

    @functools.wraps(forward)
    def timed_forward(self, inputs, **kwargs):
        details = control() if self.is_output_rank else None
        if details is None:
            return forward(self, inputs, **kwargs)
        requests = inputs.requests
        details.update(
            query_tokens=sum(len(request.token_ids) for request in requests),
            context_tokens=[
                request.num_computed_tokens + len(request.token_ids)
                for request in requests
            ],
            is_prefill=any(
                request.num_computed_tokens < details["prompt_tokens"]
                for request in requests
            ),
        )
        if details["is_prefill"]:
            details["actual_prompt_tokens"] = list(requests[0].token_ids)
        else:
            details["actual_decode_tokens"] = list(requests[0].token_ids)
        return record("forward_with_head", details, forward, self, inputs, **kwargs)

    @functools.wraps(sample)
    def fixed_sample(self, step_id):
        result = sample(self, step_id)
        details = control() if self.is_output_rank else None
        if details is not None and result is not None:
            return (details["decode_token"],) * len(result)
        return result

    owner._forward = timed_forward
    owner.sample_tokens = fixed_sample


def install_aiperf(module):
    owner = module.CompletionsEndpoint
    extract = owner.extract_completions_response_data

    @functools.wraps(extract)
    def normalize_chunk(self, payload):
        if payload.get("object") == "text_completion.chunk":
            payload = {**payload, "object": "text_completion"}
        return extract(self, payload)

    owner.extract_completions_response_data = normalize_chunk


class BenchmarkFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname not in TARGETS:
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path, target)
        original = spec.loader.exec_module

        def execute(module):
            original(module)
            if fullname.startswith("aiperf."):
                install_aiperf(module)
            else:
                install_xtokens(module)

        spec.loader.exec_module = execute
        return spec


if CONTROL or AIPERF:
    sys.meta_path.insert(0, BenchmarkFinder())
