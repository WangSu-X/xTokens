"""Composition root and command-line entrypoint for Serve."""

from __future__ import annotations

import argparse
import os

import uvicorn

from x_tokens.config import (
    ExecutorConfig,
    KVCacheConfig,
    ModelConfig,
    ParallelConfig,
    SchedulerConfig,
    ServerConfig,
    XTokensConfig,
)
from x_tokens.logger import create_uvicorn_log_config

from .app import create_app


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the xTokens Serve API")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--model", dest="served_model_name", default="x-tokens-mock")
    parser.add_argument("--engine", choices=("inproc",), default="inproc")
    parser.add_argument("--hf-model")
    parser.add_argument(
        "--executor-backend", choices=("naive_hf", "custom"), default="naive_hf"
    )
    parser.add_argument("--hf-device", default="auto")
    parser.add_argument("--hf-dtype", default="auto")
    parser.add_argument("--hf-local-files-only", action="store_true")
    parser.add_argument("--hf-max-num-seqs", type=int, default=4)
    parser.add_argument("--kv-cache-block-size", type=int, default=16)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--num-gpu-blocks", type=int)
    parser.add_argument("--disable-prefix-caching", action="store_true")
    parser.add_argument(
        "--distributed-executor-backend", choices=("uni", "mp"), default="uni"
    )
    parser.add_argument("--parallel-device", action="append")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--pipeline-parallel-size", type=int, default=1)
    parser.add_argument("--context-parallel-size", type=int, default=1)
    parser.add_argument("--data-parallel-size", type=int, default=1)
    parser.add_argument("--enable-expert-parallel", action="store_true")
    parser.add_argument("--expert-parallel-size", type=int)
    parser.add_argument("--collective-backend", default="nccl")
    parser.add_argument("--api-key", default=os.environ.get("XTOKENS_API_KEY"))
    parser.add_argument("--request-timeout", type=float, default=600.0)
    parser.add_argument("--shutdown-timeout", type=float, default=30.0)
    parser.add_argument(
        "--shutdown-policy", choices=("abort", "drain"), default="abort"
    )
    parser.add_argument("--cors-origin", action="append", default=[])
    parser.add_argument("--no-access-log", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    parallel_world_size = (
        args.tensor_parallel_size
        * args.pipeline_parallel_size
        * args.context_parallel_size
        * args.data_parallel_size
    )
    parallel_devices = args.parallel_device
    if parallel_devices is None:
        parallel_devices = [f"cuda:{index}" for index in range(parallel_world_size)]
    config = XTokensConfig(
        model_config=ModelConfig(
            served_model_name=args.served_model_name,
            model=args.hf_model,
        ),
        scheduler_config=SchedulerConfig(max_num_seqs=args.hf_max_num_seqs),
        executor_config=ExecutorConfig(
            backend=args.executor_backend,
            device=args.hf_device,
            dtype=args.hf_dtype,
            local_files_only=args.hf_local_files_only,
        ),
        kv_cache_config=KVCacheConfig(
            block_size=args.kv_cache_block_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
            num_gpu_blocks=args.num_gpu_blocks,
            enable_prefix_caching=not args.disable_prefix_caching,
        ),
        parallel_config=ParallelConfig(
            distributed_executor_backend=args.distributed_executor_backend,
            devices=tuple(parallel_devices),
            tensor_parallel_size=args.tensor_parallel_size,
            pipeline_parallel_size=args.pipeline_parallel_size,
            context_parallel_size=args.context_parallel_size,
            data_parallel_size=args.data_parallel_size,
            enable_expert_parallel=args.enable_expert_parallel,
            expert_parallel_size=args.expert_parallel_size,
            collective_backend=args.collective_backend,
        ),
        server_config=ServerConfig(
            host=args.host,
            port=args.port,
            api_key=args.api_key,
            request_timeout_s=args.request_timeout,
            shutdown_timeout_s=args.shutdown_timeout,
            shutdown_policy=args.shutdown_policy,
            cors_origins=tuple(args.cors_origin),
            access_log=not args.no_access_log,
        ),
    )
    uvicorn.run(
        create_app(config),
        host=config.server_config.host,
        port=config.server_config.port,
        access_log=config.server_config.access_log,
        log_config=create_uvicorn_log_config(),
    )


if __name__ == "__main__":
    main()
