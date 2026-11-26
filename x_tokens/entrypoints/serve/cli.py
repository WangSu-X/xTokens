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
    parser.add_argument("--model", required=True)
    parser.add_argument("--served-model-name")
    parser.add_argument(
        "--engine",
        choices=("inproc", "multiproc"),
        default="multiproc",
        help="EngineCore client transport (default: %(default)s)",
    )
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--max-num-seqs", type=int, default=4)
    parser.add_argument(
        "--max-prefill-batch-tokens",
        type=int,
        help="Set the token budget for uncached prefill work",
    )
    parser.add_argument("--enable-chunked-prefill", action="store_true")
    parser.add_argument(
        "--prefill-chunk-alignment",
        type=int,
        help="Token alignment for intermediate chunks when several prefills compete",
    )
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument(
        "--kv-cache-block-size", type=int, help="KV block size (default: 16)"
    )
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    cache_budget = parser.add_mutually_exclusive_group()
    cache_budget.add_argument("--num-gpu-blocks", type=int)
    cache_budget.add_argument(
        "--kv-cache-memory-bytes", type=int, help="KV cache byte budget per rank"
    )
    parser.add_argument("--disable-prefix-caching", action="store_true")
    parser.add_argument(
        "--distributed-executor-backend", choices=("uni", "mp"), default="uni"
    )
    parser.add_argument("--parallel-device", action="append")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--pipeline-parallel-size", type=int, default=1)
    parser.add_argument("--context-parallel-size", type=int, default=1)
    parser.add_argument("--data-parallel-size", type=int, choices=(1,), default=1)
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
    )
    parallel_devices = args.parallel_device
    if parallel_devices is None:
        parallel_devices = [f"cuda:{index}" for index in range(parallel_world_size)]
    config = XTokensConfig(
        model_config=ModelConfig(
            served_model_name=args.served_model_name or args.model,
            model=args.model,
            max_model_len=args.max_model_len,
        ),
        scheduler_config=SchedulerConfig(
            max_num_seqs=args.max_num_seqs,
            max_prefill_batch_tokens=args.max_prefill_batch_tokens,
            enable_chunked_prefill=args.enable_chunked_prefill,
            prefill_chunk_alignment=args.prefill_chunk_alignment,
        ),
        executor_config=ExecutorConfig(
            dtype=args.dtype,
            local_files_only=args.local_files_only,
        ),
        kv_cache_config=KVCacheConfig(
            block_size=args.kv_cache_block_size
            if args.kv_cache_block_size is not None
            else KVCacheConfig().block_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
            num_gpu_blocks=args.num_gpu_blocks,
            enable_prefix_caching=not args.disable_prefix_caching,
            memory_bytes=args.kv_cache_memory_bytes,
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
            core_client_mode="in_proc" if args.engine == "inproc" else "multi_proc",
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
