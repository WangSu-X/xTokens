#!/usr/bin/env python
"""Launch xTokens API Server with mock engine (no GPU/model required)."""

from __future__ import annotations

import argparse

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
from x_tokens.entrypoints.serve.app import create_app, mock_engine_factory
from x_tokens.logger import create_uvicorn_log_config


def main() -> None:
    """Run xTokens API Server with mock engine."""
    parser = argparse.ArgumentParser(
        description="Run xTokens Serve API with mock engine (no GPU/model required)"
    )
    parser.add_argument("--host", default="127.0.0.1", help="Server host")
    parser.add_argument("--port", type=int, default=8000, help="Server port")
    parser.add_argument(
        "--served-model-name",
        default="mock-model",
        help="Model name to serve as",
    )
    parser.add_argument(
        "--api-key",
        default="test-key",
        help="API key for authentication (default: test-key)",
    )
    parser.add_argument("--workers", type=int, default=1, help="Number of workers")
    parser.add_argument(
        "--no-access-log", action="store_true", help="Disable access logging"
    )

    args = parser.parse_args()

    # Create minimal config for mock engine
    config = XTokensConfig(
        model_config=ModelConfig(
            served_model_name=args.served_model_name,
            model="mock",  # Dummy model name, not used in mock
            max_model_len=2048,
        ),
        scheduler_config=SchedulerConfig(max_num_seqs=4),
        executor_config=ExecutorConfig(dtype="auto", local_files_only=False),
        kv_cache_config=KVCacheConfig(
            block_size=16,
            gpu_memory_utilization=0.9,
            enable_prefix_caching=True,
        ),
        parallel_config=ParallelConfig(),
        server_config=ServerConfig(
            api_key=args.api_key,
            request_timeout_s=30,
        ),
    )

    # Create app with mock engine factory
    app = create_app(config=config, engine_factory=mock_engine_factory)

    # Start uvicorn server
    log_config = create_uvicorn_log_config()

    print(f"\n🚀 xTokens Mock Server Starting")
    print(f"   Host: {args.host}:{args.port}")
    print(f"   Served Model: {args.served_model_name}")
    print(f"   API Key: {args.api_key}")
    print(f"   Workers: {args.workers}")
    print(f"\n📝 Example requests:")
    print(f"   curl -X GET http://{args.host}:{args.port}/v1/models \\")
    print(f"     -H 'Authorization: Bearer {args.api_key}'")
    print(f"\n   curl -X POST http://{args.host}:{args.port}/v1/completions \\")
    print(f"     -H 'Authorization: Bearer {args.api_key}' \\")
    print(f"     -H 'Content-Type: application/json' \\")
    print(f"     -d '{{\n      \"model\": \"{args.served_model_name}\",")
    print(f"      \"prompt\": \"Hello, world!\",")
    print(f"      \"max_tokens\": 10,")
    print(f"      \"stream\": false\n    }}'")
    print(f"\n" + "=" * 60 + "\n")

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        workers=args.workers,
        log_config=log_config if not args.no_access_log else None,
    )


if __name__ == "__main__":
    main()
