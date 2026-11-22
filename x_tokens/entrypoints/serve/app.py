"""FastAPI application factory for the OpenAI-compatible Serve entrypoint."""

from __future__ import annotations

import contextlib
import logging
from collections.abc import AsyncIterator

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from x_tokens.config import XTokensConfig
from x_tokens.engine.llm_engine import LLMEngine
from x_tokens.entrypoints.serve.auth_middleware import APIKeyMiddleware
from x_tokens.entrypoints.serve.errors import OpenAIError
from x_tokens.entrypoints.serve.routes import chat, completions, models
from x_tokens.entrypoints.serve.services.chat_service import ChatCompletionService
from x_tokens.entrypoints.serve.services.completion_service import CompletionService

logger = logging.getLogger(__name__)


def create_app(config: XTokensConfig | None = None) -> FastAPI:
    """Create the OpenAI-compatible API application."""
    config = config or XTokensConfig()

    @contextlib.asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        logger.info(
            "serve engine initializing: served_model=%s model=%s "
            "max_model_len=%d max_num_seqs=%d",
            config.model_config.served_model_name,
            config.model_config.model_name,
            config.model_config.max_model_len,
            config.scheduler_config.max_num_seqs,
        )
        engine = LLMEngine(config)
        app.state.engine = engine
        app.state.completion_service = CompletionService(engine)
        app.state.chat_service = ChatCompletionService(engine)

        health = await engine.health()
        if health.ready:
            logger.info("serve engine ready: ready=True")
        else:
            logger.warning("serve engine not ready: %s", health.detail)

        try:
            yield
        finally:
            logger.info("serve engine shutting down")
            await engine.close()
            logger.info("serve engine stopped")

    app = FastAPI(
        title="xTokens Inference Server",
        version="0.1.0",
        description="OpenAI-compatible LLM inference server",
        lifespan=lifespan,
    )
    app.state.config = config
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(config.server_config.cors_origins) or ["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    if config.server_config.api_key:
        app.add_middleware(APIKeyMiddleware, api_key=config.server_config.api_key)

    @app.exception_handler(RequestValidationError)
    async def validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
        errors = exc.errors()
        location = errors[0].get("loc", ()) if errors else ()
        param = str(location[-1]) if location else None
        return JSONResponse(
            status_code=422,
            content=OpenAIError(
                422,
                "Invalid request parameters",
                "invalid_request_error",
                param,
                "validation_error",
            ).body(),
        )

    app.include_router(models.router)
    app.include_router(completions.router)
    app.include_router(chat.router)
    return app
