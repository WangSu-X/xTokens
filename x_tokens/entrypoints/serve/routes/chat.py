"""Chat Completions API routes."""

from __future__ import annotations

import uuid

from fastapi import APIRouter, HTTPException, Request
from starlette.responses import JSONResponse, StreamingResponse

from x_tokens.entrypoints.serve.openai.protocol import (
    ChatCompletionRequest,
    ChatCompletionResponse,
)
from x_tokens.entrypoints.serve.services.chat_service import ChatCompletionService

router = APIRouter()


@router.post("/v1/chat/completions", response_model=None)
async def create_chat_completion(
    request: ChatCompletionRequest, raw_request: Request
) -> JSONResponse | StreamingResponse:
    """Create a chat completion as JSON or an SSE stream."""
    state = raw_request.app.state
    served_model = state.config.model_config.served_model_name
    if request.model != served_model:
        raise HTTPException(
            status_code=404, detail=f"Model '{request.model}' not found"
        )
    health = await state.engine.health()
    if not health.ready:
        raise HTTPException(status_code=503, detail="The engine is not ready")

    request_id = raw_request.headers.get("x-request-id") or (
        f"chatcmpl-{uuid.uuid4().hex[:24]}"
    )
    service: ChatCompletionService = state.chat_service
    result = await service.create_chat_completion(request, request_id=request_id)
    headers = {"X-Request-ID": request_id}

    if isinstance(result, ChatCompletionResponse):
        return JSONResponse(result.model_dump(), headers=headers)
    return StreamingResponse(result, media_type="text/event-stream", headers=headers)
