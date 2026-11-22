"""Models API routes."""

from __future__ import annotations

from fastapi import APIRouter, Request
from starlette.responses import JSONResponse

router = APIRouter()


@router.get("/ready")
async def readiness(request: Request) -> JSONResponse:
    health = await request.app.state.engine.health()
    return JSONResponse(
        {"ready": health.ready}, status_code=200 if health.ready else 503
    )


@router.get("/v1/models")
async def list_models(request: Request) -> JSONResponse:
    """List the configured model."""
    served_model_name = request.app.state.config.model_config.served_model_name

    return JSONResponse(
        {
            "object": "list",
            "data": [
                {
                    "id": served_model_name,
                    "object": "model",
                    "created": 0,
                    "owned_by": "x-tokens",
                }
            ],
        }
    )
