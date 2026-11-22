"""Authentication middleware for API Server."""

from __future__ import annotations

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import JSONResponse, Response
from starlette.types import ASGIApp


class APIKeyMiddleware(BaseHTTPMiddleware):
    """Require a bearer API key except on the readiness endpoint."""

    def __init__(self, app: ASGIApp, *, api_key: str) -> None:
        super().__init__(app)
        self.authorization = f"Bearer {api_key}"

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        if request.url.path == "/ready":
            return await call_next(request)
        if request.headers.get("Authorization") != self.authorization:
            return JSONResponse({"error": "Unauthorized"}, status_code=401)
        return await call_next(request)
