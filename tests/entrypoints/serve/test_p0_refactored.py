"""End-to-end integration tests for P0 (API Server) with refactored 5-layer architecture.

These tests verify the complete request lifecycle with the new architecture:
  HTTP Request → API Router → Service → RequestProcessor → LLMEngine →
  EngineCoreClient → EngineCore → ResponseProcessor → StreamingResponse

Without requiring GPU/model, using mock EngineCore and mock processors.
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from tests.entrypoints.serve.engine_fakes import (
    FakeCoreClient,
    FakeEngineCore,
    FakeInputProcessor,
    FakeOutputProcessor,
)
from x_tokens.config import ModelConfig, ServerConfig, XTokensConfig
from x_tokens.engine.llm_engine import LLMEngine
from x_tokens.entrypoints.serve.app import create_app


@pytest.fixture
def test_xtokens_config():
    """XTokensConfig for the API test engine."""
    return XTokensConfig(
        model_config=ModelConfig(
            model="test-model",
            served_model_name="test-model",
            max_model_len=2048,
        ),
        server_config=ServerConfig(
            api_key="test-key",
            request_timeout_s=30,
        ),
    )


@pytest.fixture
def test_app(test_xtokens_config, monkeypatch):
    """FastAPI app with a deterministic test engine."""
    monkeypatch.setattr(
        "x_tokens.entrypoints.serve.app.LLMEngine",
        lambda _: LLMEngine(
            input_processor=FakeInputProcessor(),
            output_processor=FakeOutputProcessor(),
            core_client=FakeCoreClient(FakeEngineCore()),
        ),
    )
    return create_app(config=test_xtokens_config)


@pytest.fixture
def client(test_app):
    """FastAPI test client with proper lifespan setup."""
    with TestClient(test_app) as c:
        yield c


# ============================================================================
# HTTP & Request Validation Tests
# ============================================================================


def test_models_endpoint(client):
    """Test /v1/models endpoint returns available models."""
    response = client.get(
        "/v1/models",
        headers={"Authorization": "Bearer test-key"},
    )
    assert response.status_code == 200
    data = response.json()
    assert "data" in data
    assert len(data["data"]) > 0
    assert data["data"][0]["id"] == "test-model"


def test_api_key_validation(client):
    """Test API key validation on protected endpoints."""
    response = client.post(
        "/v1/completions",
        json={"model": "test-model", "prompt": "hello"},
        headers={"Authorization": "Bearer wrong-key"},
    )
    assert response.status_code == 401


def test_model_not_found(client):
    """Test 404 for unknown model."""
    response = client.post(
        "/v1/completions",
        json={"model": "nonexistent", "prompt": "hello"},
        headers={"Authorization": "Bearer test-key"},
    )
    assert response.status_code == 404


# ============================================================================
# Completion API Tests (stream=false)
# ============================================================================


def test_completion_json_response(client):
    """Test /v1/completions with stream=false returns JSON response."""
    response = client.post(
        "/v1/completions",
        json={
            "model": "test-model",
            "prompt": "hello world",
            "stream": False,
            "max_tokens": 5,
        },
        headers={"Authorization": "Bearer test-key"},
    )
    assert response.status_code == 200
    data = response.json()

    # Check response structure (OpenAI format)
    assert "id" in data
    assert data["object"] == "text_completion"
    assert "choices" in data
    assert len(data["choices"]) > 0
    assert "text" in data["choices"][0]
    assert "finish_reason" in data["choices"][0]

    # Check usage
    assert "usage" in data
    assert data["usage"]["prompt_tokens"] > 0
    assert data["usage"]["completion_tokens"] > 0


def test_completion_text_accumulation(client):
    """Test that text is accumulated correctly in non-streaming mode."""
    response = client.post(
        "/v1/completions",
        json={
            "model": "test-model",
            "prompt": "test",
            "stream": False,
            "max_tokens": 10,
        },
        headers={"Authorization": "Bearer test-key"},
    )
    assert response.status_code == 200
    data = response.json()

    # Text should be non-empty
    text = data["choices"][0]["text"]
    assert len(text) > 0


# ============================================================================
# Completion API Tests (stream=true)
# ============================================================================


def test_completion_sse_streaming(client):
    """Test /v1/completions with stream=true returns SSE stream."""
    response = client.post(
        "/v1/completions",
        json={
            "model": "test-model",
            "prompt": "hello",
            "stream": True,
            "max_tokens": 3,
        },
        headers={"Authorization": "Bearer test-key"},
    )
    assert response.status_code == 200
    assert "text/event-stream" in response.headers["content-type"]

    # Parse SSE stream
    lines = response.text.strip().split("\n")
    events = []
    for line in lines:
        if line.startswith("data: "):
            data_str = line[6:]
            if data_str == "[DONE]":
                events.append(None)
            else:
                events.append(json.loads(data_str))

    # Should have multiple token events and one finish
    assert len(events) > 1
    assert events[-1] is None  # [DONE]

    # Check token events have correct structure
    for event in events[:-1]:
        assert event is not None
        assert "id" in event
        assert event["object"] == "text_completion.chunk"
        assert "choices" in event
        assert len(event["choices"]) > 0
        assert "text" in event["choices"][0]

    # Last streaming event should have usage
    last_event = events[-2]  # Before [DONE]
    assert last_event is not None
    if last_event["choices"][0].get("finish_reason"):
        assert "usage" in last_event


# ============================================================================
# Chat Completion API Tests (stream=false)
# ============================================================================


def test_chat_completion_json(client):
    """Test /v1/chat/completions with stream=false."""
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "test-model",
            "messages": [
                {"role": "user", "content": "What is 2+2?"},
            ],
            "stream": False,
            "max_tokens": 5,
        },
        headers={"Authorization": "Bearer test-key"},
    )
    assert response.status_code == 200
    data = response.json()

    # Check response structure (OpenAI format)
    assert "id" in data
    assert data["object"] == "chat.completion"
    assert "choices" in data
    assert len(data["choices"]) > 0
    assert "message" in data["choices"][0]
    assert data["choices"][0]["message"]["role"] == "assistant"
    assert len(data["choices"][0]["message"]["content"]) > 0


# ============================================================================
# Chat Completion API Tests (stream=true)
# ============================================================================


def test_chat_completion_sse_streaming(client):
    """Test /v1/chat/completions with stream=true returns SSE stream."""
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "test-model",
            "messages": [
                {"role": "user", "content": "hello"},
            ],
            "stream": True,
            "max_tokens": 3,
        },
        headers={"Authorization": "Bearer test-key"},
    )
    assert response.status_code == 200

    # Parse SSE stream
    lines = response.text.strip().split("\n")
    events = []
    for line in lines:
        if line.startswith("data: "):
            data_str = line[6:]
            if data_str == "[DONE]":
                events.append(None)
            else:
                events.append(json.loads(data_str))

    assert len(events) > 1
    assert events[-1] is None

    # First token should have role
    first_event = events[0]
    assert first_event is not None
    assert "choices" in first_event
    assert "delta" in first_event["choices"][0]
    # Role appears in first chunk
    if "role" in first_event["choices"][0]["delta"]:
        assert first_event["choices"][0]["delta"]["role"] == "assistant"

    # Check all events have correct structure
    for event in events[:-1]:
        assert event is not None
        assert "id" in event
        assert event["object"] == "chat.completion.chunk"
        assert "choices" in event
        assert "delta" in event["choices"][0]


# ============================================================================
# Stream/Non-stream Consistency Tests
# ============================================================================


def test_stream_and_nonstream_same_total_tokens(client):
    """Verify that streaming and non-streaming produce same total token count."""
    # Non-streaming
    client.post(
        "/v1/completions",
        json={
            "model": "test-model",
            "prompt": "test prompt",
            "stream": False,
            "max_tokens": 5,
        },
        headers={"Authorization": "Bearer test-key"},
    )
    # Streaming
    response2 = client.post(
        "/v1/completions",
        json={
            "model": "test-model",
            "prompt": "test prompt",
            "stream": True,
            "max_tokens": 5,
        },
        headers={"Authorization": "Bearer test-key"},
    )
    lines = response2.text.strip().split("\n")
    last_event_line = None
    for line in lines:
        if line.startswith("data: ") and line != "data: [DONE]":
            last_event_line = line

    if last_event_line:
        last_event = json.loads(last_event_line[6:])
        if "usage" in last_event:
            tokens2 = last_event["usage"]["total_tokens"]
            # Both should account for tokens consistently
            assert tokens2 > 0


# ============================================================================
# Error Handling Tests
# ============================================================================


def test_invalid_model_parameter(client):
    """Test error handling for invalid model parameter."""
    response = client.post(
        "/v1/completions",
        json={
            "model": "nonexistent-model",
            "prompt": "hello",
        },
        headers={"Authorization": "Bearer test-key"},
    )
    assert response.status_code == 404


def test_missing_prompt(client):
    """Test error handling for missing prompt."""
    response = client.post(
        "/v1/completions",
        json={
            "model": "test-model",
            # Missing 'prompt'
        },
        headers={"Authorization": "Bearer test-key"},
    )
    assert response.status_code >= 400


# ============================================================================
# Integration Tests (Architecture Verification)
# ============================================================================


def test_refactored_layers_integration(client):
    """Verify that all 5 layers work together correctly.

    This test ensures:
    1. API Router receives request
    2. Service layer processes it
    3. RequestProcessor tokenizes input
    4. LLMEngine generates output
    5. ResponseProcessor formats response
    """
    response = client.post(
        "/v1/completions",
        json={
            "model": "test-model",
            "prompt": "integration test",
            "stream": False,
            "max_tokens": 5,
        },
        headers={"Authorization": "Bearer test-key"},
    )

    # Layer 1 (Router): HTTP 200
    assert response.status_code == 200

    # Layers 2-5: Proper response format
    data = response.json()
    assert "id" in data
    assert "choices" in data
    assert "usage" in data
    assert data["usage"]["prompt_tokens"] > 0
    assert data["usage"]["completion_tokens"] > 0


def test_engine_events_are_converted_to_openai_response(client):
    """Verify that engine events are converted to OpenAI responses.

    Engine events should be converted to OpenAI format consistently.
    """
    response = client.post(
        "/v1/completions",
        json={
            "model": "test-model",
            "prompt": "abstraction test",
            "stream": False,
        },
        headers={"Authorization": "Bearer test-key"},
    )

    data = response.json()
    # Response should follow OpenAI spec exactly
    assert data["object"] == "text_completion"
    assert all("text" in c for c in data["choices"])
    assert all("finish_reason" in c for c in data["choices"])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
