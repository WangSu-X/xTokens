"""End-to-end integration tests for P0 (API Server) with mock EngineCore.

These tests verify the complete request lifecycle:
  HTTP request → tokenization → mock core → de-tokenization → HTTP response

Without requiring GPU/model, using mock EngineCore and real tokenizer.
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from tests.entrypoints.serve.engine_fakes import (
    FakeCoreClient,
    FakeEngineConfig,
    FakeEngineCore,
)
from x_tokens.config import ModelConfig, ServerConfig, XTokensConfig
from x_tokens.engine.input_processor import TokenizerInputProcessor
from x_tokens.engine.llm_engine import LLMEngine
from x_tokens.engine.output_processor import TokenizerOutputProcessor
from x_tokens.engine.protocol import FinishReason, GenerateRequest
from x_tokens.entrypoints.serve import app as serve_app
from x_tokens.entrypoints.serve.app import create_app

TEST_TOKENIZER_PATH = os.getenv(
    "XTOKENS_TEST_TOKENIZER", "/root/huggingface/Qwen3.8-27B-FP8"
)


@pytest.fixture
def qwen_tokenizer():
    """Load the local Qwen3.8-27B tokenizer for offline tests."""
    if not Path(TEST_TOKENIZER_PATH).exists():
        pytest.skip("Tokenizer not available")
    return TokenizerInputProcessor.from_config(
        TEST_TOKENIZER_PATH, local_files_only=True
    )


@pytest.fixture
def mock_config():
    """Config for mock engine."""
    return FakeEngineConfig()


@pytest.fixture
def mock_core(mock_config):
    """Mock EngineCore instance."""
    return FakeEngineCore(mock_config)


@pytest.fixture
def mock_engine(qwen_tokenizer, mock_core):
    """Complete mock LLMEngine with real tokenizer and mock core."""
    engine = LLMEngine(
        input_processor=qwen_tokenizer,
        output_processor=TokenizerOutputProcessor(qwen_tokenizer.tokenizer),
        core_client=FakeCoreClient(mock_core),
    )
    return engine


@pytest.fixture
def mock_xtokens_config():
    """XTokensConfig for mock engine."""
    return XTokensConfig(
        model_config=ModelConfig(
            model="gpt2",
            served_model_name="test-model",
            max_model_len=2048,
        ),
        server_config=ServerConfig(
            api_key="test-key",
            request_timeout_s=30,
        ),
    )


@pytest.fixture
def test_app(mock_xtokens_config, mock_engine, monkeypatch):
    """FastAPI app with deterministic test engine."""
    monkeypatch.setattr(serve_app, "LLMEngine", lambda _: mock_engine)
    return create_app(config=mock_xtokens_config)


@pytest.fixture
def client(test_app):
    """FastAPI test client with proper lifespan setup."""
    # TestClient doesn't automatically run lifespan, so we need to manually enter it
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
# End-to-End Completion Tests
# ============================================================================


def test_completion_json_response(client):
    """Test /v1/completions with stream=false returns JSON."""
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

    # Check response structure
    assert "choices" in data
    assert len(data["choices"]) > 0
    assert "text" in data["choices"][0]
    assert "finish_reason" in data["choices"][0]

    # Check usage
    assert "usage" in data
    assert data["usage"]["prompt_tokens"] > 0
    assert data["usage"]["completion_tokens"] > 0


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

    # Check token events
    for event in events[:-1]:
        assert event is not None
        assert "choices" in event
        assert len(event["choices"]) > 0


# ============================================================================
# End-to-End Chat Completion Tests
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

    assert "choices" in data
    assert len(data["choices"]) > 0
    assert "message" in data["choices"][0]
    assert data["choices"][0]["message"]["role"] == "assistant"
    assert len(data["choices"][0]["message"]["content"]) > 0


def test_chat_completion_sse_streaming(client):
    """Test /v1/chat/completions with stream=true."""
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
    assert first_event["choices"][0]["delta"].get("role") == "assistant"


# ============================================================================
# Tokenization & De-tokenization Tests
# ============================================================================


def test_tokenization_pipeline(qwen_tokenizer):
    """Test that text is properly tokenized."""
    request = GenerateRequest(
        request_id="test-1",
        model="test-model",
        prompt="hello world",
        sampling=None,
    )

    processed = qwen_tokenizer.process(request)

    # Should have token IDs
    assert isinstance(processed.prompt, tuple)
    assert len(processed.prompt) > 0
    assert all(isinstance(t, int) for t in processed.prompt)

    # Token IDs should be reasonable (not too large)
    assert all(0 <= t < 50000 for t in processed.prompt)


def test_detokenization_incremental(qwen_tokenizer):
    """Test incremental de-tokenization."""
    output_proc = TokenizerOutputProcessor(qwen_tokenizer.tokenizer)

    # Simulate generating tokens
    request_id = "test-1"
    tokens = [1, 2, 3, 4, 5]  # Mock token IDs

    texts = []
    for token_id in tokens:
        text = output_proc.process_token(request_id, token_id)
        texts.append(text)

    output_proc.finish(request_id)

    # Should have accumulated text
    full_text = "".join(texts)
    assert len(full_text) > 0


# ============================================================================
# Event Flow & Request Lifecycle Tests
# ============================================================================


# ============================================================================
# Event Flow & Request Lifecycle Tests
# ============================================================================


def test_request_lifecycle_events(mock_engine):
    """Test that request goes through proper lifecycle: add → tokens → finish."""

    request = GenerateRequest(
        request_id="test-lifecycle",
        model="test-model",
        prompt="test",
        sampling=None,
    )

    async def run():
        events = []
        async for event in mock_engine.generate(request):
            events.append(event)
        return events

    events = asyncio.run(run())

    # Should have token events and finish event
    assert len(events) > 0

    # Last event should be finished
    from x_tokens.engine.protocol import FinishedEvent
    assert isinstance(events[-1], FinishedEvent)
    assert events[-1].finish_reason in (FinishReason.LENGTH, FinishReason.STOP)


def test_multiple_concurrent_requests(mock_engine):
    """Test handling multiple concurrent requests."""

    requests = [
        GenerateRequest(
            request_id=f"test-{i}",
            model="test-model",
            prompt=f"prompt {i}",
            sampling=None,
        )
        for i in range(3)
    ]

    async def run():
        # Generate for all requests concurrently
        results = await asyncio.gather(
            *[
                asyncio.create_task(_collect_events(mock_engine.generate(req)))
                for req in requests
            ]
        )
        return results

    results = asyncio.run(run())

    # Each should have completed
    for events in results:
        assert len(events) > 0
        assert all(e.request_id in [f"test-{i}" for i in range(3)] for e in events)


async def _collect_events(async_gen):
    """Helper to collect all events from async generator."""
    events = []
    async for event in async_gen:
        events.append(event)
    return events


# ============================================================================
# Error Handling Tests
# ============================================================================


def test_core_error_is_returned_as_sse(mock_xtokens_config, qwen_tokenizer, monkeypatch):
    request_id = "error-request"
    core = FakeEngineCore(FakeEngineConfig(error_on_request_id=request_id))
    processor = qwen_tokenizer
    engine = LLMEngine(
        input_processor=processor,
        output_processor=TokenizerOutputProcessor(processor.tokenizer),
        core_client=FakeCoreClient(core),
    )

    monkeypatch.setattr(serve_app, "LLMEngine", lambda _: engine)
    app = create_app(config=mock_xtokens_config)

    with TestClient(app) as client:
        response = client.post(
            "/v1/completions",
            json={"model": "test-model", "prompt": "hello", "stream": True},
            headers={"Authorization": "Bearer test-key", "X-Request-ID": request_id},
        )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert response.text == (
        'data: {"error":{"message":"fake core error"}}\n\n'
        "data: [DONE]\n\n"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
