"""Composition root for registered xTokens model architectures."""

from __future__ import annotations

from typing import cast

from x_tokens.config import XTokensConfig
from x_tokens.logger import init_logger
from x_tokens.modeling.config import ModelLoadConfig
from x_tokens.modeling.loader import resolve_model_source
from x_tokens.modeling.registry import resolve_model_registration
from x_tokens.worker.model_runner import ModelRunnerFactory

from .model_executor import ModelExecutor, create_model_executor

logger = init_logger(__name__)


def create_registered_model_executor(
    config: XTokensConfig,
    pad_token_id: int,
    eos_token_ids: frozenset[int],
) -> ModelExecutor:
    load_config = ModelLoadConfig(
        config.model_config.model_name,
        local_files_only=config.executor_config.local_files_only,
        dtype=config.executor_config.dtype,
    )
    source = resolve_model_source(load_config)
    resolved = resolve_model_registration(source, config.parallel_config)
    logger.info(
        "resolved model: model=%s source=%s architecture=%s model_type=%s tp=%d",
        load_config.model,
        source.path,
        resolved.registration.architecture,
        resolved.registration.model_type,
        config.parallel_config.tensor_parallel_size,
    )
    runner_factory = cast(
        ModelRunnerFactory,
        resolved.registration.runner_factory_builder(
            source,
            resolved.config,
            load_config,
            pad_token_id,
        ),
    )
    return create_model_executor(
        config.parallel_config,
        runner_factory,
        eos_token_ids=eos_token_ids,
    )
