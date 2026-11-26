from __future__ import annotations

from functools import partial

import pytest

from x_tokens.core import EngineCore
from x_tokens.executor import ModelExecutor


def _run_core_with_test_executor(build_executor, *args) -> None:
    # A spawned child needs its own patch of the factory.
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(
            ModelExecutor, "get_class", staticmethod(lambda _: build_executor)
        )
        EngineCore.run_process(*args)


@pytest.fixture
def install_test_executor(monkeypatch):
    def install(build_executor) -> None:
        monkeypatch.setattr(
            ModelExecutor, "get_class", staticmethod(lambda _: build_executor)
        )
        monkeypatch.setattr(
            EngineCore,
            "run_process",
            staticmethod(partial(_run_core_with_test_executor, build_executor)),
        )

    return install
