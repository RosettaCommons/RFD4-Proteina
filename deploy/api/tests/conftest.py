"""JSON request fixtures shared by the CPU and live-server checks."""

import queue
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest
from pytest import MonkeyPatch

EXAMPLE = (Path(__file__).resolve().parents[1] / "examples/monomer-short.cif").read_bytes()


def design_request(content: bytes = EXAMPLE, *, task: str = "generation", **sampling: int | None) -> dict[str, Any]:
    return {
        "task": task,
        "input": {
            "format": "annotated_cif",
            "file": {"filename": "input.cif", "content": content.decode("utf-8", errors="surrogateescape")},
        },
        "sampling": sampling,
    }


@pytest.fixture
def isolated_service(tmp_path: Path, monkeypatch: MonkeyPatch) -> Iterator[None]:
    """Restore all service globals after every test, including lifespan assignments."""
    from app import api

    monkeypatch.setenv("STORE_PATH", str(tmp_path / "store"))
    monkeypatch.setattr(api, "store", None)
    monkeypatch.setattr(api, "work", tmp_path / "work")
    monkeypatch.setattr(api, "pending", queue.Queue(maxsize=8))
    monkeypatch.setattr(api, "state", {"ready": False, "error": None, "runner": None, "stop": False})
    yield


def wait_for(predicate: Callable[[], bool]) -> None:
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError("Timed out")
