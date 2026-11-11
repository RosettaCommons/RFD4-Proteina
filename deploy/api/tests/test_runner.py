"""Check request isolation and delegation without loading model dependencies."""

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from app.errors import ApiError, ErrorCode
from app.runner import Runner
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from pytest import MonkeyPatch


class FakeTokenLimitError(ValueError):
    def __init__(self, max_tokens: int):
        self.max_tokens = max_tokens
        super().__init__(f"sample exceeds max_tokens={max_tokens}")


class FakeSampleLimitError(ValueError):
    def __init__(self, max_samples: int):
        self.max_samples = max_samples
        super().__init__(f"dataset exceeds max_samples={max_samples}")


class FakeOutOfMemoryError(RuntimeError):
    pass


@pytest.fixture
def pipeline(monkeypatch: MonkeyPatch) -> tuple[Runner, Mock]:
    generate = Mock()
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(OutOfMemoryError=FakeOutOfMemoryError))
    monkeypatch.setitem(
        sys.modules,
        "rfproteina.generate",
        SimpleNamespace(
            run_generation=generate, SampleLimitError=FakeSampleLimitError, TokenLimitError=FakeTokenLimitError
        ),
    )
    runner = Runner.__new__(Runner)
    runner.model = object()
    runner.base = OmegaConf.create({"inputs": None, "out_dir": None, "seed": None, "num_replicates": 1})
    runner.omega = OmegaConf
    return runner, generate


def test_resident_model_uses_upstream_pipeline_and_default_buckets(
    tmp_path: Path, monkeypatch: MonkeyPatch, pipeline: tuple[Runner, Mock]
) -> None:
    root = Path(__file__).resolve().parents[3]
    monkeypatch.setenv("DATA_PATH", str(tmp_path))
    with initialize_config_dir(version_base="1.3", config_dir=str(root / "configs")):
        base = compose(config_name="inference", overrides=["~metrics", "inputs=null", "num_workers=0"])
    assert base.inputs is None
    assert 128 in base.compile_lengths and 512 in base.compile_lengths
    assert base.skip_existing
    calls = []

    def generate(cfg: object, *, model: object, max_tokens: int, max_samples: int) -> None:
        calls.append((cfg, model, max_tokens, max_samples))
        path = Path(cfg.out_dir) / "inference_samples" / "test_generated.cif"
        path.parent.mkdir(parents=True)
        path.write_text("data_generated")

    runner, upstream = pipeline
    runner.base = base
    upstream.side_effect = generate
    for seed in (42, 43):
        output = tmp_path / str(seed) / "outputs"
        result = runner.generate(tmp_path / "input.cif", output, {"seed": seed, "num_replicates": 2}, 1024, 4)
        assert result[0].is_file()
    assert calls[0][0] is not calls[1][0]
    assert [call[0].seed for call in calls] == [42, 43]
    assert all(model is runner.model and limit == 1024 and bound == 4 for _, model, limit, bound in calls)
    assert base.inputs is None and base.seed is None


def test_missing_outputs_are_not_reported_as_success(tmp_path: Path, pipeline: tuple[Runner, Mock]) -> None:
    runner, _ = pipeline
    with pytest.raises(RuntimeError, match="without generated CIF"):
        runner.generate(tmp_path / "input.cif", tmp_path / "outputs", {"seed": 42, "num_replicates": 1}, 1024, 4)


@pytest.mark.parametrize(
    "failure,expected,code",
    [
        (FakeTokenLimitError(512), "This API accepts at most 512 tokens per design", ErrorCode.REQUEST_TOO_LARGE),
        (FakeSampleLimitError(4), "This API accepts at most 4 design samples per request", ErrorCode.REQUEST_TOO_LARGE),
        (
            FakeOutOfMemoryError("CUDA allocation failed"),
            "The design does not fit on the GPU; reduce the input size.",
            ErrorCode.OUT_OF_MEMORY,
        ),
        (ValueError("Unrelated inference failure"), "Unrelated inference failure", None),
    ],
)
def test_admission_errors_are_worded_at_the_api_boundary(
    tmp_path: Path,
    pipeline: tuple[Runner, Mock],
    failure: Exception,
    expected: str,
    code: ErrorCode | None,
) -> None:
    runner, upstream = pipeline
    upstream.side_effect = failure
    with pytest.raises(ApiError if code else ValueError) as exc:
        runner.generate(tmp_path / "input.cif", tmp_path / "outputs", {"seed": 42, "num_replicates": 1}, 1024, 4)
    assert str(exc.value) == expected
    if code:
        assert exc.value.__cause__ is failure
        assert exc.value.code == code
        assert exc.value.status == 413
        assert exc.value.details
    else:
        assert exc.value is failure
