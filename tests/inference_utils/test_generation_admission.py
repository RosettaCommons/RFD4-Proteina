"""Admission must reject transformed inputs before DataLoader or Lightning sees them."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from atomworks.ml.utils import token
from omegaconf import OmegaConf

import rfproteina.generate as generate


@pytest.mark.parametrize("limit", [None, 512])
@pytest.mark.parametrize("count", [512, 513])
def test_generation_token_admission_preserves_cli_and_compilation(tmp_path, monkeypatch, limit, count) -> None:
    cfg = OmegaConf.create(
        {
            "precision": "32",
            "out_dir": str(tmp_path / "outputs"),
            "seed": 42,
            "num_workers": 0,
            "compile_lengths": [128, 512],
            "skip_existing": False,
            "hardware": {"ngpus_per_node_": 1, "nnodes_": 1},
        }
    )
    sample = {"atom_array": object()}
    dataset = [sample]
    model = SimpleNamespace(reserve_shape_cache=Mock())
    loader = Mock(return_value=[sample])
    trainer = SimpleNamespace(validate=Mock(), is_global_zero=True)
    trainer_factory = Mock(return_value=trainer)
    save_cache = Mock()
    setup_compile = Mock()
    monkeypatch.setattr(generate, "load_ckpt_n_configure_inference", lambda cfg, *, model: model)
    monkeypatch.setattr(generate, "get_dataset_from_cfg", lambda cfg, job_id: dataset)
    monkeypatch.setattr(generate, "_sort_dataset_by_length", lambda ds: ds)
    monkeypatch.setattr(generate, "inference_backend", lambda cfg: "native")
    monkeypatch.setattr(generate, "setup_run_compile", setup_compile)
    monkeypatch.setattr(generate, "save_compile_cache", save_cache)
    monkeypatch.setattr(generate, "DataLoader", loader)
    monkeypatch.setattr(generate.L, "Trainer", trainer_factory)
    monkeypatch.setattr(token, "get_token_count", lambda aa: count)
    if limit is not None and count > limit:
        with pytest.raises(ValueError, match="sample exceeds max_tokens=512"):
            generate.run_generation(cfg, model=model, max_tokens=limit, max_samples=4 if limit is not None else None)
        loader.assert_not_called()
        trainer_factory.assert_not_called()
        setup_compile.assert_not_called()
        model.reserve_shape_cache.assert_not_called()
    else:
        generate.run_generation(cfg, model=model, max_tokens=limit, max_samples=4 if limit is not None else None)
        assert loader.call_args.kwargs["collate_fn"].keywords["fixed_dimensions"] == {"I": [128, 512]}
        assert loader.call_args.args[0][0] is sample
        trainer.validate.assert_called_once_with(model, [sample])
        save_cache.assert_called_once()
        model.reserve_shape_cache.assert_called_once_with([128, 512])


@pytest.mark.parametrize("workers,bound", [(0, None), (0, 0), (2, 4)])
def test_prevalidation_requires_explicit_bounded_single_process_mode(tmp_path, monkeypatch, workers, bound) -> None:
    cfg = OmegaConf.create({"num_workers": workers})
    load = Mock()
    monkeypatch.setattr(generate, "load_ckpt_n_configure_inference", load)
    with pytest.raises(ValueError, match="max_samples|num_workers=0"):
        generate.run_generation(cfg, max_tokens=512, max_samples=bound)
    load.assert_not_called()


def test_dataset_bound_is_checked_before_transforming_samples(tmp_path, monkeypatch) -> None:
    cfg = OmegaConf.create(
        {
            "precision": "32",
            "out_dir": str(tmp_path / "outputs"),
            "seed": 42,
            "num_workers": 0,
            "compile_lengths": [128, 512],
        }
    )

    class TooManySamples:
        def __len__(self):
            return 5

        def __iter__(self):
            raise AssertionError("Must reject before transforming samples")

    model = SimpleNamespace()
    setup_compile = Mock()
    sort_dataset = Mock(side_effect=AssertionError("Must reject before sorting index lists"))
    monkeypatch.setattr(generate, "load_ckpt_n_configure_inference", lambda cfg, *, model: model)
    monkeypatch.setattr(generate, "get_dataset_from_cfg", lambda cfg, job_id: TooManySamples())
    monkeypatch.setattr(generate, "inference_backend", lambda cfg: "native")
    monkeypatch.setattr(generate, "setup_run_compile", setup_compile)
    monkeypatch.setattr(generate, "_sort_dataset_by_length", sort_dataset)
    with pytest.raises(ValueError, match="dataset exceeds max_samples=4"):
        generate.run_generation(cfg, model=model, max_tokens=512, max_samples=4)
    setup_compile.assert_not_called()
    sort_dataset.assert_not_called()
