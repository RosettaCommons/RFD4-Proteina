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
    monkeypatch.setattr(generate, "load_ckpt_n_configure_inference", lambda cfg, *, model: model)
    monkeypatch.setattr(generate, "get_dataset_from_cfg", lambda cfg, job_id: dataset)
    monkeypatch.setattr(generate, "_sort_dataset_by_length", lambda ds: ds)
    monkeypatch.setattr(generate, "inference_backend", lambda cfg: "native")
    monkeypatch.setattr(generate, "setup_run_compile", Mock())
    monkeypatch.setattr(generate, "save_compile_cache", save_cache)
    monkeypatch.setattr(generate, "DataLoader", loader)
    monkeypatch.setattr(generate.L, "Trainer", trainer_factory)
    monkeypatch.setattr(token, "get_token_count", lambda aa: count)
    if limit is not None and count > limit:
        with pytest.raises(ValueError, match="512 tokens"):
            generate.run_generation(cfg, model=model, max_tokens=limit)
        loader.assert_not_called()
        trainer_factory.assert_not_called()
    else:
        generate.run_generation(cfg, model=model, max_tokens=limit)
        assert loader.call_args.kwargs["collate_fn"].keywords["fixed_dimensions"] == {"I": [128, 512]}
        assert loader.call_args.args[0][0] is sample
        trainer.validate.assert_called_once_with(model, [sample])
        save_cache.assert_called_once()
    model.reserve_shape_cache.assert_called_once_with([128, 512])
