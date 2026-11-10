"""Generation must not replay the same random trajectory on every DDP rank."""

import json
import random
from pathlib import Path
from types import SimpleNamespace

import lightning as L
import numpy as np
import pytest
import torch
from lightning.pytorch.strategies import DDPStrategy
from torch.utils.data import DataLoader, Dataset

from rfproteina.callbacks.generation_seed import GenerationRankSeedCallback


@pytest.fixture(autouse=True)
def restore_rng(monkeypatch):
    import os

    states = random.getstate(), np.random.get_state(), torch.get_rng_state()
    for key in ("PL_GLOBAL_SEED", "PL_SEED_WORKERS"):
        if key in os.environ:
            monkeypatch.setenv(key, os.environ[key])
        else:
            monkeypatch.delenv(key, raising=False)
    yield
    random.setstate(states[0])
    np.random.set_state(states[1])
    torch.set_rng_state(states[2])


def _draw():
    return random.random(), np.random.random(), torch.rand(8).tolist()


def _rank_stream(job_seed, rank):
    L.seed_everything(job_seed, verbose=False)
    module = SimpleNamespace(inference_seed=job_seed)
    trainer = SimpleNamespace(world_size=8, global_rank=rank, local_rank=0)
    GenerationRankSeedCallback(job_seed).setup(trainer, module, "validate")
    return module.inference_seed, _draw()


def test_rank_streams_are_distinct_and_reproducible():
    first = [_rank_stream(42, rank) for rank in range(8)]
    assert first == [_rank_stream(42, rank) for rank in range(8)]
    assert len({seed for seed, _ in first}) == 8
    assert len({tuple(draws[2]) for _, draws in first}) == 8
    assert len({draws[0] for _, draws in first}) == 8
    assert len({draws[1] for _, draws in first}) == 8


def test_job_and_rank_are_not_added_together():
    assert _rank_stream(42, 1) != _rank_stream(43, 0)
    # Both have local_rank=0: a node-local rank must not select the stream.
    assert _rank_stream(42, 0) != _rank_stream(42, 4)


@pytest.mark.parametrize("stage,world_size", [("validate", 1), ("fit", 4)])
def test_single_process_and_training_rng_are_untouched(stage, world_size):
    L.seed_everything(42, verbose=False)
    _draw()  # Stand in for RNG consumed by model loading before Trainer setup.
    states = random.getstate(), np.random.get_state(), torch.get_rng_state()
    expected = _draw()
    random.setstate(states[0])
    np.random.set_state(states[1])
    torch.set_rng_state(states[2])
    module = SimpleNamespace(inference_seed=42)
    GenerationRankSeedCallback(42).setup(SimpleNamespace(world_size=world_size, global_rank=0), module, stage)
    assert module.inference_seed == 42
    assert _draw() == expected


class RandomInputs(Dataset):
    def __len__(self):
        return 8

    def __getitem__(self, index):
        return {
            "index": index,
            "python": random.random(),
            "numpy": np.random.random(),
            "torch": torch.rand(4),
            "worker_seed": str(torch.initial_seed()),
        }


class GenerationProbe(L.LightningModule):
    """Exercise real Lightning setup, DistributedSampler and worker prefetch."""

    def __init__(self, output_dir):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.rand(4))
        self.output_dir = str(output_dir)
        self.inference_seed = 42
        self.records = []

    def validation_step(self, batch, batch_idx):
        self.records.append(
            {
                "index": batch["index"].item(),
                "python": batch["python"].item(),
                "numpy": batch["numpy"].item(),
                "torch": batch["torch"].tolist(),
                "worker_seed": batch["worker_seed"][0],
                "search_seed": torch.randint(0, 2**31 - 1, (1,)).item(),
            }
        )

    def on_validation_epoch_end(self):
        result = {"seed": self.inference_seed, "weight": self.weight.tolist(), "records": self.records}
        Path(self.output_dir, f"rank{self.global_rank}.json").write_text(json.dumps(result))


@pytest.mark.parametrize("num_workers", [0, 2])
def test_lightning_seeds_before_first_batch_and_worker_prefetch(tmp_path, num_workers):
    results = []
    for repeat in range(2):
        directory = tmp_path / str(repeat)
        directory.mkdir()
        L.seed_everything(42, workers=False, verbose=False)
        model = GenerationProbe(directory)
        initial_weight = model.weight.tolist()
        trainer = L.Trainer(
            accelerator="cpu",
            devices=2,
            strategy=DDPStrategy(start_method="spawn", process_group_backend="gloo"),
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
            enable_model_summary=False,
            callbacks=[GenerationRankSeedCallback(42)],
        )
        trainer.validate(model, DataLoader(RandomInputs(), batch_size=1, num_workers=num_workers), verbose=False)
        ranks = [json.loads((directory / f"rank{rank}.json").read_text()) for rank in range(2)]
        assert ranks[0]["seed"] != ranks[1]["seed"]
        assert ranks[0]["weight"] == ranks[1]["weight"] == initial_weight
        assert len({row["index"] for rank in ranks for row in rank["records"]}) == 8
        assert all(len(rank["records"]) == 4 for rank in ranks)
        for left, right in zip(ranks[0]["records"], ranks[1]["records"], strict=True):
            for key in ("python", "numpy", "torch", "worker_seed", "search_seed"):
                assert left[key] != right[key], (key, num_workers)
        results.append(ranks)
    assert results[0] == results[1]
