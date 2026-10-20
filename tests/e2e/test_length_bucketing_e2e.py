from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest
import torch
from atomworks.ml.samplers import MixedSampler
from hydra import compose, initialize_config_dir
from torch.utils.data import ConcatDataset, Dataset, WeightedRandomSampler

from rfproteina.datapipes.bucketed_sampler import read_lengths
from rfproteina.datapipes.collate import collate_batch
from rfproteina.utils.datasets import assemble_distributed_loader

LENGTH_COLUMNS = ["n_standard_amino_acid_residues", "n_dna_residues", "n_rna_residues", "n_other_heavy_atoms"]


@pytest.mark.parametrize("root", ["training_ae", "training_proteina"])
def test_length_bucketing_requires_ordered_loading(root):
    """Length bucketing must never be enabled with unordered loading."""
    with initialize_config_dir(config_dir=str(Path(__file__).resolve().parents[2] / "configs"), version_base=None):
        cfg = compose(root)
    train = cfg.dataloaders.train
    assert not train.length_bucketing.enabled or train.dataloader_params.in_order


def test_length_bucketing_end_to_end():
    class Samples(Dataset):
        data = pd.DataFrame(
            [[10, 0, 0, 10], [80, 10, 0, 10], [150, 10, 10, 10], [260, 20, 10, 10]] * 6, columns=LENGTH_COLUMNS
        )
        lengths = [100, 100, 180, 256] * 6

        def __len__(self):
            return len(self.data)

        def __getitem__(self, i):
            if self.bad and i == 0:
                raise ValueError("Injected loading failure")
            return {"example_id": str(i), "feats": {"residue_type": torch.arange(self.lengths[i]) + 1}}

    dataset, weights = Samples(), torch.arange(24, 0, -1, dtype=torch.float64)
    sampler = MixedSampler([{"dataset": dataset, "sampler": WeightedRandomSampler(weights, 96), "probability": 1}])
    schema = "rfproteina.datapipes.schema.proteina.ProteinaDataSample"
    collate = partial(collate_batch, schema=schema, fixed_dimensions={"I": 256})
    cfg = {"dataloader_params": {"batch_size": 4, "num_workers": 0, "collate_fn": collate}, "n_fallback_retries": 32}
    cfg["length_bucketing"] = {"enabled": True, "buckets": [64, 128, 192, 256], "length_columns": LENGTH_COLUMNS}
    loaders = [assemble_distributed_loader(ConcatDataset([dataset]), sampler, r, 2, 96, cfg) for r in range(2)]
    for epoch in range(2):
        for dataset.bad in (False, True):
            weights[1:] = 0 if dataset.bad else torch.arange(23, 0, -1)
            expected = torch.multinomial(weights, 96, True, generator=torch.Generator().manual_seed(epoch)).tolist()
            seen, mixed = [], 0
            for loader in loaders:
                loader.sampler.set_epoch(epoch)
            for batches in zip(*loaders, strict=True):
                indices = [int(i) for batch in batches for i in batch["example_id"]]
                mixed += len({i % 4 for i in indices}) > 1
                seen.extend(indices)
                for batch in batches:
                    ids, values = list(map(int, batch["example_id"])), batch["residue_type"]
                    pad = next(b for b in [64, 128, 192, 256] if b >= max(dataset.lengths[i] for i in ids))
                    assert values.shape == (4, pad)
                    for i, row in zip(ids, values, strict=True):
                        assert torch.equal(row[: dataset.lengths[i]], torch.arange(dataset.lengths[i]) + 1)
            assert len(seen) == 96 and collate.keywords["fixed_dimensions"] == {"I": 256}
            assert (0 not in seen) if dataset.bad else sorted(seen) == sorted(expected) and mixed <= 3


@pytest.mark.parametrize("arrow", [False, True])
@pytest.mark.parametrize("columns", [LENGTH_COLUMNS, LENGTH_COLUMNS[:1], LENGTH_COLUMNS[2:]])
def test_count_lengths(columns, arrow):
    frame = pd.DataFrame({c: pd.array([i + 1, None, 0], dtype="Int64") for i, c in enumerate(columns)})
    data = pa.Table.from_pandas(frame) if arrow else frame
    np.testing.assert_array_equal(
        read_lengths(data, LENGTH_COLUMNS, "sample", np.array([2, 0, 0, 1])),
        [0, sum(range(1, len(columns) + 1)), sum(range(1, len(columns) + 1)), np.inf],
    )
    with pytest.raises(ValueError, match="no configured length columns"):
        read_lengths(data, ["missing"], "sample", np.array([0]))


@pytest.mark.parametrize("value", [-1, 1.5, np.inf, "12", True])
def test_invalid_count_lengths(value):
    with pytest.raises(ValueError, match="counts"):
        read_lengths(pd.DataFrame({"n_atoms": [value]}), ["n_atoms"], "sample", np.array([0]))
