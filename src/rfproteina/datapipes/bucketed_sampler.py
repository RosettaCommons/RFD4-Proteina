"""Group weighted epoch draws by estimated length without changing the samples."""

from collections.abc import Callable, Iterator
from typing import Any

import numpy as np
from atomworks.ml.samplers import DistributedMixedSampler, MixedSampler
from torch.utils.data import Sampler


def validate_buckets(buckets: list[int]) -> None:
    """Require positive, strictly increasing integer bucket sizes."""
    if not buckets or any(not isinstance(b, int) or isinstance(b, bool) or b <= 0 for b in buckets):
        raise ValueError("Length buckets must be positive integers.")
    if buckets != sorted(set(buckets)):
        raise ValueError("Length buckets must be strictly increasing.")


def read_lengths(data: Any, columns: list[str], name: str, indices: np.ndarray) -> np.ndarray:
    """Sum available count columns on sampled rows; unknown counts use the largest bucket."""
    if not columns or isinstance(columns, str) or any(not isinstance(c, str) or not c for c in columns):
        raise ValueError(f"{name}: length_columns must be a nonempty list of column names.")
    available = data.column_names if hasattr(data, "column_names") else data.columns
    columns = [column for column in columns if column in available]
    if not columns:
        raise ValueError(f"{name}: no configured length columns are available.")
    lengths = np.zeros(len(indices), dtype=np.float64)
    for column in columns:
        values = data[column]
        values = values.iloc[indices] if hasattr(values, "iloc") else values.take(indices)
        values = np.asarray(values.to_numpy() if hasattr(values, "to_numpy") else values)
        if values.dtype.kind not in "iuf" or np.any(np.isinf(values)):
            raise ValueError(f"{name}: {column!r} must contain numeric counts or nulls.")
        known = values[~np.isnan(values)]
        if np.any(known < 0) or np.any(known != np.floor(known)):
            raise ValueError(f"{name}: {column!r} must contain nonnegative integer counts.")
        lengths += values
    return np.where(np.isnan(lengths), np.inf, lengths)


class DistributedBucketedSampler(Sampler[int]):
    """Group global mixed-sampler draws, then give each rank a slice of each batch.

    Metadata controls ordering only. Full buckets are batched first; their tails
    share spill batches so every weighted draw is retained, including duplicates.
    Padding is determined independently by collation from actual sample lengths.
    """

    def __init__(
        self,
        sampler: DistributedMixedSampler,
        lengths_fn: Callable[[np.ndarray], np.ndarray],
        batch_size: int,
        buckets: list[int],
    ) -> None:
        validate_buckets(buckets)
        if batch_size <= 0:
            raise ValueError("Length bucketing requires a positive batch size.")
        self.global_batch_size = sampler.num_replicas * batch_size
        if sampler.total_size <= 0 or sampler.total_size % self.global_batch_size:
            raise ValueError("Bucketed epoch size must be positive and divisible by the global batch size.")
        self.global_sampler = MixedSampler(sampler.datasets_info, sampler.total_size, shuffle=sampler.shuffle)
        self.lengths_fn = lengths_fn
        self.rank = sampler.rank
        self.num_replicas = sampler.num_replicas
        self.batch_size = batch_size
        self.buckets = list(buckets)
        self.epoch = sampler.epoch

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self) -> Iterator[int]:
        self.global_sampler.set_epoch(self.epoch)
        pool = np.asarray(list(self.global_sampler), dtype=np.int64)
        lengths = self.lengths_fn(pool)
        if lengths.shape != pool.shape:
            raise ValueError("Sample lengths must align with the global epoch draws.")
        bucket_ids = np.searchsorted(self.buckets, np.minimum(lengths, self.buckets[-1]))
        batches, tails = [], []
        for bucket in range(len(self.buckets)):
            positions = np.flatnonzero(bucket_ids == bucket)
            stop = len(positions) // self.global_batch_size * self.global_batch_size
            batches.extend(positions[:stop].reshape(-1, self.global_batch_size))
            tails.append(positions[stop:])
        batches.extend(np.concatenate(tails).reshape(-1, self.global_batch_size))
        start = self.rank * self.batch_size
        for i in np.random.default_rng(self.epoch).permutation(len(batches)):
            yield from pool[batches[i][start : start + self.batch_size]].tolist()

    def __len__(self) -> int:
        return len(self.global_sampler) // self.num_replicas
