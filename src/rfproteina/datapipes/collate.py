"""
Simple collate function to pad and batch data samples from a pipeline.
"""

from rfproteina.datapipes.schema.base import BatchedDataSample, DataSample, PaddedBatchedDataSample


def collate_batch(
    batch: list[dict],
    schema: str | type[DataSample],
    fixed_dimensions: dict[str, int | list[int]] | None = None,
    keep_atom_array: bool = True,
    num_replicates: int = 1,
) -> PaddedBatchedDataSample:
    """Collate a batch of data samples into a single batch with padding.

    Args:
        batch: A list of data samples (dictionaries from pipeline).
        schema: Either a DataSample class or a string path to one
            (e.g. "rfproteina.datapipes.schema.proteina.ProteinaDataSample").
        fixed_dimensions: Optional dict of dimension names to fixed sizes.
            E.g. {"I": 512} for a fixed token count.
            Useful for torch.compile which needs static shapes.
            A value may also be a list of sizes (buckets), e.g. {"I": [256, 512]},
            in which case each batch is padded to the smallest bucket that fits it.
            If None, pads dynamically to max in batch.
        keep_atom_array: Retain structures for validation, inference or structure-aware callbacks.
        num_replicates: Number of adjacent copies of each processed sample.

    Returns:
        PaddedBatchedDataSample with padding masks.
    """
    if not keep_atom_array:
        batch = [{**sample, "atom_array": None} for sample in batch]

    # Resolve string schema path to class
    if isinstance(schema, str):
        import importlib

        module_path, class_name = schema.rsplit(".", 1)
        schema = getattr(importlib.import_module(module_path), class_name)

    if num_replicates < 1:
        raise ValueError(f"num_replicates must be at least 1, got {num_replicates}")
    samples = [schema.from_dict(item) for item in batch]
    if num_replicates > 1:
        samples = [sample for sample in samples for _ in range(num_replicates)]
    batched_samples = BatchedDataSample(samples, default_padding_dimensions=fixed_dimensions)
    return batched_samples.pad_and_stack()
