"""
Simple collate function to pad and batch data samples from a pipeline.
"""

from rfproteina.datapipes.schema.base import BatchedDataSample, DataSample, PaddedBatchedDataSample


def collate_batch(
    batch: list[dict],
    schema: str | type[DataSample],
    fixed_dimensions: dict[str, int | list[int]] | None = None,
) -> PaddedBatchedDataSample:
    """Collate a batch of data samples into a single batch with padding.

    Args:
        batch: A list of data samples (dictionaries from pipeline).
        schema: Either a DataSample class or a string path to one
            (e.g. "rfproteina.datapipes.schema.proteina.ProteinaDataSample").
        fixed_dimensions: Optional dict of dimension names to fixed sizes.
            E.g. {"I": 512, "S": 128} for fixed token and MSA row counts.
            Useful for torch.compile which needs static shapes.
            A value may also be a list of sizes (buckets), e.g. {"I": [256, 512]},
            in which case each batch is padded to the smallest bucket that fits it.
            If None, pads dynamically to max in batch.

    Returns:
        PaddedBatchedDataSample with padding masks.
    """
    # Resolve string schema path to class
    if isinstance(schema, str):
        import importlib

        module_path, class_name = schema.rsplit(".", 1)
        schema = getattr(importlib.import_module(module_path), class_name)

    samples = [schema.from_dict(item) for item in batch]
    batched_samples = BatchedDataSample(samples, default_padding_dimensions=fixed_dimensions)
    return batched_samples.pad_and_stack()
