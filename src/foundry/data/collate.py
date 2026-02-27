"""
Simple collate function to pad and batch data samples from a pipeline.
"""
from typing import Dict, Optional, Union
from foundry.data.schema.base import DataSample, BatchedDataSample

def collate_batch(
    batch: list[dict],
    schema: Union[str, type[DataSample]],
    fixed_dimensions: Optional[Dict[str, int]] = None
):
    """Collate a batch of data samples into a single batch with padding.

    Args:
        batch: A list of data samples (dictionaries from pipeline).
        schema: Either a DataSample class or a string path to one
            (e.g. "proteinfoundation.datapipes.schema.proteina.ProteinaDataSample").
        fixed_dimensions: Optional dict of dimension names to fixed sizes.
            E.g. {"I": 512, "S": 128} for fixed token and MSA row counts.
            Useful for torch.compile which needs static shapes.
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
    batched_samples = BatchedDataSample(
        samples, default_padding_dimensions=fixed_dimensions
    )
    return batched_samples.pad_and_stack()

