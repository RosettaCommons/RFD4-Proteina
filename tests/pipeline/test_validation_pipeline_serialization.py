"""Validation pipeline serialization for spawned DataLoader workers."""

import pickle
from multiprocessing.reduction import ForkingPickler

from rfproteina.datapipes.pipeline import build_design_validation_pipeline


def test_validation_pipeline_serialization():
    """Ensure spawned workers can deserialize the complete validation pipeline."""
    pipeline = build_design_validation_pipeline()
    restored = pickle.loads(ForkingPickler.dumps(pipeline))
    assert type(restored) is type(pipeline)
