"""GPU e2e: SOTA training smokes (muon × compile/no-compile × RF3/AF3 oracle).

Each variant trains ``experiment=sota`` for a small number of optimizer steps plus
a short epoch-0 validation pass on 2 GPUs (batch size 2) and asserts train loss
decreases. Variants: ``muon_nocompile_rf3``, ``muon_compile_af3``.
Gated by ``RUN_GPU_E2E_TESTS=1`` and >= 2 GPUs.
"""

from __future__ import annotations

import pytest

from tests.e2e.helpers.training_smoke import SOTA_TRAIN_VARIANTS, TrainVariant, run_sota_training_smoke


@pytest.mark.gpu
@pytest.mark.e2e
@pytest.mark.training_smoke
@pytest.mark.slow
@pytest.mark.parametrize("variant", SOTA_TRAIN_VARIANTS, ids=lambda v: v.name)
def test_sota_training_smoke_2gpu(variant: TrainVariant) -> None:
    run_sota_training_smoke(variant)
