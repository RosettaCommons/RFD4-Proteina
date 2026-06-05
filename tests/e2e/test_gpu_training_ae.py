"""GPU e2e: AE training smokes (muon/adam x compile/no-compile).

Each variant trains ``experiment_ae=smoke`` for 10 optimizer steps + 2 sanity
validation steps on 2 GPUs (batch size 2) and asserts train loss stays finite
and does not explode. Variants: ``ae_muon_nocompile``, ``ae_adam_nocompile``,
``ae_muon_compile``, ``ae_adam_compile``.
Gated by ``RUN_GPU_E2E_TESTS=1`` and >= 2 GPUs.
"""

from __future__ import annotations

import pytest

from tests.e2e.helpers.ae_training_smoke import AE_TRAIN_VARIANTS, AeTrainVariant, run_ae_training_smoke


@pytest.mark.gpu
@pytest.mark.e2e
@pytest.mark.training_smoke
@pytest.mark.slow
@pytest.mark.parametrize("variant", AE_TRAIN_VARIANTS, ids=lambda v: v.name)
def test_ae_training_smoke_2gpu(variant: AeTrainVariant) -> None:
    run_ae_training_smoke(variant)
