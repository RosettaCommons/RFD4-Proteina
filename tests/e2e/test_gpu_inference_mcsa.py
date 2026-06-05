"""GPU e2e: offline MCSA validation + minimal beam search via the SOTA ckpt.

Checkpoint-gated: set ``SOTA_CKPT_PATH=/abs/file.ckpt`` (or ``CKPT_PATH`` +
``SOTA_CKPT_NAME``) to run; otherwise skipped.
"""

from __future__ import annotations

import pytest

from tests.e2e.helpers.inference_smoke import run_beam_search_smoke, run_mcsa_validation_smoke


@pytest.mark.gpu
@pytest.mark.e2e
@pytest.mark.slow
def test_mcsa_offline_validation_2gpu() -> None:
    run_mcsa_validation_smoke("pytest_sota_mcsa")


@pytest.mark.gpu
@pytest.mark.e2e
@pytest.mark.slow
def test_beam_search_2gpu() -> None:
    run_beam_search_smoke("pytest_sota_beam")
