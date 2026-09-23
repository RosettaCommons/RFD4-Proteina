"""Compare FP32 BioIR against native SOTA inference without a stored baseline.

Run on an allocated GPU with compatible BioIR installed:
RUN_GPU_E2E_TESTS=1 pytest tests/e2e/test_bioir_regression.py -v
"""

import copy
from collections.abc import Callable, Iterator

import lightning as L
import numpy as np
import pytest
import torch
from biotite.structure import AtomArray
from hydra import compose, initialize_config_dir
from lightning.fabric.utilities.apply_func import move_data_to_device
from omegaconf import DictConfig

from rfproteina.datapipes.collate import collate_batch
from rfproteina.generate import get_dataset_from_cfg, load_ckpt_n_configure_inference
from rfproteina.proteina import Proteina
from rfproteina.utils.hydra import register_resolvers
from tests.e2e.helpers.gpu_e2e import require_gpu_e2e
from tests.e2e.helpers.regression_baselines import require_sota_ae_ckpt, require_sota_ckpt
from tests.helpers import PROJECT_ROOT

pytestmark = [pytest.mark.gpu, pytest.mark.regression, pytest.mark.slow]


def _generate(model: Proteina, batch: dict) -> AtomArray:
    L.seed_everything(42, workers=False)
    with torch.inference_mode(), torch.autocast("cuda", enabled=False):
        return model.predict_step(copy.deepcopy(batch), 0)[0]["generated_atom_array"]


@pytest.fixture(scope="module")
def reference(tmp_path_factory: pytest.TempPathFactory) -> Iterator[tuple[DictConfig, dict, AtomArray]]:
    require_gpu_e2e(min_gpus=1)
    pytest.importorskip("bionemo_ir.blocks", reason="requires the local BioIR reusable-block API")
    require_sota_ckpt()
    require_sota_ae_ckpt()
    register_resolvers()
    with initialize_config_dir(config_dir=str(PROJECT_ROOT / "configs"), version_base=None):
        cfg = compose(
            config_name="inference",
            overrides=[
                f"inputs={PROJECT_ROOT}/data/benchmarks/monomer/monomer-short.cif",
                f"out_dir={tmp_path_factory.mktemp('bioir-sota')}",
                "~metrics",
                "execution.inference.length_buckets=null",
                "execution.inference.flow=eager",
                "execution.inference.encoder=eager",
                "execution.inference.decoder=eager",
                "num_workers=0",
            ],
        )
    previous_precision = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("highest")
    try:
        L.seed_everything(42, workers=False)
        model = load_ckpt_n_configure_inference(cfg).cuda().eval()
        dataset = get_dataset_from_cfg(cfg, job_id=0)
        batch = move_data_to_device(
            collate_batch(
                [dataset[0]],
                schema="rfproteina.datapipes.schema.proteina.ProteinaDataSample",
                fixed_dimensions={"I": [128]},
            ),
            "cuda",
        )
        assert batch["padding_mask"].any() and (~batch["padding_mask"]).any()
        expected = _generate(model, batch)
        del model
        torch.cuda.empty_cache()
        yield cfg, batch, expected
    finally:
        torch.set_float32_matmul_precision(previous_precision)


def test_sota_bioir_structure_equivalence(
    reference: tuple[DictConfig, dict, AtomArray],
    record_property: Callable[[str, object], None],
) -> None:
    """Full default trajectory preserves sequence and sub-Angstrom atom coordinates.

    FP32/SDPA permits 0.1 Å all-atom RMSD and 0.5 Å maximum displacement, without alignment.
    """
    from bionemo_ir.blocks import ConditionedTransformerBlock

    cfg, batch, expected = reference
    cfg = copy.deepcopy(cfg)
    for component in ("flow", "encoder", "decoder"):
        cfg.execution.inference[component] = "bioir"
    cfg.execution.bioir.dtype = "float32"
    cfg.execution.bioir.attention_backend = "SDPA"
    model = load_ckpt_n_configure_inference(cfg).cuda().eval()
    try:
        assert all(
            getattr(trunk, "bioir_enabled", False)
            for trunk in (model.nn, model.autoencoder.encoder, model.autoencoder.decoder)
        )
        observed = _generate(model, batch)
        runtime_modules = [
            m
            for layers in model.nn._execution.converted
            for block in layers
            if block is not None
            for m in block.modules()
        ]
        assert any(isinstance(module, ConditionedTransformerBlock) for module in runtime_modules)
        for field in ("chain_id", "res_id", "res_name", "atom_name", "element"):
            np.testing.assert_array_equal(getattr(observed, field), getattr(expected, field), err_msg=field)
        assert np.isfinite(observed.coord).all()
        distance = np.linalg.norm(observed.coord.astype(np.float64) - expected.coord.astype(np.float64), axis=-1)
        rmsd = float(np.sqrt(np.mean(distance**2)))
        maximum = float(distance.max())
        record_property("all_atom_rmsd_angstrom", rmsd)
        record_property("max_atom_distance_angstrom", maximum)
        assert rmsd <= 0.1, f"All-atom RMSD {rmsd:.6f} Å exceeds 0.1 Å"
        assert maximum <= 0.5, f"Maximum atom displacement {maximum:.6f} Å exceeds 0.5 Å"
        graphs = [module for module in runtime_modules if hasattr(module, "graph_state_by_key")]
        assert graphs and all(
            any(state.graph is not None for state in graph.graph_state_by_key.values()) for graph in graphs
        )
    finally:
        del model
        torch.cuda.empty_cache()
