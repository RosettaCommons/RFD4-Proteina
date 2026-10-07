"""The design workflows the BioIR backend covers, and the rollout it runs them under.

``inference_backend=bioir`` converts the whole model, so a workflow either runs
entirely through BioIR or not at all. These tests pin which workflows that is,
which BioIR entry point each one reaches, and that ``thinking_medium`` arrives
at BioIR's VF-TSR sampler as the schedule the checkpoint was released with.

The rollout test is CPU-only on purpose: it bridges a composed native config
rather than loading weights, so a schedule or step-parameter drift fails in the
default lane instead of waiting for a GPU runner. The GPU test below is the one
that proves each workflow actually designs.
"""

from __future__ import annotations

import os
import subprocess
import sys
from itertools import groupby
from pathlib import Path
from types import SimpleNamespace

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from script_utils.benchmark_backends import one_input

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = str(ROOT / "configs")
BENCHMARKS = ROOT / "data" / "benchmarks"

#: Released ``thinking_medium``: 60 sampling steps over 185 trunk passes.
THINKING_MEDIUM_STEPS = 60
THINKING_MEDIUM_TRUNK_PASSES = 185

#: workflow -> (benchmark input, extra generation overrides)
#:
#: Partial redesign is the same input as unconditional design; what makes it a
#: different workflow is where on the trajectory it starts.
WORKFLOWS: dict[str, tuple[str, tuple[str, ...]]] = {
    "unconditional": ("monomer/monomer-short.cif", ()),
    "motif_scaffolding": ("motif_scaffolding/motif_indexed", ()),
    "ppi_binder": ("ppi/ppi_bb_only", ()),
    "small_molecule_binder": ("small_molecule_binder/sm_binder", ()),
    "nucleic_acid_binder": ("dna_binder/dna_binder_denovo_val", ()),
    "partial_redesign": (
        "monomer/monomer-short.cif",
        ("generation.args.partial_trajectory_fraction_steps=0.3",),
    ),
}

#: Workflows whose rollout starts from noise, so no partial config is built.
FROM_NOISE = tuple(name for name in WORKFLOWS if name != "partial_redesign")


def composed(*overrides: str):
    """Compose ``inference`` at ``thinking_medium`` with the BioIR backend."""
    base = [
        "inference_backend=bioir",
        "recycle_strategy=thinking_medium",
        f"inputs={BENCHMARKS / 'monomer' / 'monomer-short.cif'}",
    ]
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base="1.3"):
        return compose(config_name="inference", overrides=base + list(overrides))


def native_stub(cfg, *, nsteps: int = THINKING_MEDIUM_STEPS):
    """A Proteina stand-in carrying only what the BioIR bridge reads."""
    from rfproteina.proteina import Proteina

    model = Proteina.__new__(Proteina)
    model.inf_cfg = cfg.generation
    model.nn = SimpleNamespace(recycle_enabled=True)
    model.guidance_model = None
    model.cfg_exp = OmegaConf.create({"product_flowmatcher": {"bb_ca": {"zero_com_noise": True}, "local_latents": {}}})
    return model


# --- the rollout ----------------------------------------------------------


def test_thinking_medium_is_sixty_steps_over_one_hundred_and_eighty_five_passes():
    cfg = composed()
    bands = OmegaConf.to_container(cfg.generation.n_recycle_schedule)
    assert int(cfg.generation.args.nsteps) == THINKING_MEDIUM_STEPS
    assert sum(width for width, _ in bands) == THINKING_MEDIUM_STEPS
    assert sum(width * depth for width, depth in bands) == THINKING_MEDIUM_TRUNK_PASSES


def test_thinking_medium_reaches_bioir_vf_tsr_sampling_unchanged():
    pytest.importorskip("bionemo_ir.models.rfproteina.config")
    from rfproteina.nn.bioir_full import sampling_from_native

    cfg = composed()
    config = sampling_from_native(native_stub(cfg))

    assert config.num_steps == THINKING_MEDIUM_STEPS
    bands = list(config.n_recycle_schedule)
    assert sum(width for width, _ in bands) == THINKING_MEDIUM_STEPS
    assert sum(width * depth for width, depth in bands) == THINKING_MEDIUM_TRUNK_PASSES
    # The bands are the native depth sequence run-length encoded, not a copy of
    # the config: partial diffusion takes a tail and must re-encode it.
    native = native_stub(cfg).recycle_depths(THINKING_MEDIUM_STEPS)
    assert bands == [(len(list(run)), depth) for depth, run in groupby(native)]

    # VF-TSR is the backbone step rule; a backend that quietly dropped to ODE or
    # lost the TSR parameters would still produce designs, just different ones.
    native_step = cfg.generation.model.bb_ca.simulation_step_params
    assert config.backbone.step.sampling_mode == native_step.sampling_mode
    assert config.backbone.step.tsr_k == native_step.tsr_k
    assert config.backbone.step.tsr_sigma == native_step.tsr_sigma

    # The tail grid densifies the last steps; bridging it is checked in the
    # bridge too, but only against whatever the runtime happened to accept.
    native_schedule = cfg.generation.model.bb_ca.schedule
    assert config.backbone.schedule.tail_steps == native_schedule.tail_steps
    assert config.backbone.schedule.tail_step_multiplier == native_schedule.tail_step_multiplier


def test_guidance_and_trajectory_export_are_refused_rather_than_ignored():
    pytest.importorskip("bionemo_ir.models.rfproteina.config")
    from rfproteina.nn.bioir_full import sampling_from_native

    with pytest.raises(NotImplementedError, match="guidance"):
        sampling_from_native(native_stub(composed("generation.args.guidance_w=0.5")))
    with pytest.raises(NotImplementedError, match="trajectory"):
        sampling_from_native(native_stub(composed("generation.args.save_trajectory_every=5")))


# --- the workflows --------------------------------------------------------


@pytest.mark.parametrize("workflow", sorted(WORKFLOWS))
def test_every_covered_workflow_has_a_benchmark_input(workflow: str):
    relative, _ = WORKFLOWS[workflow]
    assert one_input(relative).is_file(), f"{workflow} names a missing benchmark input"


@pytest.mark.parametrize("workflow", sorted(FROM_NOISE))
def test_workflows_that_start_from_noise_build_no_partial_config(workflow: str):
    pytest.importorskip("bionemo_ir.models.rfproteina.config")
    from rfproteina.nn.bioir_full import partial_from_native

    _, overrides = WORKFLOWS[workflow]
    assert partial_from_native(native_stub(composed(*overrides))) is None


def test_partial_redesign_restarts_the_trajectory_partway():
    pytest.importorskip("bionemo_ir.models.rfproteina.config")
    from rfproteina.nn.bioir_full import partial_from_native

    _, overrides = WORKFLOWS["partial_redesign"]
    partial = partial_from_native(native_stub(composed(*overrides)))
    assert partial is not None
    assert partial.partial_trajectory_fraction_steps == pytest.approx(0.3)
    assert not partial.inverse_folding


def test_inverse_folding_is_its_own_partial_mode():
    pytest.importorskip("bionemo_ir.models.rfproteina.config")
    from rfproteina.nn.bioir_full import partial_from_native

    partial = partial_from_native(native_stub(composed("generation.args.inverse_folding=true")))
    assert partial is not None and partial.inverse_folding


@pytest.mark.parametrize("workflow", sorted(WORKFLOWS))
def test_every_covered_workflow_composes_the_whole_bioir_workflow(workflow: str):
    from rfproteina.nn.execution import resolve_backends

    _, overrides = WORKFLOWS[workflow]
    cfg = composed(*overrides)
    policy = resolve_backends(cfg.execution, "inference")
    assert set(policy.values()) == {"bioir"}, f"{workflow} did not select BioIR for every component"


# --- the designs ----------------------------------------------------------


def _requires_gpu_rollout() -> tuple[Path, Path]:
    if os.getenv("RUN_GPU_E2E_TESTS", "0") != "1":
        pytest.skip("set RUN_GPU_E2E_TESTS=1 to run the BioIR workflow rollouts")
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA GPU")
    data_path = os.environ.get("DATA_PATH", "").strip()
    if not data_path:
        pytest.skip("needs DATA_PATH for the released checkpoint pair")
    flow = Path(data_path) / "flow_checkpoints"
    autoencoder = Path(data_path) / "ae_checkpoints"
    if not flow.is_dir() or not autoencoder.is_dir():
        pytest.skip("needs the released flow and autoencoder checkpoints")
    return flow, autoencoder


@pytest.mark.gpu
@pytest.mark.very_slow
@pytest.mark.parametrize("workflow", sorted(WORKFLOWS))
def test_covered_workflows_design_through_bioir(workflow: str, tmp_path: Path):
    """One ``thinking_medium`` rollout per workflow, end to end on the BioIR engine."""
    _requires_gpu_rollout()
    relative, overrides = WORKFLOWS[workflow]
    out_dir = tmp_path / workflow
    command = [
        sys.executable,
        "-m",
        "rfproteina.generate",
        f"inputs={one_input(relative)}",
        "inference_backend=bioir",
        "recycle_strategy=thinking_medium",
        "num_replicates=1",
        "num_workers=0",
        "skip_existing=false",
        "seed=11",
        f"out_dir={out_dir}",
        *overrides,
    ]
    proc = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=3600)
    assert proc.returncode == 0, f"{workflow} failed:\n{proc.stdout[-4000:]}\n{proc.stderr[-4000:]}"

    designs = [path for path in out_dir.rglob("*.cif") if "input" not in path.name]
    assert designs, f"{workflow} produced no design under {out_dir}"

    import numpy as np
    from atomworks.io.utils.io_utils import load_any

    array = load_any(str(designs[0]))
    coordinates = np.asarray(array.coord if hasattr(array, "coord") else array)
    assert coordinates.size, f"{workflow} wrote an empty structure"
    assert np.isfinite(coordinates).all(), f"{workflow} wrote non-finite coordinates"


def test_bioir_keeps_its_own_triangle_kernels():
    """``tri_mult`` reconfigures native triangles; a converted model must keep BioIR's.

    The converted model is held on a plain adapter rather than registered as a
    submodule, so ``configure_triangle_backends`` cannot reach it through
    ``modules()`` -- and under a full-BioIR policy it is not run at all, since
    the native blocks it would reconfigure are replaced and offloaded.
    """
    import torch
    from torch import nn

    from rfproteina.nn import execution
    from rfproteina.nn.bioir_full import FullBioIRAdapter

    class Native(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.nn = nn.Linear(2, 2)
            self.autoencoder = nn.ModuleDict({"encoder": nn.Linear(2, 2), "decoder": nn.Linear(2, 2)})
            self.is_inference = True
            self.device = torch.device("cpu")

    native = Native()
    converted = nn.Linear(2, 2)
    native._full_bioir = FullBioIRAdapter(native, converted)
    assert not isinstance(native._full_bioir, nn.Module)
    assert all(module is not converted for module in native.modules())

    applied: list[str] = []
    original = execution.configure_triangle_backends
    try:
        execution.configure_triangle_backends = lambda model, backend: applied.append(backend)
        native._full_bioir = None
        execution.configure_execution(native, composed().execution, "inference")
        assert applied == [], "a converted model must not be handed a native triangle policy"
        execution.configure_execution(
            native, {"inference": dict.fromkeys(("flow", "encoder", "decoder"), "eager")}, "inference"
        )
        assert applied == ["cuequivariance"], "the open-source path still selects its triangle backend"
    finally:
        execution.configure_triangle_backends = original
