"""GPU e2e: ``inference_backend=bioir`` designs what the native backend designs.

* ``test_bioir_first_step_velocity_matches_native``: both backends start from the same noise, so their first
  velocity prediction is directly comparable. It is the cleanest pin on a conversion fault (weights, chirality
  alignment, conditioning), free of sampler and decoder effects.
* ``test_bioir_designs_are_sound``: full stochastic generation. Independent samples cannot be compared one to
  one, so each design is checked on its own: native chains and residue counts, a physical CA-CA distance, and
  intra-residue bond lengths against the CCD ideals, gated against the native backend's own error.
* ``test_bioir_graph_replay_follows_every_batch``: batches that mix row lengths share one captured CUDA graph;
  every row of every batch must match the graph-free velocity, which fails if a replay reuses stale inputs.
* ``test_bioir_search_is_sound``: beam search through BioIR, held to the same per-design checks.

Run: RUN_GPU_E2E_TESTS=1 pytest tests/e2e/test_bioir_equivalence.py -m gpu -v
Needs the BioIR runtime that provides ``bionemo_ir.models.rfproteina`` and the released checkpoints. Without
it the equivalence tests skip.
"""

import os
from pathlib import Path

import numpy as np
import pytest
import torch
from biotite.structure.io.pdbx import CIFFile, get_structure
from hydra import compose, initialize_config_dir

from tests.e2e.helpers.gpu_e2e import ROOT, require_gpu_e2e

BENCHMARKS = ROOT / "data" / "benchmarks"

#: workflow -> benchmark input
WORKFLOWS = {
    "unconditional": "monomer/monomer-short.cif",
    "motif_scaffolding": "motif_scaffolding/motif_indexed",
    "ppi_binder": "ppi/ppi_bb_only",
    "small_molecule_binder": "small_molecule_binder/sm_binder",
    "nucleic_acid_binder": "dna_binder/dna_binder_denovo_val",
    "enzyme": "mcsa/mcsa_37/M0040_13pk.cif.zst",
}

ODE_OVERRIDES = [
    "generation.model.bb_ca.gt.mode=ode",
    "generation.model.local_latents.gt.mode=ode",
]
#: Mean relative intra-residue bond-length error of a design: native and BioIR both sit at 1-2.5%.
BOND_ERROR_MAX = 0.04
BOND_ERROR_MARGIN = 0.01  # BioIR's median may exceed native's by this much
CA_BOND_RANGE = (3.5, 4.1)  # Angstrom, physical window for a design's mean consecutive CA-CA distance
#: Relative error ceiling on the first velocity prediction, in bf16.
VELOCITY_REL_ERR_MAX = float(os.environ.get("BIOIR_VELOCITY_REL_ERR_MAX", "0.02"))
ONE_STEP_OVERRIDES = [
    *ODE_OVERRIDES,
    "generation.args.nsteps=1",
    "generation.n_recycle_schedule=[[1,1]]",
    "generation.model.bb_ca.schedule.tail_steps=0",
    "generation.model.local_latents.schedule.tail_steps=0",
]


def _generate(
    backend: str,
    inputs: Path,
    out_dir: Path,
    extra: tuple[str, ...] = (),
    *,
    metrics: bool = False,
    seed: int = 11,
    replicates: int = 1,
    batch_size: int = 1,
):
    from rfproteina.generate import load_ckpt_n_configure_inference, run_generation

    with initialize_config_dir(config_dir=str(ROOT / "configs"), version_base="1.3"):
        cfg = compose(
            config_name="inference",
            overrides=[
                f"inputs={inputs}",
                f"out_dir={out_dir}",
                f"inference_backend={backend}",
                *(["bioir.match_reference_rng=true"] if backend == "bioir" else []),
                "recycle_strategy=thinking_medium",
                "hf_ckpt_path=null",
                f"num_replicates={replicates}",
                f"batch_size={batch_size}",
                "num_workers=0",
                f"seed={seed}",
                "compress_outputs=false",
                *([] if metrics else ["~metrics"]),
                *extra,
            ],
        )
    model = load_ckpt_n_configure_inference(cfg)
    run_generation(cfg, model=model)
    return model, sorted(out_dir.glob("inference_samples/**/*_generated.cif"))


def _backbone(path: Path):
    atoms = get_structure(CIFFile.read(path), model=1)
    assert len(atoms) and np.isfinite(atoms.coord).all()
    ca = atoms[atoms.atom_name == "CA"]
    bonds = np.linalg.norm(np.diff(ca.coord, axis=0), axis=1)
    consecutive = bonds[bonds < 6.0]  # drop chain breaks
    layout = (tuple(sorted(set(atoms.chain_id))), len(ca))  # atom counts follow the sampled sequence
    return layout, float(consecutive.mean())


def _require_bioir_runtime() -> None:
    """Skip when the installed BioIR cannot run the whole workflow (``test_ci_image_has_the_bioir_runtime`` enforces it)."""
    from rfproteina.nn.bioir_full import require_bioir

    try:
        require_bioir()
    except (ImportError, RuntimeError) as error:
        pytest.skip(f"BioIR runtime unavailable: {error}")


def _bond_error(path: Path) -> float:
    """Mean relative intra-residue bond-length error against the CCD, as the pipeline's metric reports it."""
    from atomworks.io.parser import parse

    from rfproteina.metrics.bond_length_metric import InternalResidueBondMetric
    from rfproteina.utils.parse_presets import STANDARD_ANNOTATIONS_ONLY

    atoms = parse(path, config=STANDARD_ANNOTATIONS_ONLY)["assemblies"]["1"][0]
    error = InternalResidueBondMetric().compute(atoms)["organic_organic_mean_rel_error"]
    assert np.isfinite(error), f"{path.name}: no residue bonds could be compared"
    return error


def _assert_sound(candidate: list[Path], reference: list[Path], label: str) -> None:
    ref_errors, bio_errors = [], []
    for ref_path, bio_path in zip(reference, candidate, strict=True):
        ref_layout, _ = _backbone(ref_path)
        bio_layout, bio_bond = _backbone(bio_path)
        assert bio_layout == ref_layout
        assert CA_BOND_RANGE[0] <= bio_bond <= CA_BOND_RANGE[1], f"{label}: mean CA-CA {bio_bond:.2f} A"
        ref_errors.append(_bond_error(ref_path))
        bio_errors.append(_bond_error(bio_path))
    print(f"{label}: bond error native {np.round(ref_errors, 4).tolist()} bioir {np.round(bio_errors, 4).tolist()}")
    assert max(bio_errors) <= BOND_ERROR_MAX, f"{label}: bond error {bio_errors}"
    assert np.median(bio_errors) <= np.median(ref_errors) + BOND_ERROR_MARGIN, f"{label}: {bio_errors} vs {ref_errors}"


@pytest.mark.gpu
@pytest.mark.parametrize("workflow", list(WORKFLOWS))
def test_bioir_designs_are_sound(workflow: str, tmp_path: Path):
    require_gpu_e2e(min_gpus=1)
    _require_bioir_runtime()
    from rfproteina.nn.bioir_full import full_bioir_runtime

    inputs = BENCHMARKS / WORKFLOWS[workflow]
    native, reference = _generate("native", inputs, tmp_path / "native", replicates=4)
    del native  # Free the native model's GPU memory before BioIR converts its own copy.
    torch.cuda.empty_cache()
    model, candidate = _generate("bioir", inputs, tmp_path / "bioir", replicates=4)

    runtime = full_bioir_runtime(model)
    assert runtime["completed_calls"] > 0
    assert runtime["graph_cache"]["captured"] > 0, "BioIR ran eagerly; no CUDA graph was captured"
    assert runtime["graph_cache"]["fallback"] == 0
    assert reference and len(candidate) == len(reference)
    _assert_sound(candidate, reference, workflow)


class _CapturedError(Exception):
    """Raised once the first velocity is recorded, so generation stops before any formatting."""


def _first_velocity(backend: str, inputs: Path, out_dir: Path) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """``{modality: (state, velocity)}`` the backend predicts at its first step, on CPU."""
    from unittest import mock

    from rfproteina.flow_matching.product_space_flow_matcher import ProductSpaceFlowMatcher

    seen: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}

    def native(self, x_t, nn_out, *args, **kwargs):
        for mode in x_t:
            seen[mode] = (x_t[mode].float().cpu(), nn_out[mode]["v"].float().cpu())
        raise _CapturedError

    if backend == "native":
        patch = mock.patch.object(ProductSpaceFlowMatcher, "simulation_step", native)
    else:
        from bionemo_ir._torch.modules.rfproteina import rollout

        real = rollout.velocity_flow_step

        def bioir(state, velocity, *args, **kwargs):
            seen["bb_ca" if state.shape[-1] == 3 else "local_latents"] = (state.float().cpu(), velocity.float().cpu())
            if len(seen) == 2:
                raise _CapturedError
            return real(state, velocity, *args, **kwargs)

        patch = mock.patch.object(rollout, "velocity_flow_step", bioir)
    with patch, pytest.raises(_CapturedError):
        _generate(backend, inputs, out_dir, tuple(ONE_STEP_OVERRIDES))
    return seen


@pytest.mark.gpu
@pytest.mark.parametrize("workflow", list(WORKFLOWS))
def test_bioir_first_step_velocity_matches_native(workflow: str, tmp_path: Path):
    """From the same noise the backends predict the same velocity, up to bf16 rounding."""
    require_gpu_e2e(min_gpus=1)
    _require_bioir_runtime()
    inputs = BENCHMARKS / WORKFLOWS[workflow]
    reference = _first_velocity("native", inputs, tmp_path / "native")
    torch.cuda.empty_cache()
    candidate = _first_velocity("bioir", inputs, tmp_path / "bioir")
    for modality, (state, velocity) in candidate.items():
        ref_state, ref_velocity = reference[modality]
        tokens = min(state.shape[1], ref_state.shape[1])  # native pads to a length bucket; BioIR does not
        state, ref_state = state[:, :tokens], ref_state[:, :tokens]
        torch.testing.assert_close(state, ref_state, rtol=1e-4, atol=1e-4)  # the same noise, or nothing is comparable
        velocity, ref_velocity = velocity[:, :tokens], ref_velocity[:, :tokens]
        # Native zeroes the velocity of tokens that emit no latent (ligand and fixed tokens); BioIR zeroes their
        # state after the step instead, so only the emitting tokens are comparable.
        emitting = torch.linalg.norm(ref_velocity, dim=-1) > 0
        velocity, ref_velocity = velocity[emitting], ref_velocity[emitting]
        error = float(torch.linalg.norm(velocity - ref_velocity) / torch.linalg.norm(ref_velocity))
        print(f"{workflow} {modality}: first-step velocity relative error {error:.4f}")
        assert error <= VELOCITY_REL_ERR_MAX, f"{workflow} {modality}: relative error {error:.4f}"


@pytest.mark.gpu
def test_bioir_graph_replay_follows_every_batch(tmp_path: Path):
    """Batches that mix row lengths replay one captured graph; every row must still match the graph-free result."""
    require_gpu_e2e(min_gpus=1)
    _require_bioir_runtime()
    from unittest import mock

    from bionemo_ir._torch.modules.rfproteina import rollout

    inputs = BENCHMARKS / "mcsa" / "mcsa_37"
    extra = (*ONE_STEP_OVERRIDES, "subset_to_examples=[M0188_1xel,M0555_1f8r,M0097_1ctt]")

    def first_velocities(graph: bool) -> list[torch.Tensor]:
        seen: list[torch.Tensor] = []
        real = rollout.velocity_flow_step

        def record(state, velocity, *args, **kwargs):
            if state.shape[-1] == 3:  # the backbone call of each step
                seen.append(velocity.float().cpu())
            return real(state, velocity, *args, **kwargs)

        with mock.patch.object(rollout, "velocity_flow_step", record):
            overrides = (*extra, f"bioir.cuda_graph={str(graph).lower()}")
            _generate("bioir", inputs, tmp_path / f"graph_{graph}", overrides, replicates=8, batch_size=8)
        return seen

    with_graphs, without_graphs = first_velocities(True), first_velocities(False)
    assert len(with_graphs) == len(without_graphs) >= 3
    for batch, (replayed, eager) in enumerate(zip(with_graphs, without_graphs, strict=True)):
        error = torch.linalg.norm(replayed - eager, dim=(-2, -1)) / torch.linalg.norm(eager, dim=(-2, -1))
        assert float(error.max()) <= 1e-3, f"batch {batch}: per-row relative error {error.tolist()}"


@pytest.mark.gpu
def test_bioir_search_is_sound(tmp_path: Path):
    """Beam search with the clash reward (no oracle) completes through BioIR with sound, equally many designs."""
    require_gpu_e2e(min_gpus=1)
    _require_bioir_runtime()
    from rfproteina.nn.bioir_full import full_bioir_runtime

    inputs = BENCHMARKS / "monomer"
    extra = ("inference_experiment=inference_beam_regression", "subset_to_examples=monomer-short")
    _, reference = _generate("native", inputs, tmp_path / "native", extra, metrics=True)
    model, candidate = _generate("bioir", inputs, tmp_path / "bioir", extra, metrics=True)
    assert reference and len(candidate) == len(reference)
    runtime = full_bioir_runtime(model)
    assert runtime["completed_calls"] > 0 and runtime["graph_cache"]["fallback"] == 0
    _assert_sound(candidate, reference, "search")
