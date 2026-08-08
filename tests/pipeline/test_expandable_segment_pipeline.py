"""Pipeline regression tests for unconditional / expandable-segment featurization.

Two tests:

1. ``test_train_val_pipeline_feats_match`` — running the same input through the
   training pipeline (with an unconditional design task) and the validation
   pipeline must yield identical ``feats`` dicts (every key, every value).

2. ``test_expandable_segment_features_match_unconditional`` — running the
   validation pipeline twice (plain vs. prefix-replaced-by-realized-expandable-
   segment), feeding both batches through the *same* downstream setup
   ``VanillaStrategy`` uses at inference (autoencoder encode → ``corrupt_batch``
   at ``t=1``), and then through every per-feature creator from
   :class:`FeatureFactory <rfproteina.nn.feature_factory_atom37.FeatureFactory>`,
   must yield identical tensors at every token (no expanded-region carve-out).
"""

import os
from collections.abc import Callable
from pathlib import Path

import rootutils

from rfproteina.utils.env import load_dotenv
from tests.helpers import gpu  # noqa: F401

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
load_dotenv(override=True)

import numpy as np
import pytest
import torch
from atomworks.constants import AF3_EXCLUDED_LIGANDS, PDB_MIRROR_PATH, STANDARD_AA
from atomworks.io.config import ParseConfig
from atomworks.io.parser import parse
from atomworks.io.utils.annotator import ensure_annotations
from atomworks.io.utils.atom_array_plus import as_atom_array_plus
from atomworks.io.utils.io_utils import to_cif_file
from atomworks.io.utils.selection import get_residue_starts
from atomworks.ml.transforms.filters import (
    HandleUndesiredResTokens,
    RemovePolymersWithTooFewResolvedResidues,
    RemoveUnresolvedLigandAtomsIfTooMany,
    RemoveUnresolvedPNUnits,
)
from atomworks.ml.utils.expandable_segment import (
    insert_expandable_segment,
    realize_expandable_segments,
)
from atomworks.ml.utils.rng import create_rng_state_from_seeds, rng_state
from omegaconf import OmegaConf

from rfproteina.datapipes.collate import collate_batch
from rfproteina.datapipes.pipeline import (
    build_design_train_pipeline,
    build_design_validation_pipeline,
)
from rfproteina.flow_matching.product_space_flow_matcher import (
    ProductSpaceFlowMatcher,
)
from rfproteina.nn.feature_factory import FeatureFactory
from rfproteina.partial_autoencoder.autoencoder import AutoEncoder
from rfproteina.transforms.design_task import (
    MotifScaffoldTask,
    PPIHotspotWith3DTarget,
    StructurePredictionTask,
    TipAtomActiveSiteTask,
    TipAtomPartialLigandTask,
    UnconditionalTask,
)
from rfproteina.utils.parse_presets import ANNOTATED_CIF

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PDB_IDS = [
    "6lyz",  # Plain monomer (lysozyme)
    "5ocm",  # Protein dimer + small-molecule ligand
    "6wtf",  # Protein + ligand
    "3wbm",  # Protein-RNA complex
    "4kpy",  # DNA-binding protein (with DNA chains)
    "6mtg",  # Protein with non-canonical amino acid
    "7rxs",  # Multi-chain protein
    "1ivo",  # lots of covalent modifications
    "5w40",  # Weird thing like lenacapivir
    "3wrh",  # Metal on same chain as ligand
    "4ndz",  # Sugars
]

N_REMOVE = 5
SEED = 42

# Bundled mmCIFs for pipeline tests (see also tests/pipeline/test_pipeline_invariants.py).
_TEST_DATA_DIR = Path(__file__).parent.parent / "data" / "pdb"


def _preprocessed_mirror_cif(mirror: Path, pdb_id: str) -> Path | None:
    """Resolve a structure under ``2026_01_06_pdb_preprocessed`` (ord ``/squash/pdb``).

    The archive uses two conventions per entry directory ``{shard}/{id}/``:
    - single assembly: ``{id}.cif`` (e.g. ``w4/5w40/5w40.cif``)
    - multiple bioassemblies: ``{id}_{n}.cif`` (e.g. ``08/108m/108m_0.cif``)
    Training reads exact paths from parquet; this helper picks a default by PDB ID.
    """
    pdb_id = pdb_id.lower()
    entry = mirror / pdb_id[1:3] / pdb_id
    canonical = entry / f"{pdb_id}.cif"
    if canonical.is_file():
        return canonical
    bio0 = entry / f"{pdb_id}_0.cif"
    if bio0.is_file():
        return bio0
    variants = sorted(entry.glob(f"{pdb_id}_*.cif"))
    return variants[0] if variants else None


def _load_pdb_for_test(pdb_id: str, **parse_kwargs) -> dict:
    """Load a test structure (bundled CIF or preprocessed squash; not ``cached_parse``)."""
    pdb_id = pdb_id.lower()
    local = _TEST_DATA_DIR / f"{pdb_id}.cif"
    if local.is_file():
        path = local
    else:
        mirror = Path(PDB_MIRROR_PATH) if PDB_MIRROR_PATH else None
        if mirror is None or not mirror.is_dir():
            pytest.skip(f"No bundled CIF for {pdb_id} and PDB_MIRROR_PATH is unset")
        path = _preprocessed_mirror_cif(mirror, pdb_id)
        if path is None:
            pytest.skip(
                f"No structure for {pdb_id}: add {_TEST_DATA_DIR / f'{pdb_id}.cif'} "
                f"or ensure preprocessed mirror has {mirror / pdb_id[1:3] / pdb_id}/"
            )
    config = ParseConfig(**parse_kwargs) if parse_kwargs else None
    data = parse(path, config=config)
    if "atom_array" not in data:
        assembly_ids = list(data["assemblies"].keys())
        data["atom_array"] = data["assemblies"][assembly_ids[0]][0]
    data["pdb_id"] = pdb_id
    return data


# Autoencoder checkpoint lives under ``$DATA_PATH/latent_checkpoints/ae_checkpoints``
# (per-cluster, e.g. /path/to/data on digs). Resolve it from
# DATA_PATH rather than hardcoding a single cluster's mount, matching the
# ``autoencoder_ckpt_path`` convention used across the configs.
_DATA_PATH = os.environ.get("DATA_PATH", "").strip()
AUTOENCODER_CKPT_PATH = (
    Path(_DATA_PATH) / "latent_checkpoints" / "ae_checkpoints" / "AE_ucond_512.ckpt" if _DATA_PATH else None
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
NN_CONFIG_PATH = PROJECT_ROOT / "configs" / "nn" / "local_latents_score_nn_160M_atom37.yaml"
TRAIN_CONFIG_PATH = PROJECT_ROOT / "configs" / "training_proteina.yaml"

GLOBAL_DESIGN_TASK_KWARGS = {
    "rasa_kwargs": {"p_add_rasa_condition": 0.0},
    "hbond_kwargs": {"p_add_hbond_condition": 0.0},
    "secondary_structure_kwargs": {"p_add_secondary_structure_condition": 0.0},
    "cath_kwargs": {
        "p_add_cath_c_condition": 0.0,
        "p_add_cath_ca_condition": 0.0,
        "p_add_cath_cat_condition": 0.0,
    },
}

# Test 2 uses the production NN config's feature lists by default. To deviate
# from the config without editing it, append to ``EXTRA_*`` (test additions) or
# ``REMOVE_*`` (skip these even if listed in the config). Both are empty when
# the test mirrors the config exactly.
EXTRA_SEQ_FEATURES: list[str] = []
REMOVE_SEQ_FEATURES: list[str] = []

EXTRA_SEQ_COND_FEATURES: list[str] = []
REMOVE_SEQ_COND_FEATURES: list[str] = []

EXTRA_PAIR_REPR_FEATURES: list[str] = []
REMOVE_PAIR_REPR_FEATURES: list[str] = []

EXTRA_PAIR_COND_FEATURES: list[str] = []
REMOVE_PAIR_COND_FEATURES: list[str] = []


def _resolve_feature_list(cfg_feats: list[str], extra: list[str], remove: list[str]) -> list[str]:
    """Apply ``extra`` / ``remove`` overrides on top of a config-provided list."""
    removed = set(remove)
    out = [f for f in cfg_feats if f not in removed]
    out += [f for f in extra if f not in out]
    return out


# ---------------------------------------------------------------------------
# Input cleanup and expandable-segment manipulation
# ---------------------------------------------------------------------------


def _preclean_input(atom_array):
    """Apply the train-only cleanup filters once up-front so that train and val
    pipelines see the same set of tokens.

    The input is parsed with ``add_missing_atoms=True`` so missing heavy atoms
    are populated with NaN coords. Here we apply hydrogens/terminal-oxygen/
    excluded-ligand cleanup, then replace any remaining NaN coords with the
    origin so downstream ``RemoveTokensWithAnyNaNAtom`` (after atomization)
    leaves them in place. This keeps train and val pipelines operating on the
    same residue set, so ``within_chain_source_res_idx`` and dependents agree.
    """
    aa = as_atom_array_plus(atom_array.copy())
    data = {"atom_array": aa}
    for transform in (
        RemoveUnresolvedPNUnits(),
        HandleUndesiredResTokens(undesired_res_tokens=AF3_EXCLUDED_LIGANDS),
        RemovePolymersWithTooFewResolvedResidues(min_residues=1),
        RemoveUnresolvedLigandAtomsIfTooMany(unresolved_ligand_atom_limit=5),
    ):
        data = transform(data)

    aa = as_atom_array_plus(data["atom_array"])
    nan_mask = np.isnan(aa.coord).any(axis=-1)
    aa.coord[nan_mask] = 0.0
    return aa


def _replace_prefix_with_expandable_segment(atom_array, n_residues: int):
    """Find the first contiguous run of ``n_residues`` canonical amino-acid
    residues and replace it with a realized (n, n) expandable segment.

    Restricting the replaced region to standard AAs keeps the comparison
    focused on segment-vs-canonical featurization parity. Non-canonical
    residues introduce orthogonal complications (e.g. they are atomized or
    mutated by upstream filters), which the test is not trying to cover.
    """
    aa = as_atom_array_plus(atom_array.copy())
    res_starts = get_residue_starts(aa, add_exclusive_stop=True)
    n_res = len(res_starts) - 1
    assert n_res > n_residues, "Structure too short for prefix removal."

    res_names = np.array([aa.res_name[res_starts[i]] for i in range(n_res)])
    is_canonical = np.isin(res_names, np.array(STANDARD_AA))

    run_start = None
    for i in range(n_res - n_residues + 1):
        if is_canonical[i : i + n_residues].all():
            run_start = i
            break
    assert run_start is not None, f"No contiguous run of {n_residues} canonical AAs found in structure."
    run_stop = run_start + n_residues

    drop_lo = int(res_starts[run_start])
    drop_hi = int(res_starts[run_stop])
    keep_idx = np.r_[0:drop_lo, drop_hi : aa.array_length()]
    truncated = as_atom_array_plus(aa[keep_idx].copy())

    # Insert before the first atom of the kept suffix (now at index ``drop_lo``).
    # If the dropped run was a true suffix (no kept suffix), insert after the
    # last kept atom instead.
    if drop_lo < truncated.array_length():
        insert_kwargs = {"atom_idx": drop_lo, "insert_before": True}
    else:
        insert_kwargs = {"atom_idx": truncated.array_length() - 1, "insert_before": False}

    with_segment = insert_expandable_segment(
        truncated,
        seg_min=n_residues,
        seg_max=n_residues,
        **insert_kwargs,
    )
    return realize_expandable_segments(with_segment, rng=np.random.default_rng(0))


# ---------------------------------------------------------------------------
# Tensor equality with NaN tolerance
# ---------------------------------------------------------------------------


def _values_equal(a, b) -> bool:
    """NaN-aware equality (NaN == NaN at matching positions)."""
    if type(a) is not type(b):
        return False
    if isinstance(a, torch.Tensor):
        if a.shape != b.shape or a.dtype != b.dtype:
            return False
        if a.is_floating_point():
            nan_a, nan_b = torch.isnan(a), torch.isnan(b)
            return bool(torch.equal(nan_a, nan_b)) and bool(torch.equal(a[~nan_a], b[~nan_b]))
        return bool(torch.equal(a, b))
    if isinstance(a, np.ndarray):
        if a.shape != b.shape or a.dtype != b.dtype:
            return False
        if np.issubdtype(a.dtype, np.floating):
            return bool(np.array_equal(a, b, equal_nan=True))
        return bool(np.array_equal(a, b))
    if isinstance(a, dict):
        return set(a) == set(b) and all(_values_equal(a[k], b[k]) for k in a)
    if isinstance(a, list | tuple):
        return len(a) == len(b) and all(_values_equal(x, y) for x, y in zip(a, b, strict=False))
    return a == b


# ---------------------------------------------------------------------------
# Shared feats comparison
# ---------------------------------------------------------------------------


# ``chain_iid`` and ``molecule_iid`` are partition labels: their absolute
# values are arbitrary (the train pipeline may rename chains, e.g., via
# :class:`FlagAndReassignCovalentModifications`). What must agree is the
# induced partitioning — i.e., for every pair of tokens (i, j), whether they
# share a chain/molecule. Compare same-group adjacency matrices rather than
# raw values.
PARTITION_KEYS = frozenset({"chain_iid", "molecule_iid", "condition_index-group_1_atom"})


def _same_group_matrix(x):
    if isinstance(x, torch.Tensor):
        return x.unsqueeze(0) == x.unsqueeze(1)
    return np.equal.outer(x, x)


def _assert_feats_match(feats_a: dict, feats_b: dict, label: str) -> None:
    """Assert ``feats_a`` and ``feats_b`` are identical, with partition-label
    keys compared as same-group adjacency matrices."""
    assert set(feats_a) == set(feats_b), (
        f"[{label}] Feature keys differ.\n"
        f"  A only: {sorted(set(feats_a) - set(feats_b))}\n"
        f"  B only: {sorted(set(feats_b) - set(feats_a))}"
    )
    mismatches: list[str] = []
    for k in sorted(feats_a):
        a, b = feats_a[k], feats_b[k]
        if k in PARTITION_KEYS:
            if not _values_equal(_same_group_matrix(a), _same_group_matrix(b)):
                mismatches.append(k)
        elif not _values_equal(a, b):
            mismatches.append(k)
    assert not mismatches, f"[{label}] Feature mismatch: {mismatches}"


# ---------------------------------------------------------------------------
# Test 1: train pipeline (unconditional) == val pipeline
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("pdb_id", PDB_IDS)
def test_train_val_pipeline_feats_match(pdb_id: str) -> None:
    """The unconditional train pipeline and val pipeline must produce
    identical ``feats`` dicts for the same precleaned input."""
    base = _load_pdb_for_test(pdb_id, add_missing_atoms=True)
    base["example_id"] = pdb_id
    base["atom_array"] = _preclean_input(base["atom_array"])

    input_a = {**base, "atom_array": base["atom_array"].copy()}
    with rng_state(create_rng_state_from_seeds(np_seed=SEED, torch_seed=SEED, py_seed=SEED)):
        pipe_a = build_design_train_pipeline(
            crop_size=None,
            design_tasks={
                "task": {
                    "frequency": 1.0,
                    "transform": UnconditionalTask(global_design_task_kwargs=GLOBAL_DESIGN_TASK_KWARGS),
                },
            },
            is_inference=True,
            p_random_rotation=0.0,
            p_atomize_residues=0.0,
        )
        out_a = pipe_a(input_a)

    input_b = {**base, "atom_array": base["atom_array"].copy()}
    with rng_state(create_rng_state_from_seeds(np_seed=SEED, torch_seed=SEED, py_seed=SEED)):
        pipe_b = build_design_validation_pipeline()
        out_b = pipe_b(input_b)

    _assert_feats_match(out_a["feats"], out_b["feats"], pdb_id)


# ---------------------------------------------------------------------------
# Test 1b: train pipeline (with design task) → save → reload → val pipeline
# ---------------------------------------------------------------------------


def _motif_scaffold_factory():
    return MotifScaffoldTask(global_design_task_kwargs=GLOBAL_DESIGN_TASK_KWARGS)


def _folding_factory():
    return StructurePredictionTask(global_design_task_kwargs=GLOBAL_DESIGN_TASK_KWARGS)


def _tip_atom_partial_ligand_factory():
    # Mirrors all_off.yaml settings that deviate from the class defaults.
    return TipAtomPartialLigandTask(
        n_max_residues=8,
        global_design_task_kwargs=GLOBAL_DESIGN_TASK_KWARGS,
    )


def _tip_atom_pocket_factory():
    # Defaults match all_off.yaml exactly.
    return TipAtomActiveSiteTask(global_design_task_kwargs=GLOBAL_DESIGN_TASK_KWARGS)


def _ppi_hotspot_with_3d_target_factory():
    # Mirrors all_off.yaml settings that deviate from the class defaults.
    return PPIHotspotWith3DTarget(
        hotspot_count_distribution="geometric",
        hotspot_geometric_p=0.2,
        max_hotspots_frac_to_provide=0.1,
        p_no_hotspots=0.05,
        hotspot_max_distance=4.5,
        ppi_crop_binder_fraction=0.667,
        p_sample_ppi_crop=0.75,
        bb_only_coord_conditioning=False,
        global_design_task_kwargs=GLOBAL_DESIGN_TASK_KWARGS,
    )


# Per-PDB applicable design tasks. We always test motif_scaffold and folding;
# tip_atom_partial_ligand for structures with non-polymer ligands;
# ppi_hotspot_with_3d_target for structures with two protein chains.
_PDB_DESIGN_TASKS: dict[str, list[tuple[str, Callable]]] = {
    "6lyz": [
        ("motif_scaffold", _motif_scaffold_factory),
        ("folding", _folding_factory),
        ("tip_atom_pocket", _tip_atom_pocket_factory),
    ],
    "5ocm": [
        ("motif_scaffold", _motif_scaffold_factory),
        ("folding", _folding_factory),
        ("tip_atom_partial_ligand", _tip_atom_partial_ligand_factory),
        ("ppi_hotspot_with_3d_target", _ppi_hotspot_with_3d_target_factory),
    ],
    "6wtf": [
        ("motif_scaffold", _motif_scaffold_factory),
        ("folding", _folding_factory),
        ("tip_atom_partial_ligand", _tip_atom_partial_ligand_factory),
    ],
    "3wbm": [("motif_scaffold", _motif_scaffold_factory), ("folding", _folding_factory)],
    "4kpy": [("motif_scaffold", _motif_scaffold_factory), ("folding", _folding_factory)],
    "6mtg": [("motif_scaffold", _motif_scaffold_factory), ("folding", _folding_factory)],
    "7rxs": [
        ("motif_scaffold", _motif_scaffold_factory),
        ("folding", _folding_factory),
        ("tip_atom_partial_ligand", _tip_atom_partial_ligand_factory),
        ("tip_atom_pocket", _tip_atom_pocket_factory),
    ],
    "1ivo": [
        ("motif_scaffold", _motif_scaffold_factory),
        ("folding", _folding_factory),
        ("tip_atom_partial_ligand", _tip_atom_partial_ligand_factory),
        ("ppi_hotspot_with_3d_target", _ppi_hotspot_with_3d_target_factory),
    ],
    "5w40": [
        ("motif_scaffold", _motif_scaffold_factory),
        ("folding", _folding_factory),
        ("tip_atom_pocket", _tip_atom_pocket_factory),
    ],
    "3wrh": [
        ("motif_scaffold", _motif_scaffold_factory),
        ("folding", _folding_factory),
        ("tip_atom_partial_ligand", _tip_atom_partial_ligand_factory),
    ],
}

_PDB_TASK_PARAMS = [
    pytest.param(pdb_id, task_name, factory, id=f"{pdb_id}-{task_name}")
    for pdb_id, tasks in _PDB_DESIGN_TASKS.items()
    for task_name, factory in tasks
]


@pytest.mark.parametrize("pdb_id,task_name,task_factory", _PDB_TASK_PARAMS)
@pytest.mark.parametrize("remove_atomize", [False, True], ids=["keep_atomize", "drop_atomize"])
def test_train_save_load_val_pipeline_feats_match(
    pdb_id: str,
    task_name: str,
    task_factory: Callable,
    remove_atomize: bool,
    tmp_path,
) -> None:
    """Run the train pipeline with a design task, save the resulting atom
    array to CIF, reload via the inference parse preset, then run the val
    pipeline. The val pipeline ``feats`` must match the train pipeline
    ``feats``.

    With ``remove_atomize=True`` the ``atomize`` annotation is dropped right
    after reload to verify the val pipeline regenerates it correctly.
    """
    base = _load_pdb_for_test(pdb_id, add_missing_atoms=True)
    base["example_id"] = pdb_id
    base["atom_array"] = _preclean_input(base["atom_array"])

    # PPIHotspotWith3DTarget.can_apply requires ``query_pn_unit_iids`` (a
    # length-2 list of distinct protein pn_unit_iids) — normally supplied by
    # dimer datasets. Inject the first one or two protein pn_unit_iids here
    # so PPI is eligible on multi-protein-chain inputs; harmless for other
    # tasks.
    ensure_annotations(base["atom_array"], "is_protein")
    protein_pn_unit_iids = list(np.unique(base["atom_array"].pn_unit_iid[base["atom_array"].is_protein]))[:2]

    # --- Train pipeline with the design task ---
    input_train = {
        **base,
        "atom_array": base["atom_array"].copy(),
        "query_pn_unit_iids": protein_pn_unit_iids,
    }
    with rng_state(create_rng_state_from_seeds(np_seed=SEED, torch_seed=SEED, py_seed=SEED)):
        pipe_train = build_design_train_pipeline(
            crop_size=None,
            design_tasks={
                "task": {"frequency": 1.0, "transform": task_factory()},
            },
            is_inference=True,
            p_random_rotation=0.0,
            p_atomize_residues=0.0,
        )
        out_train = pipe_train(input_train)

    # --- Save the train-pipeline atom_array to CIF and reload it ---
    cif_path = tmp_path / f"{pdb_id}_{task_name}.cif"
    to_cif_file(out_train["atom_array"], cif_path, id=pdb_id)

    parsed = parse(cif_path, config=ANNOTATED_CIF)
    reloaded_aa = parsed["assemblies"]["1"][0]

    if remove_atomize:
        if "atomize" not in reloaded_aa.get_annotation_categories():
            pytest.skip(reason="No atomize annotation present after reload; nothing to remove.")
        reloaded_aa.del_annotation("atomize")

    # --- Val pipeline on the reloaded atom_array ---
    input_val = {
        "atom_array": reloaded_aa,
        "chain_info": parsed["chain_info"],
        "ligand_info": parsed["ligand_info"],
        "metadata": parsed["metadata"],
        "example_id": pdb_id,
    }
    with rng_state(create_rng_state_from_seeds(np_seed=SEED, torch_seed=SEED, py_seed=SEED)):
        pipe_val = build_design_validation_pipeline()
        out_val = pipe_val(input_val)

    label = f"{pdb_id}-{task_name}-{'drop_atomize' if remove_atomize else 'keep_atomize'}"
    _assert_feats_match(out_train["feats"], out_val["feats"], label)


# ---------------------------------------------------------------------------
# Test 2 helpers: collate, AE encode, corruption, per-feature comparison
# ---------------------------------------------------------------------------


def _run_validation_pipeline(input_dict: dict) -> dict:
    """Run the validation pipeline once under a fixed RNG."""
    with rng_state(create_rng_state_from_seeds(np_seed=SEED, torch_seed=SEED, py_seed=SEED)):
        pipe = build_design_validation_pipeline()
        return pipe(input_dict)


def _collate(pipeline_out: dict):
    """Wrap a single pipeline output into a batch of size 1 using the production
    schema collate."""
    return collate_batch(
        [pipeline_out],
        schema="rfproteina.datapipes.schema.proteina.ProteinaDataSample",
    )


def _encode_local_latents(batch, autoencoder, latent_dim: int) -> torch.Tensor:
    """Mirror :py:meth:`Proteina._get_clean_sample` for ``local_latents``."""
    can_be_encoded = batch["can_be_encoded"]
    padding_mask = batch["padding_mask"]
    encodable_mask = can_be_encoded & padding_mask

    if not encodable_mask.any():
        b, n = batch["coords_nm"].shape[:2]
        return torch.zeros(b, n, latent_dim, device=batch["coords_nm"].device)

    residue_type_for_ae = batch["residue_type"] - 1
    residue_type_for_ae = torch.where(encodable_mask, residue_type_for_ae, torch.zeros_like(residue_type_for_ae))
    ae_batch = {
        "coords": batch["coords"],
        "coords_nm": batch["coords_nm"],
        "coord_mask": batch.get("coord_mask", torch.ones_like(batch["coords"][..., 0])),
        "residue_type": residue_type_for_ae,
        "padding_mask": encodable_mask,
        "can_be_encoded": encodable_mask,
    }
    encoded = autoencoder.encode(ae_batch)
    latents = encoded["z_latent"]
    return latents * encodable_mask.unsqueeze(-1).float()


def _prepare_inference_batch(
    pipeline_out: dict,
    autoencoder,
    fm: ProductSpaceFlowMatcher,
    latent_dim: int,
    device: torch.device,
):
    """Mirror :class:`VanillaStrategy`'s setup up to the first model call:
    collate → ``preprocess_batch_for_generation`` → ``add_clean_samples`` →
    ``corrupt_batch(t=1)`` → zero ``x_sc`` and ``x_recycle``.

    ``t=1`` and a fixed torch seed make ``x_t`` (drawn from the prior) identical
    across the two runs, so any downstream feature difference reflects a real
    featurization disagreement.
    """
    batch = _collate(pipeline_out)
    _move_batch_to_device(batch, device)

    batch["coords_nm"] = batch["coords"] / 10.0
    batch["coord_mask"] = batch["atom37_mask"]

    x_1 = {
        "bb_ca": batch["coords_nm"][:, :, 1, :],
        "local_latents": _encode_local_latents(batch, autoencoder, latent_dim),
    }
    batch["x_1"] = x_1

    # Mirror the first step of ``full_simulation``: ``x_t`` is pure prior noise
    # (independent of ``x_1``), masked to encodable/non-padding tokens. ``t`` is
    # set to 1.0 (start of inference). A fixed torch seed makes the noise draw
    # identical across the two runs.
    batch_shape = batch["padding_mask"].shape[:1]
    n = batch["padding_mask"].shape[1]

    torch.manual_seed(SEED)
    x_t = fm.sample_noise(n=n, shape=tuple(batch_shape), mask=batch["padding_mask"], device=device, training=False)
    is_encoded_mask = batch.get("can_be_encoded")
    if is_encoded_mask is not None:
        x_t["local_latents"] = x_t["local_latents"] * is_encoded_mask[..., None]
    x_t["local_latents"] = x_t["local_latents"] * batch["padding_mask"][..., None]
    x_t["bb_ca"] = x_t["bb_ca"] * batch["padding_mask"][..., None]

    batch["x_t"] = x_t
    batch["t"] = {dm: torch.ones(batch_shape, device=device) for dm in fm.data_modes}
    batch["x_sc"] = {k: torch.zeros_like(v) for k, v in batch["x_t"].items()}
    batch["x_recycle"] = {k: torch.zeros_like(v) for k, v in batch["x_t"].items()}
    batch["x_sc_is_active"] = torch.zeros((), device=device)
    return batch


def _build_feature_factory(feats: list[str], mode: str, nn_kwargs: dict, latent_dim: int) -> FeatureFactory:
    """Instantiate a :class:`FeatureFactory` containing only the requested
    features. We bypass its random ``Linear``/``LayerNorm`` projection by
    calling ``feat_creators[i]`` directly downstream."""
    return FeatureFactory(
        feats=feats,
        dim_feats_out=1,
        use_ln_out=False,
        mode=mode,
        latent_dim=latent_dim,
        **nn_kwargs,
    )


def _compare_features(
    factory: FeatureFactory,
    feat_names: list[str],
    batch_a,
    batch_b,
    mode: str,
) -> list[str]:
    """Run each feature creator on both batches and return mismatch strings."""
    mismatches: list[str] = []
    for name, creator in zip(feat_names, factory.feat_creators, strict=True):
        torch.manual_seed(SEED)
        out_a = creator(batch_a)
        torch.manual_seed(SEED)
        out_b = creator(batch_b)
        if out_a.shape != out_b.shape:
            mismatches.append(f"[{mode}] {name}: shape mismatch {tuple(out_a.shape)} vs {tuple(out_b.shape)}")
            continue
        if not _values_equal(out_a, out_b):
            mismatches.append(f"[{mode}] {name}: tensor values differ")
    return mismatches


def _move_batch_to_device(batch, device: torch.device) -> None:
    """Move all tensors in a :class:`PaddedBatchedDataSample` (and its nested
    ``x_*``/``t`` dicts and padding masks) onto ``device`` in place."""
    for k, v in list(batch.data_samples.tensors.items()):
        batch.data_samples.tensors[k] = v.to(device)
    for k, v in list(batch.padding_masks.items()):
        batch.padding_masks[k] = v.to(device)
    for nested_key in ("x_1", "x_t", "x_sc", "x_recycle", "x_0", "t"):
        if nested_key in batch:
            batch[nested_key] = {dm: t.to(device) for dm, t in batch[nested_key].items()}


# ---------------------------------------------------------------------------
# Test 2: per-feature parity, val pipeline only, expandable vs. plain
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def autoencoder_and_fm():
    """Load the autoencoder and instantiate the production flow matcher once
    per module. CUDA-only."""
    if not torch.cuda.is_available():
        pytest.skip(reason="Test 2 requires CUDA")
    if AUTOENCODER_CKPT_PATH is None or not AUTOENCODER_CKPT_PATH.is_file():
        pytest.skip(f"Autoencoder checkpoint not found: {AUTOENCODER_CKPT_PATH} (set DATA_PATH)")

    device = torch.device("cuda")

    cfg_train = OmegaConf.load(TRAIN_CONFIG_PATH)
    cfg_nn = OmegaConf.load(NN_CONFIG_PATH)

    autoencoder = AutoEncoder.load_from_checkpoint(str(AUTOENCODER_CKPT_PATH), strict=True, weights_only=False)
    autoencoder = autoencoder.to(device).eval()
    for p in autoencoder.parameters():
        p.requires_grad = False
    latent_dim = autoencoder.latent_dim

    OmegaConf.set_struct(cfg_train, False)
    cfg_train.product_flowmatcher.local_latents.dim = latent_dim
    fm = ProductSpaceFlowMatcher(cfg_train).to(device)

    return autoencoder, fm, cfg_nn, latent_dim, device


@pytest.mark.parametrize("pdb_id", PDB_IDS)
def test_expandable_segment_features_match_unconditional(
    pdb_id: str,
    autoencoder_and_fm,
    gpu,  # noqa: F811
) -> None:
    """Plain val-pipeline run vs. val-pipeline run with the first ``N_REMOVE``
    residues replaced by a realized expandable segment must produce identical
    per-feature tensors at every token."""
    if pdb_id == "4ndz":
        pytest.skip("4ndz is too large for the GPU CI memory budget")

    autoencoder, fm, cfg_nn, latent_dim, device = autoencoder_and_fm

    base = _load_pdb_for_test(pdb_id, add_missing_atoms=True)
    base["example_id"] = pdb_id
    base["atom_array"] = _preclean_input(base["atom_array"])

    input_a = {**base, "atom_array": base["atom_array"].copy()}
    out_a = _run_validation_pipeline(input_a)

    modified_aa = _replace_prefix_with_expandable_segment(base["atom_array"].copy(), n_residues=N_REMOVE)
    input_b = {**base, "atom_array": modified_aa}
    out_b = _run_validation_pipeline(input_b)

    batch_a = _prepare_inference_batch(out_a, autoencoder, fm, latent_dim, device)
    batch_b = _prepare_inference_batch(out_b, autoencoder, fm, latent_dim, device)

    n_a = batch_a["padding_mask"].shape[1]
    n_b = batch_b["padding_mask"].shape[1]
    assert n_a == n_b, f"Token count differs: plain={n_a}, expanded={n_b}"

    # The NN config uses ``${nn.xxx}`` interpolations that only resolve when
    # nested under a parent config; wrap before resolving.
    cfg_nn_resolved = OmegaConf.create({"nn": cfg_nn})
    nn_kwargs = OmegaConf.to_container(cfg_nn_resolved.nn, resolve=True)
    assert isinstance(nn_kwargs, dict)

    # Pull the production feature lists from the config and apply optional
    # extra/remove overrides for tests. The lists are popped because they are
    # not constructor kwargs of :class:`FeatureFactory`.
    seq_feats = _resolve_feature_list(nn_kwargs.pop("feats_seq", []), EXTRA_SEQ_FEATURES, REMOVE_SEQ_FEATURES)
    seq_cond_feats = _resolve_feature_list(
        nn_kwargs.pop("feats_cond_seq", []), EXTRA_SEQ_COND_FEATURES, REMOVE_SEQ_COND_FEATURES
    )
    pair_repr_feats = _resolve_feature_list(
        nn_kwargs.pop("feats_pair_repr", []), EXTRA_PAIR_REPR_FEATURES, REMOVE_PAIR_REPR_FEATURES
    )
    pair_cond_feats = _resolve_feature_list(
        nn_kwargs.pop("feats_pair_cond", []), EXTRA_PAIR_COND_FEATURES, REMOVE_PAIR_COND_FEATURES
    )

    seq_factory = _build_feature_factory(seq_feats, "seq", nn_kwargs, latent_dim).to(device)
    seq_cond_factory = _build_feature_factory(seq_cond_feats, "seq", nn_kwargs, latent_dim).to(device)
    pair_repr_factory = _build_feature_factory(pair_repr_feats, "pair", nn_kwargs, latent_dim).to(device)
    pair_cond_factory = _build_feature_factory(pair_cond_feats, "pair", nn_kwargs, latent_dim).to(device)

    mismatches: list[str] = []
    mismatches += _compare_features(seq_factory, seq_feats, batch_a, batch_b, "seq")
    mismatches += _compare_features(seq_cond_factory, seq_cond_feats, batch_a, batch_b, "seq_cond")
    mismatches += _compare_features(pair_repr_factory, pair_repr_feats, batch_a, batch_b, "pair_repr")
    mismatches += _compare_features(pair_cond_factory, pair_cond_feats, batch_a, batch_b, "pair_cond")

    assert not mismatches, (
        f"Per-feature mismatch between plain and expanded-segment val pipelines "
        f"for {pdb_id}:\n  " + "\n  ".join(mismatches)
    )
