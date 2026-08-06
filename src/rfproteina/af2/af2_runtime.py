"""AlphaFold 2 runtime used inside the RFProteina AF2 container.

The input path in this module is deliberately database free: every chain gets a
one-row, query-only MSA and templates are supplied directly as mmCIF strings.
"""

from __future__ import annotations

import copy
import importlib
import importlib.util
import io
import json
import os
import pathlib
import site
from collections.abc import Mapping, Sequence
from typing import Any

import jax
import jaxlib
import numpy as np
from alphafold.common import confidence, protein, residue_constants
from alphafold.data import feature_processing, parsers, pipeline, pipeline_multimer
from alphafold.model import config, data, model
from packaging.version import Version

DEFAULT_BUCKETS = (64, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048, 2560, 3072)
DEFAULT_MONOMER_MODEL = "model_1_ptm"
DEFAULT_MULTIMER_MODEL = "model_1_multimer_v3"

_META_KEY = "__rfproteina_metadata__"
_PROCESSED_FEATURES_KEY = "__rfproteina_processed_features__"
_PAE_HEAD_KEY = "__rfproteina_pae_head__"
_EXTENDED_PTM_CONFIG = {"calc_extended_ptm": True, "use_probs_extended": True}


def _compile_jit_kwargs(compile_mode: str) -> dict[str, Any]:
    """Return ColabFold's XLA compile options for the requested mode."""
    if compile_mode not in {"fast", "tuned", "full"}:
        raise ValueError(f"Unsupported AF2 compile mode {compile_mode!r}; choose fast, tuned, or full")
    if Version(jaxlib.__version__) < Version("0.10.1"):
        tuned = "block_m: 64 block_n: 64 block_k: 16 split_k: 1 " "num_stages: 3 num_warps: 4 num_ctas: 1"
    else:
        tuned = "block_m: 64 block_n: 64 block_k: 16 num_stages: 3 num_warps: 4 num_ctas: 1"
    compiler_options = {
        "fast": {"xla_gpu_enable_triton_gemm": False},
        "tuned": {"xla_gpu_override_gemm_autotuner": tuned},
        "full": None,
    }[compile_mode]
    return {"compiler_options": compiler_options} if compiler_options else {}


def _ensure_pallas_compiler_params(pallas_triton: Any | None = None) -> None:
    """Bridge JAX 0.5's TritonCompilerParams name used by Pallas kernels."""
    if pallas_triton is None:
        try:
            pallas_triton = importlib.import_module("jax.experimental.pallas.triton")
        except ImportError:
            return
    if not hasattr(pallas_triton, "CompilerParams") and hasattr(pallas_triton, "TritonCompilerParams"):
        pallas_triton.CompilerParams = pallas_triton.TritonCompilerParams


def _ensure_colabfold_confidence_metrics(source_module: Any | None = None) -> None:
    """Expose ColabFold confidence helpers expected by its model package."""
    if hasattr(confidence, "get_confidence_metrics"):
        return
    if source_module is None:
        current_file = pathlib.Path(getattr(confidence, "__file__", "")).resolve()
        for package_dir in site.getsitepackages():
            candidate = pathlib.Path(package_dir) / "alphafold/common/confidence.py"
            if not candidate.is_file() or candidate.resolve() == current_file:
                continue
            spec = importlib.util.spec_from_file_location("_rfproteina_colabfold_confidence", candidate)
            if spec is None or spec.loader is None:
                continue
            source_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(source_module)
            break
    if source_module is None or not hasattr(source_module, "get_confidence_metrics"):
        return
    for name in dir(source_module):
        if not name.startswith("__") and not hasattr(confidence, name):
            setattr(confidence, name, getattr(source_module, name))


def _configure_model_for_inference(
    model_config: Any,
    *,
    use_pallas: bool,
    use_bfloat16: bool,
) -> None:
    if use_pallas and not use_bfloat16:
        raise ValueError("AF2 use_pallas requires bfloat16 inference")
    if use_pallas:
        _ensure_pallas_compiler_params()
    _ensure_colabfold_confidence_metrics()
    model_config.model.global_config.bfloat16 = bool(use_bfloat16)
    model_config.model.global_config.use_pallas = bool(use_pallas)


def _configure_database_free_limits(model_config: Any, *, multimer_mode: bool) -> None:
    """Use one query row plus one masked extra row for database-free inference."""
    if multimer_mode:
        evoformer = model_config.model.embeddings_and_evoformer
        evoformer.num_msa = 1
        evoformer.num_extra_msa = 1
    else:
        model_config.data.eval.max_msa_clusters = 1
        model_config.data.eval.max_templates = 1
        model_config.data.common.max_extra_msa = 1
        model_config.data.common.reduce_msa_clusters_by_max_templates = False


def _configure_masked_msa_replace_fraction(
    model_config: Any,
    *,
    multimer_mode: bool,
    replace_fraction: float,
) -> None:
    """Set AF2 BERT MSA mask rate for the next ``process_features`` call."""
    if not 0.0 <= float(replace_fraction) <= 1.0:
        raise ValueError(f"pct_seq_mask / replace_fraction must be in [0, 1], got {replace_fraction!r}")
    fraction = float(replace_fraction)
    if multimer_mode:
        model_config.model.embeddings_and_evoformer.masked_msa.replace_fraction = fraction
    else:
        model_config.data.eval.masked_msa_replace_fraction = fraction


def _all_atom_positions_from_cif(
    cif_string: str,
    chain_ids: Sequence[str],
) -> np.ndarray:
    """Assemble ColabFold-style ``all_atom_positions`` ``[L, 37, 3]`` from a query CIF."""
    if not chain_ids:
        raise ValueError("initial_guess requires at least one chain id")
    per_chain: list[np.ndarray] = []
    for chain_id in chain_ids:
        chain_protein = _protein_chain_from_mmcif(cif_string, str(chain_id))
        if chain_protein.aatype.size == 0:
            raise ValueError(f"initial_guess CIF has no residues for chain {chain_id!r}")
        per_chain.append(np.asarray(chain_protein.atom_positions, dtype=np.float32))
    return np.concatenate(per_chain, axis=0)


def _pad_all_atom_positions(positions: np.ndarray, target_len: int) -> np.ndarray:
    """Pad or validate residue axis to match processed AF2 feature length."""
    positions = np.asarray(positions, dtype=np.float32)
    if positions.ndim != 3 or positions.shape[1:] != (residue_constants.atom_type_num, 3):
        raise ValueError("all_atom_positions must have shape [L, atom_type_num, 3], " f"got {positions.shape}")
    if positions.shape[0] == target_len:
        return positions
    if positions.shape[0] < target_len:
        return _pad_array(positions, (target_len, *positions.shape[1:]))
    raise ValueError(f"initial_guess has {positions.shape[0]} residues but processed features expect {target_len}")


def nearest_bucket(num_res: int, buckets: Sequence[int] = DEFAULT_BUCKETS) -> int:
    """Return the smallest configured bucket that can contain ``num_res``."""
    if num_res <= 0:
        raise ValueError(f"num_res must be positive, got {num_res}")
    for bucket in buckets:
        if num_res <= bucket:
            return int(bucket)
    raise ValueError(f"Sequence has {num_res} residues, exceeding the largest AF2 bucket ({buckets[-1]})")


def _resolve_path_or_string(value: str) -> str:
    if "\n" not in value and os.path.exists(value):
        return pathlib.Path(value).read_text()
    return value


def _query_msa(sequence: str, chain_id: str) -> parsers.Msa:
    return parsers.Msa(
        sequences=[sequence],
        deletion_matrix=[[0] * len(sequence)],
        descriptions=[f"query_{chain_id}"],
    )


def _empty_template_features(num_res: int) -> dict[str, np.ndarray]:
    num_types = len(residue_constants.restypes_with_x_and_gap)
    return {
        "template_aatype": np.zeros((0, num_res, num_types), dtype=np.float32),
        "template_all_atom_masks": np.zeros((0, num_res, residue_constants.atom_type_num), dtype=np.float32),
        "template_all_atom_positions": np.zeros((0, num_res, residue_constants.atom_type_num, 3), dtype=np.float32),
        "template_domain_names": np.asarray([], dtype=object),
        "template_sequence": np.asarray([], dtype=object),
        "template_sum_probs": np.zeros((0,), dtype=np.float32),
    }


def _protein_chain_from_mmcif(mmcif_string: str, chain_id: str) -> protein.Protein:
    """Read one template chain, tolerating a single-chain CIF with a renamed ID."""
    try:
        template_protein = protein.from_mmcif_string(mmcif_string, chain_id=chain_id)
        if template_protein.aatype.size:
            return template_protein
    except (KeyError, ValueError):
        pass

    template_protein = protein.from_mmcif_string(mmcif_string)
    unique_chains = np.unique(template_protein.chain_index)
    if unique_chains.size != 1:
        raise ValueError(
            f"Template for query chain {chain_id!r} does not contain that chain and "
            f"contains {unique_chains.size} coordinate chains"
        )
    return template_protein


def _template_features(
    *,
    query_sequence: str,
    chain_id: str,
    template_spec: Mapping[str, Any] | None,
) -> dict[str, np.ndarray]:
    """Construct one AF2 template row using an explicit residue-index mapping."""
    num_res = len(query_sequence)
    if template_spec is None:
        return _empty_template_features(num_res)

    mmcif_string = _resolve_path_or_string(str(template_spec["mmcif"]))
    template_protein = _protein_chain_from_mmcif(mmcif_string, chain_id)

    query_indices_value = template_spec.get("query_indices")
    template_indices_value = template_spec.get("template_indices")
    if query_indices_value is None:
        if len(template_protein.aatype) != num_res:
            raise ValueError(
                f"Identity template mapping for chain {chain_id!r} requires equal sequence lengths "
                f"({num_res} query residues vs {len(template_protein.aatype)} template residues)"
            )
        query_indices = list(range(num_res))
        template_indices = query_indices
    else:
        if template_indices_value is None:
            raise ValueError(f"Template for chain {chain_id!r} has query_indices but no template_indices")
        query_indices = [int(index) for index in query_indices_value]
        template_indices = [int(index) for index in template_indices_value]
        if len(query_indices) != len(template_indices):
            raise ValueError(
                f"Template index lists for chain {chain_id!r} have different lengths: "
                f"{len(query_indices)} and {len(template_indices)}"
            )

    positions = np.zeros((num_res, residue_constants.atom_type_num, 3), dtype=np.float32)
    masks = np.zeros((num_res, residue_constants.atom_type_num), dtype=np.float32)
    template_sequence = ["-"] * num_res
    restypes = [*residue_constants.restypes, "X"]

    for query_index, template_index in zip(query_indices, template_indices, strict=True):
        if not 0 <= query_index < num_res:
            raise IndexError(f"Query template index {query_index} is out of range for chain {chain_id!r}")
        if not 0 <= template_index < len(template_protein.aatype):
            raise IndexError(f"Source template index {template_index} is out of range for chain {chain_id!r}")
        positions[query_index] = template_protein.atom_positions[template_index]
        masks[query_index] = template_protein.atom_mask[template_index]
        template_sequence[query_index] = restypes[int(template_protein.aatype[template_index])]

    sequence_string = "".join(template_sequence)
    aatype = residue_constants.sequence_to_onehot(
        sequence_string,
        residue_constants.HHBLITS_AA_TO_ID,
    ).astype(np.float32)
    return {
        "template_aatype": aatype[None],
        "template_all_atom_masks": masks[None],
        "template_all_atom_positions": positions[None],
        "template_domain_names": np.asarray([f"template_{chain_id}".encode()], dtype=object),
        "template_sequence": np.asarray([sequence_string.encode()], dtype=object),
        "template_sum_probs": np.asarray([1.0], dtype=np.float32),
    }


def _make_chain_features(
    sequence: str,
    chain_id: str,
    template_spec: Mapping[str, Any] | None,
) -> dict[str, np.ndarray]:
    sequence_features = pipeline.make_sequence_features(
        sequence=sequence,
        description=f"chain_{chain_id}",
        num_res=len(sequence),
    )
    msa_features = pipeline.make_msa_features([_query_msa(sequence, chain_id)])
    return {
        **sequence_features,
        **msa_features,
        **_template_features(
            query_sequence=sequence,
            chain_id=chain_id,
            template_spec=template_spec,
        ),
    }


def parse_protein_chains(cif_string: str) -> dict[str, str]:
    """Parse coordinate-chain sequences without requiring ``_chem_comp`` loops."""
    from Bio.PDB import MMCIFParser

    structure = MMCIFParser(QUIET=True).get_structure("rfproteina_query", io.StringIO(cif_string))
    model_structure = next(structure.get_models(), None)
    if model_structure is None:
        raise ValueError("Input CIF contains no coordinate model")
    chains: dict[str, str] = {}
    for chain in model_structure:
        sequence = "".join(
            residue_constants.restype_3to1[residue.get_resname().upper()]
            for residue in chain
            if residue.get_resname().upper() in residue_constants.restype_3to1
        )
        if sequence:
            chains[str(chain.id)] = sequence
    if not chains:
        raise ValueError("Input CIF contains no protein chains")
    return chains


def _add_query_pairing_features(chain_features: dict[str, np.ndarray]) -> None:
    """Expose the query row as the all-sequence MSA used by multimer pairing."""
    for name in ("msa", "deletion_matrix_int", "num_alignments", "msa_species_identifiers"):
        chain_features[f"{name}_all_seq"] = np.array(chain_features[name], copy=True)


def _pad_array(array: np.ndarray, target_shape: Sequence[int], value: int | float = 0) -> np.ndarray:
    if len(array.shape) != len(target_shape):
        raise ValueError(f"Cannot pad rank-{array.ndim} array to shape {tuple(target_shape)}")
    if any(current > target for current, target in zip(array.shape, target_shape, strict=True)):
        raise ValueError(f"Cannot pad shape {array.shape} down to {tuple(target_shape)}")
    padding = [(0, target - current) for current, target in zip(array.shape, target_shape, strict=True)]
    return np.pad(array, padding, constant_values=value)


def make_fixed_size_numpy(
    features: Mapping[str, np.ndarray],
    *,
    num_res: int,
    msa_cluster_size: int = 1,
    num_templates: int = 0,
) -> dict[str, np.ndarray]:
    """NumPy equivalent of AF2 ``make_fixed_size`` for multimer raw features."""
    output = {name: np.asarray(value) for name, value in features.items()}
    residue_axis = {
        "aatype",
        "all_atom_mask",
        "all_atom_positions",
        "asym_id",
        "between_segment_residues",
        "deletion_mean",
        "entity_id",
        "entity_mask",
        "residue_index",
        "seq_mask",
        "sym_id",
    }
    msa_features = {"bert_mask", "deletion_matrix", "msa", "msa_mask"}
    template_features = {"template_aatype", "template_all_atom_mask", "template_all_atom_positions"}

    for name, value in list(output.items()):
        if name in residue_axis and value.ndim:
            output[name] = _pad_array(value, (num_res, *value.shape[1:]))
        elif name in msa_features:
            pad_value = residue_constants.restype_num + 1 if name == "msa" else 0
            output[name] = _pad_array(value, (msa_cluster_size, num_res, *value.shape[2:]), pad_value)
        elif name == "cluster_bias_mask":
            output[name] = _pad_array(value, (msa_cluster_size,))
        elif name in template_features:
            output[name] = _pad_array(value, (num_templates, num_res, *value.shape[2:]))
    return output


def _crop_template_rows(features: dict[str, np.ndarray], num_templates: int) -> None:
    for name in ("template_aatype", "template_all_atom_mask", "template_all_atom_positions"):
        if name in features:
            features[name] = features[name][:num_templates]
    if "num_templates" in features:
        features["num_templates"] = np.asarray(num_templates, dtype=np.int32)


def _prepare_database_free_multimer_features(
    features: dict[str, np.ndarray],
    *,
    bucket: int,
) -> dict[str, np.ndarray]:
    """Pad masked MSA/template rows so Pallas never sees zero-sized stacks."""
    features = pipeline_multimer.pad_msa(dict(features), min_num_seq=2)
    _crop_template_rows(features, num_templates=1)
    return make_fixed_size_numpy(
        features,
        num_res=bucket,
        msa_cluster_size=2,
        num_templates=1,
    )


def _build_multimer_features(
    chains: Mapping[str, str],
    templates: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, np.ndarray], list[str], list[int]]:
    all_chain_features: dict[str, dict[str, np.ndarray]] = {}
    for chain_id, sequence in chains.items():
        chain_features = _make_chain_features(sequence, chain_id, templates.get(chain_id))
        _add_query_pairing_features(chain_features)
        all_chain_features[chain_id] = pipeline_multimer.convert_monomer_features(
            chain_features,
            chain_id=chain_id,
        )

    assembled = pipeline_multimer.add_assembly_features(all_chain_features)
    chain_ids, chain_lengths = _multimer_chain_layout(assembled)
    return feature_processing.pair_and_merge(all_chain_features=assembled), chain_ids, chain_lengths


def _multimer_chain_layout(
    assembled: Mapping[str, Mapping[str, np.ndarray]],
) -> tuple[list[str], list[int]]:
    """Return chain IDs and lengths in the residue order used by AF2 tensors."""
    chain_ids = [str(np.asarray(features["auth_chain_id"]).item()) for features in assembled.values()]
    chain_lengths = [int(np.asarray(features["seq_length"]).item()) for features in assembled.values()]
    return chain_ids, chain_lengths


def chain_count(payload: Mapping[str, Any]) -> int:
    """Return the number of protein chains in a manifest payload."""
    if "cif" not in payload:
        raise KeyError("AF2 payload is missing required key 'cif'")
    return len(parse_protein_chains(_resolve_path_or_string(str(payload["cif"]))))


def residue_count(payload: Mapping[str, Any]) -> int:
    """Return total protein residues, using manifest metadata when available."""
    if "num_res" in payload:
        return int(payload["num_res"])
    if "cif" not in payload:
        raise KeyError("AF2 payload is missing required key 'cif'")
    chains = parse_protein_chains(_resolve_path_or_string(str(payload["cif"])))
    return sum(len(sequence) for sequence in chains.values())


def build_raw_features_from_payload(
    payload: Mapping[str, Any],
    *,
    buckets: Sequence[int] = DEFAULT_BUCKETS,
) -> dict[str, Any]:
    """Build database-free AF2 features from a manifest payload.

    Required payload key: ``cif`` (inline mmCIF or a readable path). Optional
    keys are ``template``, ``model_name``, ``seed``, ``initial_guess``, and
    ``pct_seq_mask``.
    """
    if "cif" not in payload:
        raise KeyError("AF2 payload is missing required key 'cif'")
    cif_string = _resolve_path_or_string(str(payload["cif"]))
    chains = parse_protein_chains(cif_string)
    templates = payload.get("template") or {}
    unknown_template_chains = set(templates) - set(chains)
    if unknown_template_chains:
        raise ValueError(f"Templates supplied for unknown chains: {sorted(unknown_template_chains)}")

    replace_fraction = float(payload.get("pct_seq_mask", 0.15))
    if not 0.0 <= replace_fraction <= 1.0:
        raise ValueError(f"pct_seq_mask must be in [0, 1], got {replace_fraction!r}")
    initial_guess = bool(payload.get("initial_guess", False))

    multimer_mode = len(chains) >= 2
    default_model = DEFAULT_MULTIMER_MODEL if multimer_mode else DEFAULT_MONOMER_MODEL
    model_name = str(payload.get("model_name") or default_model)
    if multimer_mode != ("multimer" in model_name):
        required = "a multimer model" if multimer_mode else "a model_*_ptm monomer model"
        raise ValueError(f"{len(chains)} input chain(s) require {required}, got {model_name!r}")
    if not multimer_mode and not model_name.endswith("_ptm"):
        raise ValueError(f"Monomer inference requires a model_*_ptm model, got {model_name!r}")

    num_res = sum(map(len, chains.values()))
    bucket = nearest_bucket(num_res, buckets)
    feature_chain_ids = list(chains)
    feature_chain_lengths = [len(sequence) for sequence in chains.values()]
    if multimer_mode:
        raw_features, feature_chain_ids, feature_chain_lengths = _build_multimer_features(chains, templates)
        raw_features = _prepare_database_free_multimer_features(
            raw_features,
            bucket=bucket,
        )
    else:
        chain_id, sequence = next(iter(chains.items()))
        raw_features = _make_chain_features(sequence, chain_id, templates.get(chain_id))

    metadata: dict[str, Any] = {
        "chain_ids": list(chains),
        "chain_lengths": [len(sequence) for sequence in chains.values()],
        "feature_chain_ids": feature_chain_ids,
        "feature_chain_lengths": feature_chain_lengths,
        "num_res": num_res,
        "bucket": bucket,
        "multimer_mode": multimer_mode,
        "model_name": model_name,
        "seed": int(payload.get("seed", 0)),
        "pct_seq_mask": replace_fraction,
        "initial_guess": initial_guess,
    }
    if initial_guess:
        metadata["cif"] = cif_string
    raw_features[_META_KEY] = metadata
    return raw_features


def _first_ensemble(value: np.ndarray, multimer_mode: bool) -> np.ndarray:
    array = np.asarray(value)
    return array if multimer_mode else array[0]


def _prediction_protein(
    prediction_result: Mapping[str, Any],
    processed_features: Mapping[str, np.ndarray],
    metadata: Mapping[str, Any],
) -> protein.Protein:
    multimer_mode = bool(metadata["multimer_mode"])
    num_res = int(metadata["num_res"])
    aatype = _first_ensemble(processed_features["aatype"], multimer_mode)[:num_res]
    residue_index = _first_ensemble(processed_features["residue_index"], multimer_mode)[:num_res] + 1
    if multimer_mode:
        asym_id = np.asarray(processed_features["asym_id"])[:num_res]
        unique_asym_ids = list(dict.fromkeys(asym_id.tolist()))
        asym_to_index = {asym: index for index, asym in enumerate(unique_asym_ids)}
        chain_index = np.asarray([asym_to_index[asym] for asym in asym_id], dtype=np.int32)
    else:
        chain_index = np.zeros((num_res,), dtype=np.int32)

    atom_positions = np.asarray(prediction_result["structure_module"]["final_atom_positions"])[:num_res]
    atom_mask = np.asarray(prediction_result["structure_module"]["final_atom_mask"])[:num_res]
    plddt = np.asarray(prediction_result["plddt"])[:num_res]
    b_factors = np.repeat(plddt[:, None], residue_constants.atom_type_num, axis=-1)
    return protein.Protein(
        aatype=aatype,
        atom_positions=atom_positions,
        atom_mask=atom_mask,
        residue_index=residue_index,
        chain_index=chain_index,
        b_factors=b_factors,
    )


def _chain_summary(
    prediction_result: Mapping[str, Any],
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    num_res = int(metadata["num_res"])
    chain_ids = list(metadata.get("feature_chain_ids") or metadata["chain_ids"])
    chain_lengths = [int(length) for length in (metadata.get("feature_chain_lengths") or metadata["chain_lengths"])]
    chain_index = np.concatenate(
        [np.full((length,), index, dtype=np.int32) for index, length in enumerate(chain_lengths)]
    )
    plddt = np.asarray(prediction_result["plddt"], dtype=np.float32)[:num_res]
    ptm = float(prediction_result.get("ptm", 0.0))
    iptm = float(prediction_result.get("iptm", 0.0))

    pae_value = prediction_result.get("predicted_aligned_error")
    if pae_value is None:
        raise ValueError(
            "AF2 prediction is missing predicted_aligned_error; refusing to report absent PAE as a perfect score"
        )
    pae = np.asarray(pae_value, dtype=np.float32)[:num_res, :num_res]

    chain_plddt: list[float] = []
    chain_pae: list[float] = []
    chain_pair_pae: list[list[float]] = []
    chain_pair_pae_min: list[list[float]] = []
    for chain_i in range(len(chain_ids)):
        indices_i = np.where(chain_index == chain_i)[0]
        chain_plddt.append(float(np.mean(plddt[indices_i])) / 100.0)
        chain_pae.append(float(np.mean(pae[np.ix_(indices_i, indices_i)])))
        pair_mean_row = []
        pair_min_row = []
        for chain_j in range(len(chain_ids)):
            indices_j = np.where(chain_index == chain_j)[0]
            block = pae[np.ix_(indices_i, indices_j)]
            pair_mean_row.append(float(np.mean(block)))
            pair_min_row.append(float(np.min(block)))
        chain_pair_pae.append(pair_mean_row)
        chain_pair_pae_min.append(pair_min_row)

    chain_ptm = [ptm] * len(chain_ids)
    chain_pair_iptm = np.zeros((len(chain_ids), len(chain_ids)), dtype=np.float32)
    logits = breaks = None
    raw_pae = prediction_result.get(_PAE_HEAD_KEY)
    if isinstance(raw_pae, Mapping):
        logits = np.asarray(raw_pae["logits"])[:num_res, :num_res]
        breaks = np.asarray(raw_pae["breaks"])

    if logits is not None and breaks is not None:
        for chain_i in range(len(chain_ids)):
            chain_indices = np.where(chain_index == chain_i)[0]
            chain_ptm[chain_i] = float(
                confidence.predicted_tm_score(
                    logits=logits[np.ix_(chain_indices, chain_indices)],
                    breaks=breaks,
                )
            )
            for chain_j in range(chain_i, len(chain_ids)):
                pair_mask = (chain_index == chain_i) | (chain_index == chain_j)
                pair_indices = np.where(pair_mask)[0]
                pair_score = confidence.predicted_tm_score(
                    logits=logits[np.ix_(pair_indices, pair_indices)],
                    breaks=breaks,
                    asym_id=chain_index[pair_mask] if chain_i != chain_j else None,
                    interface=chain_i != chain_j,
                )
                chain_pair_iptm[chain_i, chain_j] = pair_score
                chain_pair_iptm[chain_j, chain_i] = pair_score
    else:
        np.fill_diagonal(chain_pair_iptm, np.asarray(chain_ptm, dtype=np.float32))
        if len(chain_ids) > 1:
            chain_pair_iptm[~np.eye(len(chain_ids), dtype=bool)] = iptm

    overall_plddt = float(np.mean(plddt)) / 100.0
    ranking_score = 0.8 * iptm + 0.2 * ptm if len(chain_ids) > 1 else overall_plddt
    summary = {
        "overall_plddt": round(overall_plddt, 4),
        "overall_pae": round(float(np.mean(pae)), 4),
        "ptm": round(ptm, 4),
        "iptm": round(iptm, 4),
        "ranking_score": round(ranking_score, 4),
        "chain_ids": chain_ids,
        "chain_plddt": np.round(chain_plddt, 4).tolist(),
        "chain_ptm": np.round(chain_ptm, 4).tolist(),
        "chain_pae": np.round(chain_pae, 4).tolist(),
        "chain_pair_pae": np.round(chain_pair_pae, 4).tolist(),
        "chain_pair_pae_min": np.round(chain_pair_pae_min, 4).tolist(),
        "chain_pair_iptm": np.round(chain_pair_iptm, 4).tolist(),
    }
    if model_name := metadata.get("model_name"):
        summary["af2_model_name"] = str(model_name)
    return summary


def write_outputs(
    *,
    prediction_result: Mapping[str, Any],
    output_dir: os.PathLike[str] | str,
    job_name: str,
    processed_features: Mapping[str, np.ndarray] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Write AF2 PDB/mmCIF outputs and an AF3-shaped confidence summary."""
    if processed_features is None:
        processed_features = prediction_result.get(_PROCESSED_FEATURES_KEY)
    if metadata is None:
        metadata = prediction_result.get(_META_KEY)
    if processed_features is None or metadata is None:
        raise ValueError("write_outputs needs processed features and RFProteina metadata")

    output_path = pathlib.Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    predicted_protein = _prediction_protein(prediction_result, processed_features, metadata)
    model_type = "Multimer" if metadata["multimer_mode"] else "Monomer"
    (output_path / f"{job_name}_model.cif").write_text(
        protein.to_mmcif(predicted_protein, file_id=job_name, model_type=model_type)
    )
    (output_path / f"{job_name}_model.pdb").write_text(protein.to_pdb(predicted_protein))

    summary = _chain_summary(prediction_result, metadata)
    (output_path / f"{job_name}_summary_confidences.json").write_text(json.dumps(summary, indent=1))
    return summary


def prepare_input(
    *,
    payload: Mapping[str, Any],
    model_name: str,
    buckets: Sequence[int],
) -> dict[str, Any]:
    """Build padded raw features for one design (producer-thread helper)."""
    prepared_payload = dict(payload)
    prepared_payload["model_name"] = model_name
    return build_raw_features_from_payload(prepared_payload, buckets=buckets)


def predict_and_write(
    *,
    payload: Mapping[str, Any],
    prepared_input: Mapping[str, Any],
    model_name: str,
    model_runner: model.RunModel,
    output_dir: os.PathLike[str] | str,
) -> dict[str, Any]:
    """Run one resident model on prepared features and write AF2 outputs."""
    raw_features = dict(prepared_input)
    metadata = copy.deepcopy(raw_features.pop(_META_KEY, {}))
    metadata["model_name"] = model_name
    seed = int(payload.get("seed", metadata.get("seed", 0)))
    multimer_mode = bool(metadata.get("multimer_mode", "asym_id" in raw_features))
    if bool(model_runner.multimer_mode) != multimer_mode:
        input_mode = "multimer" if multimer_mode else "monomer"
        raise ValueError(f"Model {model_name!r} is incompatible with {input_mode} input features")
    if not multimer_mode:
        bucket = int(metadata.get("bucket", nearest_bucket(len(raw_features["residue_index"]))))
        model_runner.config.data.eval.crop_size = bucket
        model_runner.config.data.eval.max_msa_clusters = 1
        model_runner.config.data.eval.max_templates = 1
        model_runner.config.data.common.max_extra_msa = 1
        model_runner.config.data.common.reduce_msa_clusters_by_max_templates = False

    _configure_masked_msa_replace_fraction(
        model_runner.config,
        multimer_mode=multimer_mode,
        replace_fraction=float(metadata.get("pct_seq_mask", 0.15)),
    )
    processed_features = model_runner.process_features(raw_features, random_seed=seed)
    if metadata.get("initial_guess"):
        cif_string = metadata.get("cif")
        if not cif_string:
            raise KeyError("initial_guess requires CIF contents in feature metadata")
        chain_ids = list(metadata.get("feature_chain_ids") or metadata.get("chain_ids") or [])
        positions = _all_atom_positions_from_cif(str(cif_string), chain_ids)
        aatype = _first_ensemble(processed_features["aatype"], multimer_mode)
        processed_features["all_atom_positions"] = _pad_all_atom_positions(positions, int(aatype.shape[0]))

    raw_prediction = model_runner.predict(processed_features, random_seed=seed)
    if isinstance(raw_prediction, tuple):
        raw_prediction, _num_recycles = raw_prediction
    prediction = dict(raw_prediction)
    pae_head = prediction.pop("pae_matrix_with_logits", None)
    if isinstance(pae_head, Mapping):
        prediction[_PAE_HEAD_KEY] = pae_head
    prediction[_PROCESSED_FEATURES_KEY] = processed_features
    prediction[_META_KEY] = metadata
    job_name = str(payload.get("name", "design")).replace(" ", "_")
    return write_outputs(
        prediction_result=prediction,
        output_dir=output_dir,
        job_name=job_name,
    )


def load_model_runners(
    *,
    model_dir: os.PathLike[str] | str,
    model_names: Sequence[str],
    num_recycles: int = 3,
    gpu_device: int = 0,
    buckets: Sequence[int] = DEFAULT_BUCKETS,
    jax_compilation_cache_dir: os.PathLike[str] | str | None = None,
    use_pallas: bool = True,
    use_bfloat16: bool = True,
    compile_mode: str = "tuned",
) -> dict[str, model.RunModel]:
    """Load resident AF2 models and return ``{model_name: RunModel}``."""
    del buckets  # Applied per request during feature padding / crop_size setup.
    if use_pallas and not use_bfloat16:
        raise ValueError("AF2 use_pallas requires bfloat16 inference")
    jit_kwargs = _compile_jit_kwargs(compile_mode)
    if jax_compilation_cache_dir is not None:
        jax.config.update("jax_compilation_cache_dir", os.fspath(jax_compilation_cache_dir))

    # Force JAX to see the requested GPU before model construction.
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(int(gpu_device)))
    _ = jax.local_devices(backend="gpu")

    runners: dict[str, model.RunModel] = {}
    data_dir = os.fspath(model_dir)
    for model_name in model_names:
        model_config = config.model_config(model_name)
        multimer_mode = "multimer" in model_name
        _configure_model_for_inference(
            model_config,
            use_pallas=use_pallas,
            use_bfloat16=use_bfloat16,
        )
        _configure_database_free_limits(model_config, multimer_mode=multimer_mode)
        model_config.model.num_recycle = int(num_recycles)
        if not multimer_mode:
            model_config.data.common.num_recycle = int(num_recycles)
        params = data.get_model_haiku_params(model_name=model_name, data_dir=data_dir)
        runners[model_name] = model.RunModel(
            model_config,
            params,
            extended_ptm_config=_EXTENDED_PTM_CONFIG,
            jit_kwargs=jit_kwargs,
        )
    return runners
