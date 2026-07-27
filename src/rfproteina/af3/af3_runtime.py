# Copyright 2024 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# This file adapts AlphaFold3's Apache-licensed inference entrypoint for
# RFProteina's persistent manifest worker.
"""Checked-in AlphaFold3 package adapter for the persistent fold worker.

This module owns the CIF-to-input, model-runner, and output-writing behavior
that RFProteina needs from AlphaFold3. It intentionally replaces the old
runtime dependency on a separately mounted ``run_alphafold_from_cif.py``.

The implementation is based on the Apache-licensed inference entrypoint in
``sokrypton/alphafold3`` at commit
``d43d4971e68a2b88551288a670a397be427067ce``. The container still supplies the
pinned ``alphafold3`` Python package and its compiled extensions; it does not
supply or override this adapter.
"""

from __future__ import annotations

import dataclasses
import functools
import json
import os
import pathlib
from collections.abc import Callable, Sequence
from typing import Any

import alphafold3.cpp
import haiku as hk
import jax
import numpy as np
import tokamax
from absl import logging
from alphafold3.common import folding_input
from alphafold3.constants import chemical_components
from alphafold3.model import features, model, params, post_processing
from alphafold3.model.components import utils
from jax import numpy as jnp


def make_model_config(
    *,
    flash_attention_implementation: str = "triton",
    num_diffusion_samples: int = 1,
    num_recycles: int = 5,
) -> model.Model.Config:
    """Return an inference config for one resident parameter family."""
    config = model.Model.Config()
    config.global_config.flash_attention_implementation = flash_attention_implementation
    config.heads.diffusion.eval.num_samples = num_diffusion_samples
    config.num_recycles = num_recycles
    return config


class ModelRunner:
    """Load one Haiku checkpoint and expose a cached JIT inference function."""

    def __init__(
        self,
        config: model.Model.Config,
        device: jax.Device,
        model_dir: pathlib.Path,
        cache_dir: pathlib.Path | None = None,
    ) -> None:
        self._model_config = config
        self._device = device
        self._model_dir = model_dir
        self._autotune_cache_path = cache_dir / "tokamax_autotune.json" if cache_dir else None
        self._autotune_result = self._load_autotune_cache()
        self._autotune_attempted = self._autotune_result is not None

    def _load_autotune_cache(self) -> tokamax.AutotuningResult | None:
        if self._autotune_cache_path and self._autotune_cache_path.exists():
            try:
                with self._autotune_cache_path.open() as handle:
                    return tokamax.AutotuningResult.load(handle)
            except (OSError, ValueError):
                pass
        return None

    def _save_autotune_cache(self) -> None:
        if self._autotune_cache_path is None or self._autotune_result is None:
            return
        # Ranks racing on a shared cache dir must never observe a partial file.
        temp_path = self._autotune_cache_path.with_suffix(f".{os.getpid()}.tmp")
        with temp_path.open("w") as handle:
            self._autotune_result.dump(handle)
        os.replace(temp_path, self._autotune_cache_path)

    @functools.cached_property
    def model_params(self) -> hk.Params:
        return params.get_model_haiku_params(model_dir=self._model_dir)

    @functools.cached_property
    def _model(self) -> Callable[[jnp.ndarray, features.BatchDict], model.ModelResult]:
        @hk.transform
        def forward_fn(batch: dict) -> model.ModelResult:  # type: ignore[type-arg]
            return model.Model(self._model_config)(batch)

        return functools.partial(jax.jit(forward_fn.apply, device=self._device), self.model_params)

    def run_inference(self, featurised_example: features.BatchDict, rng_key: jnp.ndarray) -> model.ModelResult:
        featurised_example = jax.device_put(
            jax.tree_util.tree_map(jnp.asarray, utils.remove_invalidly_typed_feats(featurised_example)),
            self._device,
        )
        if not self._autotune_attempted and self._autotune_cache_path is not None:
            self._autotune_attempted = True
            try:
                self._autotune_result = tokamax.autotune(self._model, rng_key, featurised_example)
                self._save_autotune_cache()
            except Exception:
                # Autotuning is a performance optimisation, so any failure falls back to the
                # default kernels -- but log it, or an unsupported GPU and a real bug look alike.
                logging.exception("tokamax autotuning failed; falling back to default kernels")
                self._autotune_result = None

        if self._autotune_result is not None:
            with self._autotune_result:
                result = self._model(rng_key, featurised_example)
        else:
            result = self._model(rng_key, featurised_example)
        result = jax.tree.map(np.asarray, result)
        result = jax.tree.map(
            lambda value: value.astype(jnp.float32) if value.dtype == jnp.bfloat16 else value,
            result,
        )
        result = dict(result)
        result["__identifier__"] = self.model_params["__meta__"]["__identifier__"].tobytes()
        return result

    def extract_inference_results(
        self,
        batch: features.BatchDict,
        result: model.ModelResult,
        target_name: str,
    ) -> list[model.InferenceResult]:
        return list(model.Model.get_inference_result(batch=batch, result=result, target_name=target_name))


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class ResultsForSeed:
    seed: int
    inference_results: Sequence[model.InferenceResult]


def _resolve_path_or_string(value: str) -> str:
    if "\n" not in value and os.path.exists(value):
        return pathlib.Path(value).read_text()
    return value


def build_fold_input_from_cif(
    *,
    cif_string: str,
    cif_name: str,
    paired_msa: dict[str, str] | None,
    unpaired_msa: dict[str, str] | None,
    template: dict[str, Any] | None,
    user_ccd: str | None,
) -> folding_input.Input:
    """Create an AF3 input from a CIF and optional per-chain overrides."""
    user_ccd_string = _resolve_path_or_string(user_ccd) if user_ccd is not None else None
    ccd = chemical_components.Ccd(user_ccd=user_ccd_string)
    fold_input = folding_input.Input.from_mmcif(cif_string, ccd=ccd)
    fold_input = dataclasses.replace(fold_input, name=cif_name, user_ccd=user_ccd_string)

    paired_msa_by_chain = {chain_id: _resolve_path_or_string(value) for chain_id, value in (paired_msa or {}).items()}
    unpaired_msa_by_chain = {
        chain_id: _resolve_path_or_string(value) for chain_id, value in (unpaired_msa or {}).items()
    }
    template_by_chain: dict[str, dict[str, Any]] = {}
    for chain_id, value in (template or {}).items():
        if isinstance(value, str):
            template_by_chain[chain_id] = {
                "mmcif": _resolve_path_or_string(value),
                "query_indices": None,
                "template_indices": None,
            }
        else:
            template_by_chain[chain_id] = {
                "mmcif": _resolve_path_or_string(value["mmcif"]),
                "query_indices": value.get("query_indices"),
                "template_indices": value.get("template_indices"),
            }

    updated_chains = []
    for chain in fold_input.chains:
        if not isinstance(chain, folding_input.ProteinChain):
            updated_chains.append(chain)
            continue
        templates: Sequence[folding_input.Template] = ()
        if chain.id in template_by_chain:
            template_value = template_by_chain[chain.id]
            query_indices = template_value["query_indices"]
            template_indices = template_value["template_indices"]
            if query_indices is None:
                query_to_template = {index: index for index in range(len(chain.sequence))}
            else:
                if template_indices is None:
                    raise ValueError(f"Template for chain {chain.id!r} has query_indices but no template_indices")
                query_to_template = dict(zip(query_indices, template_indices, strict=True))
            templates = (
                folding_input.Template(
                    mmcif=template_value["mmcif"],
                    query_to_template_map=query_to_template,
                ),
            )
        chain._templates = templates
        chain._paired_msa = paired_msa_by_chain.get(chain.id, "")
        chain._unpaired_msa = unpaired_msa_by_chain.get(chain.id, "")
        updated_chains.append(chain)
    fold_input = dataclasses.replace(fold_input, chains=updated_chains)

    # The loop above fills MSAs in for protein chains only, so an RNA chain would reach
    # featurisation with unpaired_msa=None, which AF3 rejects; "" is its "custom MSA, no sequences".
    for chain in fold_input.rna_chains:
        if chain.unpaired_msa is None:
            chain._unpaired_msa = unpaired_msa_by_chain.get(chain.id, "")
    return fold_input


def _augment_summary_confidences(output_dir: pathlib.Path, job_name: str) -> None:
    """Add the aggregate confidence fields consumed by RFProteina."""
    confidence_path = output_dir / f"{job_name}_confidences.json"
    summary_path = output_dir / f"{job_name}_summary_confidences.json"
    if not confidence_path.exists() or not summary_path.exists():
        raise FileNotFoundError(f"Missing confidence outputs under {output_dir}")

    full_confidence = json.loads(confidence_path.read_text())
    summary = json.loads(summary_path.read_text())
    atom_plddts = np.asarray(full_confidence["atom_plddts"], dtype=np.float32)
    atom_chain_ids = np.asarray(full_confidence["atom_chain_ids"])
    token_chain_ids = np.asarray(full_confidence["token_chain_ids"])
    pae = np.asarray(full_confidence["pae"], dtype=np.float32)

    chain_ids = list(dict.fromkeys(token_chain_ids.tolist()))
    chain_plddt = []
    chain_pae = []
    chain_pair_pae = []
    for chain_id in chain_ids:
        atom_mask = atom_chain_ids == chain_id
        chain_plddt.append(float(np.mean(atom_plddts[atom_mask])) / 100.0 if np.any(atom_mask) else 0.0)
        token_indices = np.where(token_chain_ids == chain_id)[0]
        chain_pae.append(float(np.mean(pae[np.ix_(token_indices, token_indices)])))

        pair_row = []
        for other_chain_id in chain_ids:
            other_indices = np.where(token_chain_ids == other_chain_id)[0]
            pair_row.append(float(np.mean(pae[np.ix_(token_indices, other_indices)])))
        chain_pair_pae.append(pair_row)

    summary.update(
        {
            "chain_ids": chain_ids,
            "chain_plddt": [round(value, 4) for value in chain_plddt],
            "chain_pae": [round(value, 4) for value in chain_pae],
            "chain_pair_pae": [[round(value, 4) for value in row] for row in chain_pair_pae],
            "overall_plddt": round(float(np.mean(atom_plddts)) / 100.0, 4),
            "overall_pae": round(float(np.mean(pae)), 4),
        }
    )
    summary_path.write_text(json.dumps(summary, indent=1))


def write_outputs(
    *,
    all_inference_results: Sequence[ResultsForSeed],
    output_dir: os.PathLike[str] | str,
    job_name: str,
    compress_large_output_files: bool = False,
    save_full_outputs: bool = False,
) -> None:
    """Write the top-ranked model and compact confidences for one design."""
    output_path = pathlib.Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    ranked_results = [
        (float(result.metadata["ranking_score"]), seed_result.seed, sample_index, result)
        for seed_result in all_inference_results
        for sample_index, result in enumerate(seed_result.inference_results)
    ]
    if not ranked_results:
        raise RuntimeError(f"AF3 returned no inference samples for {job_name!r}")
    ranked_results.sort(key=lambda item: item[0], reverse=True)

    output_terms = None
    if save_full_outputs:
        output_terms = (pathlib.Path(alphafold3.cpp.__file__).parent / "OUTPUT_TERMS_OF_USE.md").read_text()
    post_processing.write_output(
        inference_result=ranked_results[0][3],
        output_dir=output_path,
        terms_of_use=output_terms,
        name=job_name,
        compress=compress_large_output_files,
    )
    _augment_summary_confidences(output_path, job_name)

    if save_full_outputs:
        samples_dir = output_path / "samples"
        samples_dir.mkdir(exist_ok=True)
        for _score, seed, sample_index, result in ranked_results:
            sample_name = f"{job_name}_seed-{seed}_sample-{sample_index}"
            post_processing.write_output(
                inference_result=result,
                output_dir=samples_dir,
                name=sample_name,
                compress=compress_large_output_files,
            )
    else:
        (output_path / f"{job_name}_confidences.json").unlink(missing_ok=True)
