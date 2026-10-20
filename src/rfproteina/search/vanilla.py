"""Vanilla generation strategy (no search)."""

from typing import Any

import numpy as np
import torch
from atomworks.io.utils.atom_array_plus import as_atom_array_plus_stack

from rfproteina.search.base_strategy import SearchStrategy

RESAMPLED_FORWARD_MARGINAL = "resampled_forward_marginal"
INVERSE_FOLDING_METHODS = {RESAMPLED_FORWARD_MARGINAL}


def resolve_inverse_folding_method(cfg_args: Any) -> str | None:
    """Validate and return the configured inverse-folding method."""
    if not cfg_args.get("inverse_folding", False):
        return None
    method = cfg_args.get("inverse_folding_method", RESAMPLED_FORWARD_MARGINAL)
    if method not in INVERSE_FOLDING_METHODS:
        raise ValueError(f"inverse_folding_method must be one of {sorted(INVERSE_FOLDING_METHODS)}, got {method!r}.")
    return method


def _validate_partial_t(partial_t_dict: dict[str, float]) -> dict[str, float]:
    """Reject start times at or above 1, where the score conversion is undefined.

    Raises:
      ValueError: If any modality's start time is >= 1.
    """
    at_or_above_one = {data_mode: t for data_mode, t in partial_t_dict.items() if t >= 1.0}
    if at_or_above_one:
        raise ValueError(
            "Partial-diffusion start time must be < 1 (t=1 is the clean state, and the score "
            f"conversion divides by 1 - t); resolved {at_or_above_one}. Lower partial_t_backbone / "
            "partial_t_latent, or raise partial_trajectory_fraction_steps."
        )
    return partial_t_dict


def resolve_partial_t(
    cfg_args: Any,
    fm: Any,
    sampling_model_args: dict,
    ref_nsteps: int = 10000,
) -> dict[str, float]:
    """Extract and validate partial diffusion config, returning per-modality t values.

    Args:
        cfg_args: ``model.inf_cfg.args`` (OmegaConf DictConfig or plain dict).
        fm: ProductSpaceFlowMatcher instance.
        sampling_model_args: Per-modality sampling config (``model.inf_cfg.model``).
        ref_nsteps: Dense schedule resolution for coupling explicit modality times.
            Fractional restarts use ``cfg_args.nsteps`` and its actual schedule.

    Returns:
        Per-modality partial_t dict keyed by data-mode name, or ``{}`` for full generation.

    Raises:
        ValueError: On invalid / conflicting config combinations.
    """
    pt_backbone = cfg_args.get("partial_t_backbone", None)
    pt_latent = cfg_args.get("partial_t_latent", None)
    do_couple = cfg_args.get("couple_partial_t", False)
    frac = cfg_args.get("partial_trajectory_fraction_steps", None)
    inverse_folding_method = resolve_inverse_folding_method(cfg_args)

    if inverse_folding_method is not None:
        if pt_backbone is not None or pt_latent is not None or do_couple or frac is not None:
            raise ValueError(
                "inverse_folding is mutually exclusive with partial_t_backbone, "
                "partial_t_latent, couple_partial_t, and partial_trajectory_fraction_steps."
            )
        return {"bb_ca": 0.0, "local_latents": 0.0}

    if frac is not None:
        if pt_backbone is not None or pt_latent is not None or do_couple:
            raise ValueError(
                "partial_trajectory_fraction_steps is mutually exclusive with "
                "partial_t_backbone, partial_t_latent, and couple_partial_t."
            )
        if not (0.0 < frac <= 1.0):
            raise ValueError(f"partial_trajectory_fraction_steps must be in (0, 1], got {frac}.")
        nsteps = cfg_args["nsteps"]
        remaining_steps = round(frac * nsteps)
        if remaining_steps < 1:
            raise ValueError(
                f"partial_trajectory_fraction_steps={frac} with nsteps={nsteps} rounds to zero denoising steps. "
                "Increase the fraction or nsteps."
            )
        ref_ts, _ = fm.sample_schedule(nsteps=nsteps, sampling_model_args=sampling_model_args)
        k = nsteps - remaining_steps
        return _validate_partial_t({dm: ref_ts[dm][k].item() for dm in fm.data_modes})

    n_set = (pt_backbone is not None) + (pt_latent is not None)

    if n_set == 0:
        return {}

    if n_set == 2:
        if do_couple:
            raise ValueError(
                "couple_partial_t=true requires exactly one of partial_t_backbone or "
                "partial_t_latent; got both. Remove the one to be derived."
            )
        return _validate_partial_t({"bb_ca": pt_backbone, "local_latents": pt_latent})

    # n_set == 1
    if not do_couple:
        which = "partial_t_backbone" if pt_backbone is not None else "partial_t_latent"
        raise ValueError(
            f"Only {which} is set. Either provide both partial_t_backbone and "
            "partial_t_latent, or set couple_partial_t=true to derive the missing value."
        )

    # Step-index coupling: find k in the specified modality's schedule, read all others at k.
    ref_ts, _ = fm.sample_schedule(nsteps=ref_nsteps, sampling_model_args=sampling_model_args)
    if pt_backbone is not None:
        ref_sched, ref_val = ref_ts["bb_ca"], pt_backbone
    else:
        ref_sched, ref_val = ref_ts["local_latents"], pt_latent
    above = ref_sched >= ref_val - 1e-6
    k = above.nonzero(as_tuple=True)[0][0].item() if above.any() else ref_nsteps
    return _validate_partial_t({dm: ref_ts[dm][k].item() for dm in fm.data_modes})


def build_partial_schedule(
    partial_t_dict: dict[str, float],
    nsteps: int,
    fm: Any,
    sampling_model_args: dict,
    *,
    full_nsteps: int | None = None,
) -> tuple[dict, dict]:
    """Build per-modality time schedules covering [partial_t, 1.0] with exactly nsteps steps.

    Fractional restarts slice the configured full schedule when ``full_nsteps`` is
    supplied. Explicit modality times use a dense reference to estimate an expanded
    schedule whose final ``nsteps`` intervals cover the requested range.

    Args:
        partial_t_dict: Per-modality starting t values.
        nsteps: Number of denoising steps to take.
        fm: ProductSpaceFlowMatcher instance.
        sampling_model_args: Per-modality sampling config.
        full_nsteps: Full trajectory length for fractional restarts. The starting
            times must match the first point of its final ``nsteps`` intervals.

    Returns:
        Tuple of (ts, gt) partial schedules, each a dict of tensors with
        ts[dm] shape [nsteps+1] and gt[dm] shape [nsteps].
    """
    if full_nsteps is not None:
        if not 1 <= nsteps <= full_nsteps:
            raise ValueError(f"Partial step count must be in [1, {full_nsteps}], got {nsteps}.")
        full_ts, full_gt = fm.sample_schedule(nsteps=full_nsteps, sampling_model_args=sampling_model_args)
        ts = {dm: full_ts[dm][-(nsteps + 1) :] for dm in fm.data_modes}
        gt = {dm: full_gt[dm][-nsteps:] for dm in fm.data_modes}
        for dm in fm.data_modes:
            if ts[dm][0] != ts[dm].new_tensor(partial_t_dict[dm]):
                raise ValueError(f"{dm}: partial corruption time must match the first time of the full schedule tail.")
        return ts, gt

    nsteps_dense = max(nsteps * 100, 1000)
    dense_ts, _ = fm.sample_schedule(nsteps=nsteps_dense, sampling_model_args=sampling_model_args)

    partial_ts: dict = {}
    partial_gt: dict = {}
    for data_mode in fm.data_modes:
        pt = float(partial_t_dict.get(data_mode, 0.0))

        above = dense_ts[data_mode] >= pt - 1e-6
        i_start = above.nonzero(as_tuple=True)[0][0].item() if above.any() else nsteps_dense
        n_above = max(nsteps_dense - i_start, 1)

        nsteps_expanded = round(nsteps * nsteps_dense / n_above)
        full_ts, full_gt = fm.sample_schedule(nsteps=nsteps_expanded, sampling_model_args=sampling_model_args)
        partial_ts[data_mode] = full_ts[data_mode][-(nsteps + 1) :]
        partial_gt[data_mode] = full_gt[data_mode][-nsteps:]

    return partial_ts, partial_gt


class VanillaStrategy(SearchStrategy):
    """Vanilla generation without search.

    This strategy simply generates structures without any search or optimization.
    It performs standard flow matching sampling and returns the generated structures.

    Example:
        >>> strategy = VanillaStrategy()
        >>> results = strategy.generate(model, batch)
    """

    def generate(
        self,
        model: Any,  # Proteina instance
        batch: dict,
    ) -> list[dict[str, Any]]:
        """Generate structures without search.

        Args:
            model: Proteina model instance
            batch: Input batch from dataloader

        Returns:
            List of generated-structure dicts.
        """
        # Preprocess batch
        batch = model.preprocess_batch_for_generation(batch)
        inverse_folding_method = resolve_inverse_folding_method(model.inf_cfg.args)

        # Extract clean samples (same as training)
        batch = model.add_clean_samples(batch)
        # x_1: generated tokens have clean values, non-generated have zeros

        # Extract dimensions from batch
        nsamples = batch["coords"].shape[0]
        n = batch["padding_mask"].shape[1]

        partial_t_dict = resolve_partial_t(model.inf_cfg.args, model.fm, model.inf_cfg.model)

        # Call simulation (batch already has atom37 format from pipeline)
        nsteps = model.inf_cfg.args.nsteps
        frac = model.inf_cfg.args.get("partial_trajectory_fraction_steps", None)
        device = model.device
        save_trajectory_every = model.inf_cfg.args.get("save_trajectory_every", 0)
        dump_noisy_trajectories = model.inf_cfg.args.get("dump_noisy_trajectories", False)
        bb_ca_only_trajectories = model.inf_cfg.args.get("bb_ca_only_trajectories", False)
        actual_nsteps = round(frac * nsteps) if partial_t_dict and frac is not None else nsteps
        recycle_depths = model.recycle_depths(actual_nsteps, full_nsteps=nsteps)
        if len(partial_t_dict) > 0:
            # Default to full trajectory if not specified
            if "bb_ca" not in partial_t_dict:
                partial_t_dict["bb_ca"] = 0.0
            if "local_latents" not in partial_t_dict:
                partial_t_dict["local_latents"] = 0.0

            # Corrupt the input batch to the given noise level(s)
            batched_partial_t = {
                dm: torch.full(size=(nsamples,), fill_value=partial_t, device=device)
                for dm, partial_t in partial_t_dict.items()
            }
            batch = model.fm.corrupt_batch(batch, t=batched_partial_t)
            target_backbone = (
                batch["x_1"]["bb_ca"].clone() if inverse_folding_method == RESAMPLED_FORWARD_MARGINAL else None
            )

            simulation_step_params = {
                data_mode: model.inf_cfg.model[data_mode]["simulation_step_params"] for data_mode in model.fm.data_modes
            }

            partial_ts, partial_gt = build_partial_schedule(
                partial_t_dict=partial_t_dict,
                nsteps=actual_nsteps,
                fm=model.fm,
                sampling_model_args=model.inf_cfg.model,
                full_nsteps=nsteps if frac is not None else None,
            )

            # Start from the corrupted input (not pure noise)
            gen_samples, _, extra_info = model.fm.partial_simulation(
                batch=batch,
                x=batch["x_t"],
                predict_for_sampling=model.predict_for_sampling,
                recycle_depths=recycle_depths,
                nsteps=actual_nsteps,
                ts=partial_ts,
                gt=partial_gt,
                simulation_step_params=simulation_step_params,
                device=device,
                save_trajectory_every=save_trajectory_every,
                dump_noisy_trajectories=dump_noisy_trajectories,
                guidance_w=model.inf_cfg.args.get("guidance_w", 1.0),
                resample_target_backbone=inverse_folding_method == RESAMPLED_FORWARD_MARGINAL,
                target_backbone=target_backbone,
            )

        else:
            gen_samples, extra_info = model.fm.full_simulation(
                batch=batch,
                predict_for_sampling=model.predict_for_sampling,
                recycle_depths=recycle_depths,
                nsteps=actual_nsteps,
                nsamples=nsamples,
                n=n,
                sampling_model_args=model.inf_cfg.model,
                device=device,
                save_trajectory_every=save_trajectory_every,
                dump_noisy_trajectories=dump_noisy_trajectories,
                guidance_w=model.inf_cfg.args.get("guidance_w", 1.0),
                guidance_model=model.guidance_model,
            )

        batch["sequence_logits"] = extra_info.get("sequence_logits")

        # The full batch supplies the AE decoder's trained seq/pair features.
        # The decoder derives its masks and emits only generated tokens.
        sample_prots = model.sample_formatting(
            x=gen_samples,
            ret_mode="coors37_n_aatype",
            batch=batch,
        )

        # Use the shared processing method to convert to atom arrays
        insertion_logits = extra_info.get("insertion_logits")  # [nsamples, n, n] or None
        generation_list = model.format_sample_prots_to_atom_arrays(
            sample_prots, batch, batch_idx=0, insertion_logits=insertion_logits
        )

        # Decode and attach trajectory snapshots if any were collected
        trajectory_steps = extra_info.get("trajectory_steps", [])
        if trajectory_steps:
            _attach_trajectories(
                trajectory_steps=trajectory_steps,
                generation_list=generation_list,
                model=model,
                batch=batch,
                nsamples=nsamples,
                bb_ca_only=bb_ca_only_trajectories,
                dump_noisy_trajectories=dump_noisy_trajectories,
            )

        return generation_list


def _attach_trajectories(
    trajectory_steps: list[dict],
    generation_list: list[dict],
    model: Any,
    batch: dict,
    nsamples: int,
    bb_ca_only: bool,
    dump_noisy_trajectories: bool,
) -> None:
    """Decode trajectory snapshots and attach per-step structures to generation results.

    For each trajectory step, decodes the x_1 clean prediction (and optionally x_t noisy state)
    into AtomArrays and stores them on the corresponding generation_list entry.

    When ``bb_ca_only=True``, all steps share a fixed atom count (one CA per residue), so they
    are stacked into an :class:`~atomworks.io.utils.atom_array_plus.AtomArrayPlusStack` under
    ``'trajectory_x1'`` (and ``'trajectory_xt'``).

    When ``bb_ca_only=False``, atom counts may vary across steps (different predicted sequences
    → different sidechain heavy atoms), so a plain :class:`list` of
    :class:`~atomworks.io.utils.atom_array_plus.AtomArrayPlus` is stored instead.  In this case
    saving to XTC is not possible; use ``save_trajectories_to_cif=True`` to persist the
    trajectory as a multi-block CIF.

    Args:
        trajectory_steps: List of snapshot dicts from full_simulation, each with 'step',
            'x_1' (and optionally 'x_t') keys containing CPU-side tensor dicts.
        generation_list: List of generation result dicts to attach trajectory steps to.
        model: Proteina model instance used for decoding.
        batch: Original batch dict.
        nsamples: Number of samples in the batch.
        bb_ca_only: If True, skip the full decoder and store only CA atoms, enabling XTC saving.
            If False, decode full sidechains; result is a list (variable atom counts).
        dump_noisy_trajectories: If True, also decode x_t steps and attach as 'trajectory_xt'.
    """
    keys_to_decode = ["x_1"] + (["x_t"] if dump_noisy_trajectories else [])
    arrays_per_key: dict[str, list[list]] = {key: [[] for _ in range(nsamples)] for key in keys_to_decode}

    for step_snapshot in trajectory_steps:
        for key in keys_to_decode:
            if key not in step_snapshot:
                continue
            x_step = {dm: step_snapshot[key][dm].to(model.device) for dm in step_snapshot[key]}

            if bb_ca_only:
                sample_prots_step = model._format_sample_bb_ca(
                    x=x_step,
                    ret_mode="coors37_n_aatype",
                    padding_mask=batch["padding_mask"],
                )
            else:
                sequence_logits = step_snapshot.get("sequence_logits")
                step_batch = {
                    **batch,
                    "sequence_logits": sequence_logits.to(model.device) if sequence_logits is not None else None,
                }
                sample_prots_step = model.sample_formatting(
                    x=x_step,
                    ret_mode="coors37_n_aatype",
                    batch=step_batch,
                )

            step_gen_list = model.format_sample_prots_to_atom_arrays(
                sample_prots_step, batch, batch_idx=0, bb_ca_only=bb_ca_only
            )
            for i, result in enumerate(step_gen_list):
                arr = result["generated_atom_array"]
                if bb_ca_only:
                    arr = arr[arr.atom_name == "CA"]
                arrays_per_key[key][i].append(arr)

    for i, gen_item in enumerate(generation_list):
        for key, traj_key in [("x_1", "trajectory_x1"), ("x_t", "trajectory_xt")]:
            steps = arrays_per_key.get(key, [[]] * nsamples)[i]
            if not steps:
                continue

            if bb_ca_only:
                # Fixed atom count — normalise time-varying annotations to the final step's values
                for annot in steps[-1].get_annotation_categories():
                    all_values = [step.get_annotation(annot) for step in steps]
                    final_value = all_values[-1]
                    if not all(np.array_equal(val, final_value) for val in all_values):
                        for step_arr in steps:
                            step_arr.set_annotation(annot, final_value)
                gen_item[traj_key] = as_atom_array_plus_stack(steps)
            else:
                # Variable atom counts — store as plain list; saving requires multi-block CIF.
                gen_item[traj_key] = steps
