"""Vanilla generation strategy (no search)."""

from functools import partial
from typing import Any

from proteinfoundation.search.base_strategy import SearchStrategy
import numpy as np
import torch


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
        reward_model: Any | None = None,  # RewardModel instance (ignored)
    ) -> list[dict[str, Any]]:
        """Generate structures without search.

        Args:
            model: Proteina model instance
            batch: Input batch from dataloader
            reward_model: Ignored for vanilla generation

        Returns:
            List of generated structures
        """
        # Preprocess batch
        batch = model.preprocess_batch_for_generation(batch)

        # Extract clean samples (same as training)
        batch = model.add_clean_samples(batch)
        # x_1: generated tokens have clean values, non-generated have zeros

        # Extract dimensions from batch
        nsamples = batch["coords"].shape[0]
        n = batch["padding_mask"].shape[1]

        # Check for partial_t values in batch
        user_prefix_to_dm = {"backbone": "bb_ca", "latent": "local_latents"}
        partial_t_dict = {}
        for aa in batch["atom_array_batch"]:
            for user_prefix,dm in user_prefix_to_dm.items():
                annot_name = f"partial_t_{user_prefix}"
                if annot_name in aa.get_annotation_categories() and not np.isnan(aa.get_annotation(annot_name)).all():
                    partial_t_annot = aa.get_annotation(annot_name)
                    non_nan_mask = ~np.isnan(partial_t_annot)
                    if not (partial_t_annot[non_nan_mask] == partial_t_annot[non_nan_mask][0]).all():
                        # Once we train on this, it will be easy to provide at inference using the contig/AtomArray format
                        raise ValueError("Multiple partial_t values within an AtomArray are not currently supported.")
                    elif dm not in partial_t_dict:
                        partial_t_dict[dm] = partial_t_annot[non_nan_mask][0]
                    elif partial_t_annot[non_nan_mask][0] != partial_t_dict[dm]:
                        # This is mostly a sanity check since we provide no way of specifying this at inference
                        # There's nothing wrong with having multiple values across a batch if we find a use-case for this
                        raise ValueError("Multiple partial_t values across batch elements are not currently supported.")

        # Call simulation (batch already has atom37 format from pipeline)
        fn_predict_for_sampling = partial(model.predict_for_sampling, n_recycle=model.inf_cfg.get("n_recycle", 0))
        nsteps = model.inf_cfg.args.nsteps
        device = model.device
        if len(partial_t_dict) > 0:
            # Default to no noise if not specified
            if "bb_ca" not in partial_t_dict:
                partial_t_dict["bb_ca"] = 0.0  
            if "local_latents" not in partial_t_dict:
                partial_t_dict["local_latents"] = 0.0
            
            # Corrupt the input batch to the given noise level(s)
            batched_partial_t = {
                dm: torch.full(size=(nsamples,), fill_value=partial_t, device=device) for dm, partial_t in partial_t_dict.items()
            }
            batch = model.fm.corrupt_batch(batch, t=batched_partial_t)

            simulation_step_params = {
                data_mode: model.inf_cfg.model[data_mode]["simulation_step_params"] for data_mode in model.fm.data_modes
            }

            # Build per-modality time schedule that correctly handles any schedule
            # shape (uniform, power, log, ...) and any per-modality partial_t.
            #
            # Strategy: generate a dense reference schedule to locate each partial_t
            # in index space, then use that fraction to expand nsteps so the tail of
            # a full schedule covers exactly [partial_t, 1] with nsteps steps.
            nsteps_dense = max(nsteps * 100, 1000)
            dense_ts, _ = model.fm.sample_schedule(
                nsteps=nsteps_dense, sampling_model_args=model.inf_cfg.model
            )

            partial_ts = {}
            partial_gt = {}
            for data_mode in model.fm.data_modes:
                pt = float(partial_t_dict[data_mode])
                # Clamp to guarantee at least 1 denoising step (handles partial_t=1.0)
                pt = min(pt, 1.0 - 1.0 / nsteps)

                # Find first dense index where t >= pt; this locates pt in the
                # schedule's index space regardless of the schedule's nonlinearity.
                above = dense_ts[data_mode] >= pt - 1e-6
                i_start = above.nonzero(as_tuple=True)[0][0].item() if above.any() else nsteps_dense
                n_above = max(nsteps_dense - i_start, 1)  # dense steps from pt → 1

                # Scale nsteps so the tail of the full schedule has exactly nsteps steps
                nsteps_expanded = round(nsteps * nsteps_dense / n_above)

                full_ts, full_gt = model.fm.sample_schedule(
                    nsteps=nsteps_expanded, sampling_model_args=model.inf_cfg.model
                )
                partial_ts[data_mode] = full_ts[data_mode][-(nsteps + 1):]
                partial_gt[data_mode] = full_gt[data_mode][-nsteps:]

            # Start from the corrupted input (not pure noise)
            gen_samples, _, extra_info = model.fm.partial_simulation(
                batch=batch,
                x=batch["x_t"],
                x_1_pred=None,
                mask=batch["padding_mask"],
                predict_for_sampling=fn_predict_for_sampling,
                nsteps=nsteps,
                ts=partial_ts,
                gt=partial_gt,
                self_cond=model.inf_cfg.args.self_cond,
                simulation_step_params=simulation_step_params,
                device=device,
                guidance_w=model.inf_cfg.args.get("guidance_w", 1.0),
                ag_ratio=model.inf_cfg.args.get("ag_ratio", 0.0),
            )

        else:
            gen_samples, extra_info = model.fm.full_simulation(
                batch=batch,
                predict_for_sampling=fn_predict_for_sampling,
                nsteps=nsteps,
                nsamples=nsamples,
                n=n,
                self_cond=model.inf_cfg.args.self_cond,
                sampling_model_args=model.inf_cfg.model,
                device=device,
                save_trajectory_every=0,
                guidance_w=model.inf_cfg.args.get("guidance_w", 1.0),
                ag_ratio=model.inf_cfg.args.get("ag_ratio", 0.0),
            )

        # Format samples (only coors37_n_aatype mode)
        sample_prots = model.sample_formatting(
            x=gen_samples,
            extra_info=extra_info,
            ret_mode="coors37_n_aatype",
        )

        # Use the shared processing method to convert to atom arrays
        generation_list = model.format_sample_prots_to_atom_arrays(sample_prots, batch, batch_idx=0)

        return generation_list
