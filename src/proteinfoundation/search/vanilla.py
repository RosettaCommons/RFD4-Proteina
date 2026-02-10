"""Vanilla generation strategy (no search)."""

from functools import partial
from typing import Any

from proteinfoundation.search.base_strategy import SearchStrategy
import numpy as np


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
        partial_t_dict = {}
        for aa in batch["atom_array_batch"]:
            for user_prefix in ["backbone", "latent"]:
                annot_name = f"{user_prefix}_partial_t"
                if annot_name in aa.get_annotation_categories() and not np.isnan(aa.get_annotation(annot_name)).all():
                    partial_t_annot = aa.get_annotation(annot_name)
                    non_nan_mask = ~np.isnan(partial_t_annot)
                    if not (partial_t_annot[non_nan_mask] == partial_t_annot[non_nan_mask][0]).all():
                        # Once we train on this, it will be easy to provide at inference using the contig/AtomArray format
                        raise ValueError("Multiple partial_t values within an AtomArray are not currently supported.")
                    elif user_prefix not in partial_t_dict:
                        partial_t_dict[user_prefix] = partial_t_annot[non_nan_mask][0]
                    elif partial_t_annot[non_nan_mask][0] != partial_t_dict[user_prefix]:
                        # This is mostly a sanity check since we provide no way of specifying this at inference
                        # There's nothing wrong with having multiple values across a batch if we find a use-case for this
                        raise ValueError("Multiple partial_t values across batch elements are not currently supported.")

        # Call simulation (batch already has atom37 format from pipeline)
        fn_predict_for_sampling = partial(model.predict_for_sampling, n_recycle=model.inf_cfg.get("n_recycle", 0))
        nsteps = model.inf_cfg.args.nsteps
        device = model.device
        if len(partial_t_dict) > 0:
            # Default to no noise if not specified
            if "backbone" not in partial_t_dict:
                partial_t_dict["backbone"] = 0.0  
            if "latent" not in partial_t_dict:
                partial_t_dict["latent"] = 0.0
            
            # Corrupt the input batch to the given noise level(s)
            batched_partial_t = {
                "bb_ca": np.full_like(batch["t"], fill_value=partial_t_dict["backbone"]),
                "local_latents": np.full_like(batch["t"], fill_value=partial_t_dict["latent"]),
            }
            batch = model.fm.corrupt_batch(batch, t=batched_partial_t)

            simulation_step_params = {
                data_mode: model.inf_cfg.model[data_mode]["simulation_step_params"] for data_mode in model.fm.data_modes
            }

            x0 = model.fm.sample_noise(
                n,
                shape=(nsamples * self.beam_width,),
                device=device,
                mask=batch["padding_mask"],
                training=False,
            )
            gen_samples, extra_info = model.fm.partial_simulation(
                batch=batch,
                x=x0,
                x_1_pred=None,
                mask=batch["padding_mask"],
                predict_for_sampling=fn_predict_for_sampling,
                nsteps=nsteps,
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
