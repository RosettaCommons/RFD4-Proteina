"""Vanilla generation strategy (no search)."""

from functools import partial
from typing import Any

from proteinfoundation.search.base_strategy import SearchStrategy


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

        # Call simulation (batch already has atom37 format from pipeline)
        fn_predict_for_sampling = partial(model.predict_for_sampling, n_recycle=model.inf_cfg.get("n_recycle", 0))
        gen_samples, extra_info = model.fm.full_simulation(
            batch=batch,
            predict_for_sampling=fn_predict_for_sampling,
            nsteps=model.inf_cfg.args.nsteps,
            nsamples=nsamples,
            n=n,
            self_cond=model.inf_cfg.args.self_cond,
            sampling_model_args=model.inf_cfg.model,
            device=model.device,
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
