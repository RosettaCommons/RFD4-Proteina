"""Best-of-N search strategy."""

from collections import defaultdict
from typing import Any

from proteinfoundation.metrics.metrics_utils import run_metrics_for_result, run_oracle_refolding_and_pass_criteria
from proteinfoundation.search.base_strategy import SearchStrategy
from proteinfoundation.search.vanilla import VanillaStrategy


class BestOfNStrategy(SearchStrategy):
    """Best-of-N search strategy.

    Generates N independent samples per input and returns the one with highest reward.
    Simple but effective baseline for reward-guided generation.

    Args:
        n_samples: Number of samples to generate per input
        keep_all: If True, keep all samples; if False, only keep best per input

    Example:
        >>> strategy = BestOfNStrategy(n_samples=10)
        >>> results = strategy.generate(model, batch, reward_model)
    """

    def __init__(self, n_samples: int = 10, keep_all: bool = False):
        """Initialize Best-of-N strategy.

        Args:
            n_samples: Number of samples to generate and evaluate per input
            keep_all: Whether to keep all samples or only the best per input
        """
        self.n_samples = n_samples
        self.keep_all = keep_all

    def generate(
        self,
        model: Any,  # Proteina instance
        batch: dict,
        reward_model: Any | None = None,  # RewardModel instance
    ) -> list[dict[str, Any]]:
        """Generate N samples per input and return best (or all) per input.

        Args:
            model: Proteina model instance
            batch: Input batch from dataloader
            reward_model: Reward model for scoring samples

        Returns:
            List containing best sample per input (or all samples if keep_all=True)

        Raises:
            ValueError: If reward_model is None
        """
        if reward_model is None:
            raise ValueError("Best-of-N strategy requires a reward model")

        vanilla_strategy = VanillaStrategy()

        # Determine batch size (number of inputs)
        batch_size = batch["coords"].shape[0]

        output_base_dir = getattr(model, "_current_output_base_dir", None)
        output_suffix: str = getattr(model, "_current_output_suffix", "")

        # Group samples by original input index
        samples_by_input = defaultdict(list)

        # Generate N samples for each input
        for n_iter in range(self.n_samples):
            samples = vanilla_strategy.generate(model, batch, reward_model=None)
            # samples is a list of length batch_size, one per input

            # Compute rewards and group by input index
            for idx, sample in enumerate(samples):
                base_example_id = f"{sample.get('example_id', f'sample_{idx}')}_{n_iter}"
                example_id = f"{base_example_id}{output_suffix}"

                # Phase 1: design metrics
                metrics_output = {}
                if model.design_metric_manager is not None:
                    metrics_output = run_metrics_for_result(
                        sample,
                        batch,
                        idx,
                        model.design_metric_manager,
                        output_base_dir=output_base_dir,
                        example_id=example_id,
                        metrics_tags_override=reward_model.reward_metrics_tags,
                    )

                # Phase 2: oracle + refolding + pass_criteria
                oracle_output = run_oracle_refolding_and_pass_criteria(
                    atom_array=sample["generated_atom_array"],
                    example_id=example_id,
                    output_dir=output_base_dir,
                    oracle_mm=model.oracle_metric_manager,
                    refold_mm=model.refolding_metric_manager,
                    pass_mm=model.pass_criteria_metric_manager,
                    metrics_tags=reward_model.reward_metrics_tags,
                    prior_metrics=metrics_output,
                )

                merged = {**metrics_output, **oracle_output}
                reward = reward_model.aggregate_metrics_output(merged)
                sample["example_id"] = base_example_id
                sample["reward"] = reward
                samples_by_input[idx].append(sample)

        # Select best sample for each input
        results = []
        for idx in range(batch_size):
            input_samples = samples_by_input[idx]

            if self.keep_all:
                # Sort by reward and add all samples for this input
                input_samples.sort(key=lambda x: x["reward"], reverse=True)
                results.extend(input_samples)
            else:
                # Keep only the best sample for this input
                best_sample = max(input_samples, key=lambda x: x["reward"])
                results.append(best_sample)

        return results
