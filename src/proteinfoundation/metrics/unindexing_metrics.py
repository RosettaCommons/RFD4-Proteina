from beartype.typing import Any

from foundry.metrics.metric import Metric


class UnindexedInsertionMetric(Metric):
    """
    Fetches all floating point values from the prediction metadata
    """

    @property
    def kwargs_to_compute_args(self) -> dict[str, Any]:
        return {
            "prediction_metadata": ("prediction_metadata",),
        }

    def compute(self, prediction_metadata: dict[str, Any]) -> dict[str, float]:
        out_dict = {}

        for key in ["insertion_rmsd", "insertion_mae", "insertion_rmcd", "join_point_rmsd"]:
            if key in prediction_metadata:
                out_dict[key] = float(prediction_metadata[key])

        return out_dict
