from collections import defaultdict

import numpy as np
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

    def compute(self, prediction_metadata: list[dict[str, Any]]) -> dict[str, float]:
        out_dict = {}

        all_metric_values = defaultdict(list)

        for metadata_dict in prediction_metadata:
            for key in ["insertion_rmsd", "insertion_mae", "insertion_rmcd", "join_point_rmsd"]:
                if key in metadata_dict:
                    all_metric_values[key].append(metadata_dict[key])

        for key, values in all_metric_values.items():
            out_dict[key] = float(np.mean(values))

        return out_dict
