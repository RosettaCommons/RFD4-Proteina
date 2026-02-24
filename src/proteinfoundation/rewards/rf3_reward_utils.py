# RF3 reward utilities for structure prediction evaluation

import os

from foundry.utils.ddp import RankedLogger

logger = RankedLogger(__name__, rank_zero_only=True)

# Supported RF3 models - kept for compatibility
SUPPORTED_RF3_MODELS = [
    "rf3_latest",
    "rf3_default",
]


def parse_rf3_score_file(score_file_path: str) -> dict[str, float]:
    """Parse RF3 .score file to extract confidence metrics.

    The RF3 score file is in CSV format with multiple samples per batch.
    This function finds the best overall_pae from the last 5 rows and returns
    the corresponding metrics in a format compatible with downstream functions.

    Args:
        score_file_path: Path to the .score file (CSV format)

    Returns:
        Dictionary containing parsed metrics compatible with Protenix format
    """
    import pandas as pd

    if not os.path.exists(score_file_path):
        logger.warning(f"Score file not found: {score_file_path}")
        return {}

    try:
        # Read CSV file
        df = pd.read_csv(score_file_path)

        if df.empty:
            logger.warning(f"Score file is empty: {score_file_path}")
            return {}

        # Find the row with the best (lowest) min_ipae
        if "pae_interface" not in df.columns:
            logger.warning(f"'pae_interface' column not found in score file: {score_file_path}")
            return {}
        # Find the index of the minimum overall_pae in the last 5 rows
        best_idx = df["min_pae_interface"].idxmin()
        best_row = df.loc[best_idx]
        # Extract metrics from the best row in format compatible with downstream functions
        common_metrics = {
            "pLDDT": float(best_row.get("overall_plddt", 0.0)),
            "ipAE": float(best_row.get("pae_interface", 0)),
            "min_ipAE": float(best_row.get("min_pae_interface", 0)),
            "pAE": float(best_row.get("overall_pae", 0)),
            "batch_idx": int(best_row.get("batch_idx", 0)),  # Keep track of which sample was best
        }

        logger.info(
            f"Selected best sample from batch_idx {common_metrics['batch_idx']} with min_ipAE={common_metrics['min_ipAE']:.3f}, ipAE={common_metrics['ipAE']:.3f}, pLDDT={common_metrics['pLDDT']:.3f}"
        )

        return common_metrics

    except Exception as e:
        logger.error(f"Error parsing RF3 score file {score_file_path}: {e}")
        return {}
