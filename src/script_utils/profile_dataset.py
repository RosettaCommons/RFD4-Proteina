#!/usr/bin/env python
"""Profile dataset loading timing with line_profiler.

Measures per-example load time for a list of known slow indices and optionally
produces line-level timing breakdowns via line_profiler.

Usage
-----
Run from the ``projects/rfproteina/`` directory:

.. code-block:: bash

    # Default config (plinder_with_missing_atoms)
    python script_utils/profile_dataset.py

    # Specify a dataset config by its path relative to configs/
    python script_utils/profile_dataset.py datasets/train/plinder/plinder_with_missing_atoms

    # Debug with debugpy (connect a DAP client to localhost:2345 before the script proceeds)
    python -m debugpy --listen 2345 --wait-for-client \\
        projects/rfproteina/script_utils/profile_dataset.py \\
        datasets/train/plinder/plinder_with_missing_atoms

Line-profiling
--------------
To enable line-level timing, uncomment the ``lp.add_function(...)`` calls in
``main()`` for the functions you want to profile, then re-run the script.
Output is printed in milliseconds.
"""

import sys
import time
from pathlib import Path
from typing import Any

import line_profiler
import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from rfproteina.utils.env import load_dotenv  # noqa: E402

load_dotenv()

from hydra import compose, initialize_config_dir  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

from rfproteina.utils.datasets import recursively_instantiate_datasets_and_samplers  # noqa: E402
from rfproteina.utils.logging import print_config_tree  # noqa: E402

CONFIGS_DIR = Path(__file__).parent.parent / "configs"

SLOW_INDICES = [117, 425, 265, 473, 266, 316, 4, 307, 309, 308, 310, 49, 376, 201, 64, 486, 490, 382]


def run(dataset: Any, indices: list[int]) -> None:
    for i in indices:
        start_time = time.time()
        _result = dataset[i]
        print(f"  example {i} done in {time.time() - start_time:.3f} seconds")


def main() -> None:
    config_name = sys.argv[1] if len(sys.argv) > 1 else "datasets/train/plinder/plinder_with_missing_atoms"

    with initialize_config_dir(config_dir=str(CONFIGS_DIR.resolve()), version_base=None):
        cfg = compose(config_name=config_name)

    # The dataset configs reference ${datasets.global_design_task_kwargs} which is normally
    # provided by the experiment config. Inject an empty default so interpolation resolves.
    OmegaConf.update(
        cfg,
        "datasets.global_design_task_kwargs",
        {
            "rasa_kwargs": {"p_add_rasa_condition": 0.0},
            "hbond_kwargs": {"p_add_hbond_condition": 0.0},
            "secondary_structure_kwargs": {"p_add_secondary_structure_condition": 0.0},
            "cath_kwargs": {
                "p_add_cath_c_condition": 0.0,
                "p_add_cath_ca_condition": 0.0,
                "p_add_cath_cat_condition": 0.0,
            },
        },
        merge=True,
        force_add=True,
    )
    print_config_tree(cfg, resolve=True)

    result = recursively_instantiate_datasets_and_samplers(cfg.datasets.train)
    dataset = result["dataset"]

    lp = line_profiler.LineProfiler()
    # lp.add_function(compute_graph_laplacian_eigenvectors)
    # lp.add_function(featurize_bonds_as_laplacian_positional_encoding)
    # lp.add_function(dataset.__class__.__getitem__)
    # lp.add_function(MolecularDataset._apply_loader)
    # lp.add_function(MolecularDataset._apply_transform)
    lp(run)(dataset, SLOW_INDICES)
    lp.print_stats(output_unit=1e-3)  # ms


if __name__ == "__main__":
    main()
