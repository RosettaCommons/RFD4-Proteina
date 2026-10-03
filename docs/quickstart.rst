Quickstart
==========

Install RFD4-Proteína, get the checkpoints, and generate your first designs.
Run every command from the repository root.

Prerequisites
-------------

You need Linux x86_64, an NVIDIA GPU with a driver compatible with CUDA 13,
``git``, and ``curl``. The setup script installs Python 3.12 with ``uv``.

Install
-------

.. code-block:: bash

   git clone https://github.com/RosettaCommons/RFD4-Proteina.git
   cd RFD4-Proteina
   make setup
   source .venv/bin/activate

Setup creates ``.venv``, installs the dependencies declared in ``pyproject.toml``,
and downloads the two released checkpoints (about 8 GB) from the public bucket.
It configures ``DATA_PATH`` in ``.env.local``, defaulting to ``./data_path``.
Existing data paths and downloaded checkpoints are reused.

See :doc:`installation` for custom paths, pip installation, and optional refolding.

Generate a design
-----------------

On a GPU node, with the environment activated:

.. code-block:: bash

   rfd4-generate inputs=data/benchmarks/monomer \
       subset_to_examples=monomer-short num_replicates=1 out_dir=./runs/smoke

This generates one 100-residue protein under ``runs/smoke/inference_samples/``.
Metrics, when enabled, land in ``runs/smoke/inference_metrics/``.
``num_replicates`` counts designs per example: omitting ``subset_to_examples``
runs all three monomer examples (100, 150, and 250 residues).

For batch jobs, see :ref:`slurm-installation`.

Next steps
----------

That run used the defaults for everything. :doc:`The full workflow <workflow>`
maps out what a run actually does, from the input file to the metrics that come
back, with a diagram you can click through to any stage.

The stages you are most likely to want next:

* :doc:`Preparing inputs <workflow/inputs>` covers motifs, hotspots, ligands,
  bonds, and RASA conditions.
* :doc:`Running the model <workflow/running>` covers design tasks, evaluation,
  and every parameter you can set.
* :doc:`Metric guidance <workflow/metric-guidance>` covers steering the sampling
  trajectory with a differentiable objective.
* :doc:`Reward guided search <workflow/search>` branches, scores, and keeps the
  best trajectories.
