Quickstart
==========

Install RFD4-Proteína, get the checkpoints, and generate your first designs.
Run every command from the repository root.

Prerequisites
-------------

You need Linux, an NVIDIA GPU, ``git``, and ``curl``. RFD4-Proteína runs on
Python 3.12. The setup script builds that environment for you with `uv
<https://docs.astral.sh/uv/>`_, so you do not need to install Python yourself.

Install and download weights
----------------------------

Install uv, then run:

.. code-block:: bash

   git clone https://github.com/RosettaCommons/RFD4-Proteina.git
   cd RFD4-Proteina
   bash setup.sh
   source .venv/bin/activate
   export DATA_PATH="$PWD/data_path"
   bash scripts/download_example_data.sh "$DATA_PATH" --skip-of3

AtomWorks is installed from a pinned revision of its public release branch.
No submodule, private profile, or credentials are required. The checkpoint pair
occupies about 7.3 GB. Leave CLUSTER unset for a local install.

Generate a design
-----------------

Generate an unconditional protein design:

.. code-block:: bash

   python src/rfproteina/generate.py \
       inputs=data/benchmarks/monomer \
       num_replicates=1 \
       out_dir=./runs/smoke

Generated CIFs and per-design metadata land under
``./runs/smoke/inference_samples/``. If you enable metrics, their CSV files
land separately under ``./runs/smoke/inference_metrics/``.

Watch out for one thing here. ``num_replicates`` counts *per example*, not in
total. ``monomer`` holds three examples, ``monomer-short`` at 100 residues,
``monomer-medium`` at 150, and ``monomer-long`` at 250. So ``num_replicates=1``
gives you **three** designs, one of each length. Point ``inputs`` at a file with
a single example, or pass ``num_replicates=N`` to get ``3 × N`` designs.

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
