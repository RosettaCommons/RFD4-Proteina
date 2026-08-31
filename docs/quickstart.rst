Quickstart
==========

Install RFD4-Proteína, get the checkpoints, and generate your first designs.
Run every command from the repository root.

Prerequisites
-------------

You need Linux, an NVIDIA GPU, ``git``, and ``curl``. RFD4-Proteína runs on
Python 3.12. The setup script builds that environment for you with `uv
<https://docs.astral.sh/uv/>`_, so you do not need to install Python yourself.

Install
-------

Clone the repository, then run the setup script:

.. code-block:: bash

   git clone --recurse-submodules \
       https://github.com/RosettaCommons/RFD4-Proteina.git
   cd RFD4-Proteina
   bash setup.sh
   source .venv/bin/activate

The setup script installs ``uv``, initializes the AtomWorks submodule, creates a
Python 3.12 virtual environment, and installs the project dependencies.

It will ask which cluster you submit jobs to. On a configured cluster, choose
``1) digs``, which is the default. That writes ``CLUSTER=digs`` and
``CONTAINER_TYPE=apptainer`` into your shell profile, and those two settings
wire in the shared data directory, the checkpoints, and the apptainer image.
Anywhere else, choose ``5) other`` and configure the local profile in the next
section.

Configure model data
--------------------

.. admonition:: On a configured compute cluster?
   :class: ipd-cluster

   Skip this section. There is nothing to configure and nothing to download.
   Choosing ``digs`` at the setup prompt makes the entry points read
   ``.ipd/.env.digs``, which already points ``DATA_PATH`` at the shared data
   directory and the released checkpoints. Open a new shell (or ``source`` your
   shell profile) so ``CLUSTER=digs`` is set, then go straight to
   `Generate a design`_.

Create a local data directory and tell RFD4-Proteína where to find it:

.. code-block:: bash

   export CLUSTER=local
   export DATA_PATH="$PWD/data_path"
   mkdir -p "$DATA_PATH"
   printf 'DATA_PATH=%s\n' "$DATA_PATH" > .ipd/.env.local

Download the example data and released model checkpoints:

.. code-block:: bash

   ./scripts/download_example_data.sh "$DATA_PATH"

The two checkpoints come to about 8 GB. On a non-configured cluster this also fetches
**OpenFold3** (~5 GB) — a free, Apache-licensed refolding oracle that needs no gated access —
if ``apptainer`` is installed, so refolding metrics work with no further setup. Skip it with
``--skip-of3`` if you'd rather configure AlphaFold3's licensed weights instead; see
``skills/rfd4-proteina-setup/references/clusters.md``.
Everything downloads over HTTPS from the public bucket ``YOUR_DATASET_STORE``
(``https://YOUR_DATASET_STORE``), so you need no
credentials. To pull from somewhere else, set ``RFPROTEINA_DATA_URL``.

Generate a design
-----------------

.. admonition:: On a configured compute cluster?
   :class: ipd-cluster

   Call the script **directly** instead of prefixing it with ``python``. Its
   shebang re-execs it inside the apptainer image resolved from
   ``.ipd/.env.digs``, so the virtualenv never comes into play.

   .. code-block:: bash

      ./src/rfproteina/generate.py \
          inputs=data/benchmarks/monomer \
          num_replicates=1 \
          out_dir=./runs/smoke

   Run this on an interactive GPU. For batch work, the SLURM launchers such as
   ``scripts/launch_generate.sh`` enter the container for you.

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
