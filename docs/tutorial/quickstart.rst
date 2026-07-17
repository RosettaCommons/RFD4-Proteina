Quickstart
==========

This tutorial installs RFD4-Proteína, downloads the released checkpoints, and
generates one protein design. Run each command from the repository root.

Prerequisites
-------------

You need Linux, an NVIDIA GPU, ``git``, and ``curl``. RFD4-Proteína uses Python
3.12; the setup script creates the correct environment with `uv
<https://docs.astral.sh/uv/>`_.

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
Python 3.12 virtual environment, and installs the project dependencies. On a
local machine, choose ``other`` when it asks which compute cluster you use; the
next step configures the local profile explicitly.

Configure model data
--------------------

Create a local data directory and tell RFD4-Proteína where to find it:

.. code-block:: bash

   export CLUSTER=local
   export DATA_PATH="$PWD/data_path"
   mkdir -p "$DATA_PATH"
   printf 'DATA_PATH=%s\n' "$DATA_PATH" > .ipd/.env.local

Download the example data and released model checkpoints:

.. code-block:: bash

   ./scripts/download_example_data.sh "$DATA_PATH"

The two model checkpoints require approximately 8 GB in total and are
downloaded over public HTTPS.

Generate a design
-----------------

Run a single unconditional protein-design sample:

.. code-block:: bash

   python src/rfproteina/generate.py \
       inputs=data/benchmarks/monomer.json \
       num_replicates=1 \
       out_dir=./runs/smoke

The generated CIF and accompanying metadata are written below
``./runs/smoke/``.

Next steps
----------

* Read the :doc:`annotated-CIF guide <../guides/annotated_cif_inputs>` to
  prepare motifs, hotspots, ligands, bonds, and RASA conditions.
* See the :doc:`inference guide <../guides/inference>` for design tasks,
  partial diffusion, evaluation, and parameter reference.
* Try :doc:`reward-guided search <../guides/search>` to branch, score, and
  retain the strongest generation trajectories.
