Installation
============

Prerequisites
-------------

RFD4-Proteína requires:

* Linux with an NVIDIA GPU
* Python 3.12
* ``git`` and ``curl``
* Enough local storage for two model checkpoints (approximately 8 GB)

Local installation with uv
--------------------------

The repository setup script provides the recommended local installation:

.. code-block:: bash

   git clone --recurse-submodules \
       https://github.com/RosettaCommons/RFD4-Proteina.git
   cd RFD4-Proteina
   bash setup.sh
   source .venv/bin/activate

If the repository was cloned without submodules, the setup script initializes
the vendored AtomWorks dependency automatically. Choose ``other`` at the
cluster prompt for a local workstation; select a named cluster only when its
shared storage and container configuration are available to you.

IPD / DIGS cluster
------------------

On DIGS, setup is the single command above — choose ``1) digs`` at the cluster
prompt. Nothing else is required: the shared data directory and the apptainer
SIF are already wired in the committed ``.ipd/.env.digs``, which the entry
scripts source automatically, so you can skip the "Configure data and
checkpoints" step below and run inference directly. ``setup.sh`` is idempotent
and safe to re-run.

.. code-block:: bash

   git clone --recurse-submodules \
       https://github.com/RosettaCommons/RFD4-Proteina.git
   cd RFD4-Proteina
   bash setup.sh          # choose 1) digs at the cluster prompt
   source .venv/bin/activate

Configure data and checkpoints
------------------------------

RFD4-Proteína resolves checkpoints and shared data through ``DATA_PATH``.
For a local installation:

.. code-block:: bash

   export CLUSTER=local
   export DATA_PATH="$PWD/data_path"
   mkdir -p "$DATA_PATH"
   printf 'DATA_PATH=%s\n' "$DATA_PATH" > .ipd/.env.local
   ./scripts/download_example_data.sh "$DATA_PATH"

The per-cluster environment file is loaded by the entry points. Exporting
``DATA_PATH`` in the current shell is also useful for scripts and interactive
sessions.

All publicly released assets — both checkpoints, the tutorial ``test_dataset``,
and the CCD mirror — are served anonymously from the public bucket
``YOUR_DATASET_STORE`` at ``https://YOUR_DATASET_STORE``.
No AWS CLI, credentials, or IPD account are required; set ``RFPROTEINA_DATA_URL``
to point the download script at a mirror.

Container installation
----------------------

Apptainer definitions and local image links live under ``.ipd/apptainer/``.
Run inference in an existing image with:

.. code-block:: bash

   apptainer exec --nv --bind "$PWD" .ipd/apptainer/latent-dev.sif \
       python src/rfproteina/generate.py \
       inputs=data/benchmarks/monomer.json \
       num_replicates=1 \
       out_dir=./runs/smoke

When no image is staged, build one from the supplied definition:

.. code-block:: bash

   apptainer build latent-dev.sif .ipd/apptainer/latent-dev.def

Dockerfiles are available under ``.ipd/docker/``. On configured clusters,
running an entry script directly can re-execute it inside the selected
container, while ``python src/rfproteina/generate.py`` uses the active Python
environment.

Verify the installation
-----------------------

Generate the smoke-test design from the :doc:`quickstart
<tutorial/quickstart>`, then run the CPU test suite:

.. code-block:: bash

   pytest -m "not gpu"

Documentation development
-------------------------

Install and build the documentation locally:

.. code-block:: bash

   uv pip install -r docs/docs_requirements.txt
   make -C docs html

Open ``docs/_build/html/index.html`` in a browser.
