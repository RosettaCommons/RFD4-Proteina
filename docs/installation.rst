Installation
============

Prerequisites
-------------

RFD4-Proteína needs:

* Linux with an NVIDIA GPU
* An NVIDIA driver new enough for CUDA 13 (580.65.06+ on Linux — check with ``nvidia-smi``)
* Python 3.12
* ``git`` and ``curl``

Hardware requirements
----------------------

No minimum GPU model is enforced. ``batch_size`` sizing depends on your input length and card —
confirm your longest input fits at ``batch_size=1`` before raising it.

Enough local storage for two model checkpoints (about 8 GB), plus ~5 GB more for OpenFold3,
downloaded by default on non-configured clusters (see `Configure data and checkpoints`_).

Local installation with uv
--------------------------

Use the repository setup script.

.. code-block:: bash

   git clone --recurse-submodules \
       https://github.com/RosettaCommons/RFD4-Proteina.git
   cd RFD4-Proteina
   bash setup.sh
   source .venv/bin/activate

If you cloned without submodules, the setup script initializes the vendored
AtomWorks dependency for you. At the cluster prompt, choose ``other`` for a local
workstation. Pick a named cluster only when you actually have access to its
shared storage and container configuration. ``setup.sh`` is idempotent, so
re-running it is safe.

The AtomWorks submodule clones over SSH, so you need a GitHub SSH key registered. Without one,
clone without ``--recurse-submodules`` and rewrite the URL for just this repository, not every
GitHub clone on your machine:

.. code-block:: bash

   git clone https://github.com/RosettaCommons/RFD4-Proteina.git
   cd RFD4-Proteina
   git config url."https://github.com/".insteadOf "git@github.com:"
   bash setup.sh

AtomWorks is currently a private repository, so this still needs HTTPS credentials configured
(e.g. via ``gh auth login``, which sets up a git credential helper) — it only changes the
transport, not who can authenticate.

.. admonition:: On a configured compute cluster?
   :class: ipd-cluster

   Run exactly the commands above and choose ``1) digs`` at the cluster prompt,
   which is the default. That is the entire installation. The shared data
   directory, the released checkpoints, and the apptainer image are already
   wired up in the committed ``.ipd/.env.digs``, which the entry scripts source
   automatically.

   So skip `Configure data and checkpoints`_ and `Container installation`_.
   Nothing to download, nothing to build. Go straight to
   `Run your first design`_.

Configure data and checkpoints
------------------------------

RFD4-Proteína resolves checkpoints and shared data through ``DATA_PATH``. For a
local installation:

.. code-block:: bash

   export CLUSTER=local
   export DATA_PATH="$PWD/data_path"
   mkdir -p "$DATA_PATH"
   printf 'DATA_PATH=%s\n' "$DATA_PATH" > .ipd/.env.local
   ./scripts/download_example_data.sh "$DATA_PATH"

The entry points load the per cluster environment file for you. Exporting
``DATA_PATH`` in your current shell also helps for scripts and interactive
sessions.

The two released checkpoints come from the public bucket
``YOUR_DATASET_STORE`` at ``https://YOUR_DATASET_STORE``,
served anonymously. Add ``--with-test-dataset`` to the download command when
you also need the small training dataset used by the developer tutorial. The
script does not install a CCD mirror. You need no AWS CLI, credentials, or IPD
account. To download from a mirror instead, set ``RFPROTEINA_DATA_URL``.

On a non-configured cluster, the same command also fetches **OpenFold3** — a free, Apache-licensed
refolding oracle that needs no gated access — if ``apptainer`` is installed, so refolding
metrics work with no further setup. configured clusters already have AlphaFold3 pre-configured and
skip this automatically. Pass ``--skip-of3`` if you'd rather configure AlphaFold3's licensed
weights yourself instead; see ``skills/rfd4-proteina-setup/references/clusters.md``.

Container installation
----------------------

Apptainer definitions and cluster image links live under ``.ipd/apptainer/``.
On a configured cluster, run the entry point directly; its shebang loads
``.ipd/.env.<cluster>`` and enters the image with the required binds:

.. code-block:: bash

   ./src/rfproteina/generate.py \
       inputs=data/benchmarks/monomer \
       num_replicates=1 \
       out_dir=./runs/smoke

``LATENT_APPTAINER_SIF`` comes from the per cluster environment file
``.ipd/.env.<cluster>``. Do not invoke ``apptainer exec`` with that variable
until the environment file has been loaded; a new shell does not define it by
itself.

When no image is staged, build one from the supplied definition and record its
absolute path in your local cluster profile:

.. code-block:: bash

   apptainer build "$PWD/latent-dev.sif" .ipd/apptainer/latent-dev.def
   printf 'DATA_PATH=%s\nLATENT_APPTAINER_SIF=%s\n' \
       "$DATA_PATH" "$PWD/latent-dev.sif" > .ipd/.env.local
   export CLUSTER=local

Dockerfiles live under ``.ipd/docker/``. On a configured cluster, running an
entry script directly can re-execute it inside the selected container, while
``python src/rfproteina/generate.py`` uses your active Python environment.

Run your first design
---------------------

Check your installation by generating something on a GPU rather than by running
the test suite. One design exercises the checkpoints, the data paths and the
model together. Grab an interactive GPU session, then follow :doc:`Generate a
design <quickstart>` in the quickstart. It gives the command for both the local
and the IPD case, and explains what lands in your output directory.

Documentation development
-------------------------

Install and build the documentation locally.

.. code-block:: bash

   uv pip install -r docs/docs_requirements.txt
   make -C docs html

Then open ``docs/_build/html/index.html`` in a browser.
