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

Install uv, then run:

.. code-block:: bash

   git clone https://github.com/RosettaCommons/RFD4-Proteina.git
   cd RFD4-Proteina
   bash setup.sh
   source .venv/bin/activate

Setup installs Python 3.12 and the public AtomWorks Git revision pinned in
pyproject.toml from release/atomworks-3-0. No submodule, private repository
credentials, Rosetta, or cluster profile is needed. Setup does not edit shell
startup files.

Configure data and checkpoints
------------------------------

.. code-block:: bash

   export DATA_PATH="$PWD/data_path"
   bash scripts/download_example_data.sh "$DATA_PATH" --skip-of3

The default release uses the 340k EMA flow checkpoint and z16 autoencoder.
Downloads are public HTTPS. Keep DATA_PATH exported or record it in the
optional project-root .env file. Leave CLUSTER unset for local use;
an explicitly selected cluster requires a matching user-created runtime profile.

Add --with-test-dataset to fetch the developer tutorial dataset. Omit
--skip-of3 to also download the optional OpenFold3 refolding runtime when
Apptainer is installed. Plain generation does not require a refolding oracle.

Container installation
----------------------

Install the same Python environment in your GPU-enabled container. Mount the
checkout, checkpoints, and outputs, export DATA_PATH, and call
python src/rfproteina/generate.py inside the container. Private cluster
images and profiles are not distributed with this repository.

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
