Installation
============

Install directly from a clone into a Python virtual environment. The native
CUDA 13 profile supports Linux x86_64 with Python 3.12 and a compatible NVIDIA
driver. Install RFD4-Proteina from this repository.

Prerequisites
-------------

* Linux, ``git``, and ``curl``.
* An NVIDIA GPU and a driver compatible with CUDA 13.
* Enough storage for the Python environment and released checkpoints (about 8 GB).

.. _native-python-installation:

Native Python installation
--------------------------

Clone and run setup:

.. code-block:: bash

   git clone https://github.com/RosettaCommons/RFD4-Proteina.git
   cd RFD4-Proteina
   make setup

This creates ``.venv`` and installs the dependencies declared in ``pyproject.toml``,
including CUDA metrics and the matching TMol wheel. It also downloads the two
released checkpoints (about 8 GB) and configures ``DATA_PATH`` for generation.
Use ``bash setup.sh --skip-weights`` for an environment-only install.

For a minimal local install into an existing Python 3.12 environment, use
``python -m pip install -e .``. To include CUDA metrics and TMol with
standard pip instead of the setup script:

.. code-block:: bash

   python3.12 -m venv .venv
   source .venv/bin/activate
   python -m pip install --extra-index-url https://download.pytorch.org/whl/cu132 \
       -e '.[metrics-cuda13,tmol-cuda13]'
   python -m pip check
   export DATA_PATH="${DATA_PATH:-$PWD/data_path}"
   ./scripts/download_example_data.sh "$DATA_PATH" --skip-of3
   rfd4-generate inputs=data/benchmarks/monomer \
       subset_to_examples=monomer-short num_replicates=1 out_dir=./runs/smoke

The installed commands ``rfd4-generate``, ``rfd4-evaluate``,
``rfd4-train``, and ``rfd4-train-ae`` use their Python environment
directly. Activate that environment first (``source .venv/bin/activate`` for
setup's venv), or use an explicit path such as ``.venv/bin/rfd4-generate``.
The command uses the interpreter where it was installed; it does not choose
a container or automatically activate another environment. Wheels include the
Hydra configs and benchmark inputs. A wheel install reads local environment
files from the working directory; an editable install reads them from the
checkout root.

AtomWorks is installed automatically from the Git revision declared in
``pyproject.toml``. System drivers and external metric tools are
configured separately. CUDA 12 and native ARM installations have not been validated.

Environment files and paths
---------------------------

Use **.env.local** for new configurations. Setup creates it from
``.env.local.example`` only when neither
``.env.local`` nor ``.env`` exists. Existing settings are preserved, and
setup does not edit shell startup files. If the active file and shell both omit
``DATA_PATH``, setup adds the default data directory to that file.
``make setup`` runs ``bash setup.sh``.

Set ``DATA_PATH`` to the directory containing released checkpoints and model data.
Optional mirror, cache, credential, and refolding-oracle settings also belong in
this ignored file. Use shell-compatible ``KEY=value`` assignments; quote paths
containing spaces. ``PROJECT_ROOT`` supplies the checkout root.

Exported shell variables take precedence over environment-file values.
Both installed commands and launchers read ``.env.local`` if present,
otherwise ``.env``; the two files are not merged. If you already use ``.env``,
continue editing that file or rename it to ``.env.local``. Creating a new
``.env.local`` makes the loader ignore ``.env``.

Configure data and checkpoints
------------------------------

``make setup`` downloads the released checkpoint pair automatically, using
``DATA_PATH`` from your shell or environment file, or ``./data_path`` by default.
Existing checkpoint files are skipped, so setup can be rerun after an interrupted
download. A failed download stops setup with an error.

To download manually after a pip or ``--skip-weights`` installation:

.. code-block:: bash

   ./scripts/download_example_data.sh ./data_path --skip-of3

If you changed ``DATA_PATH`` in your environment file, pass that directory to the
script instead. The released checkpoint pair is downloaded over public HTTPS
from ``YOUR_DATASET_STORE`` at
``https://YOUR_DATASET_STORE``. No credentials are required.
Set ``RFPROTEINA_DATA_URL`` for a mirror. Add ``--with-test-dataset`` for the
small training tutorial dataset. The script does not install a CCD mirror.

Setup downloads only the model checkpoints. To add OpenFold3, rerun the download
script without ``--skip-of3`` on a host with Apptainer installed. For refolding,
set ``AF3_IMAGE``, ``AF3_MODEL_DIR``, and
``AF3_JAX_CACHE_DIR`` in your environment file, following the download script's
printed paths. The oracle runs in a separate Apptainer container. Its host launcher
requires Python 3 (standard library only), alongside the native model environment.
Use ``metrics/refolding_oracles=of3`` for OpenFold3.

Run your first design
---------------------

Activate the environment created by setup and run the installed command:

.. code-block:: bash

   source .venv/bin/activate
   rfd4-generate inputs=data/benchmarks/monomer \
       subset_to_examples=monomer-short num_replicates=1 out_dir=./runs/smoke

The command uses the Python interpreter from that environment.
Subprocesses inherit that environment. Run SLURM launchers without a cluster
argument; see :ref:`slurm-installation` for scheduler settings and outputs.

.. _slurm-installation:

SLURM clusters
--------------

On the login node, you can launch from the environment where you installed the
repo with pip. No cluster name is needed. Launchers use the active virtualenv/Conda environment,
falling back to the checkout ``.venv`` or Python with RFD4-Proteina installed.
The same environment must be available on the compute nodes.

No scheduler profile is required. Partition and account are omitted
unless configured, leaving their selection to SLURM. Set scheduler environment
variables or the launcher's resource flags to match your site. For reusable
settings, optionally create a scheduler profile:

.. code-block:: bash

   cp scripts/user_info.example scripts/.user_info_local
   ${EDITOR:-vi} scripts/.user_info_local

Set ``PARTITION``, ``CPU_PARTITION``, ``ACCOUNT`` (and ``SKIP_ACCOUNT=false``
if your site requires one), ``RUNTIME``, ``MEM``, and ``CPUS_PER_TASK`` to your
site's allocation. Set ``GRES`` if the partition requires a GPU type (for example,
``GRES=gpu:a100:1``). Use ``SLURM_EXTRA_OPTIONS`` for a required QoS, reservation,
or constraint. ``ROOT_REMOTE`` must be on shared storage. These choices belong
to your cluster; there is no universal partition or account name.

Submit 32 designs of the short monomer example across four jobs:

.. code-block:: bash

   ./scripts/launch_generate.sh --njobs 4 --gpus-per-job 1 --no-foldseek \
       inputs=data/benchmarks/monomer subset_to_examples=monomer-short \
       num_replicates=32 batch_size=1 metrics_tags='[design]' \
       run_name_=first_campaign out_dir="$PWD/runs/first_campaign"
   squeue -u "$USER"

The launcher selects the Python environment, resolves the configuration, writes the
batch scripts, submits the array, and queues metric aggregation. You do not need
to activate the checkout ``.venv`` when using setup's default environment.
These launchers submit SLURM jobs; for execution without SLURM use the model
entry point or installed ``rfd4-generate`` command. Inspect ``slurm_run_outputs/``
under the run directory if a job fails. Designs and metrics are in
``inference_samples/`` and ``inference_metrics/`` respectively.
``num_replicates`` counts designs per input example across the jobs; it is not
multiplied by ``--njobs``. Use ``skip_existing=true`` with the same output directory
to resume. For a binder campaign, replace ``inputs`` with your annotated CIF;
see :doc:`workflow/inputs` for conditioning.

The command above computes structure metrics. To score refolding/designability,
rerun the data download without ``--skip-of3`` on a host with Apptainer, configure
the printed OpenFold3 paths in ``.env.local``, and add
``metrics/refolding_oracles=of3 metrics_tags='[design,unconditional_refolding]'``.
Use the task-specific refolding tag for binders or motif scaffolds as described in
:doc:`workflow/metrics`. Refolding runs in its own container even when generation
uses the native environment.

Optional development tools
-------------------------------

Developers can add test, notebook, and formatting tools to the environment:

.. code-block:: bash

   bash setup.sh --venv
   source .venv/bin/activate

This installs RFD4-Proteina in editable mode and its development dependencies, scaffolds
custom metric configuration, and registers a notebook kernel. Use the installed
``rfd4-*`` commands from the activated environment.

Documentation development
-------------------------

After installing the optional development tools:

.. code-block:: bash

   uv pip install -r docs/docs_requirements.txt
   make -C docs html

Then open ``docs/_build/html/index.html``.
