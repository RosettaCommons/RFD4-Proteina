Installation
============

**Recommended: clone the repository and run ``make setup``**, as shown below.
This installs the environment, downloads the configured released model weights, and configures
``DATA_PATH``. See the :doc:`quickstart` to generate your first design.

Prerequisites
-------------

* Linux x86_64, ``git``, and ``curl``.
* Python 3.12 
* An NVIDIA GPU and a driver compatible with CUDA 12 or CUDA 13. See :ref:`supported-hardware`.
* A C++ compiler and the matching CUDA 12 or 13 development toolkit (``nvcc``) only if building TMol from source.
* Enough storage for the Python environment and configured released model weights.

.. _supported-hardware:

Supported hardware
------------------

RFD4-Proteina has been tested and confirmed to run on the following NVIDIA GPUs.

.. list-table::
   :header-rows: 1
   :widths: 16 50 34

   * - Architecture
     - Data center
     - Workstation and desktop
   * - Blackwell
     - B200, B300, GB200, GB300, RTX PRO 6000 Blackwell Server Edition
     - RTX PRO 6000 Blackwell Workstation Edition, GB10 (DGX Spark)
   * - Hopper
     - H100 SXM, H100 PCIe, H100 NVL, H200, H200 NVL, GH200 144GB
     -
   * - Ada Lovelace
     - L40S
     - RTX 6000 Ada Generation
   * - Ampere
     - A100 SXM4 40GB and 80GB, A100 PCIe 40GB and 80GB
     -

.. _native-python-installation:

Native Python installation
--------------------------

``make setup`` installs the CUDA 13 profile. CUDA 12 is also supported; use the
manual CUDA 12 command below or the :ref:`PyPI quickstart <quickstart-pip>`.

Clone and run setup:

.. code-block:: bash

   git clone --depth 1 https://github.com/RosettaCommons/RFD4-Proteina.git
   cd RFD4-Proteina
   make setup

This creates ``.venv`` and installs the dependencies declared in ``pyproject.toml``,
including CUDA metrics and TMol 0.2.0. It downloads the configured released model
weights and configures ``DATA_PATH`` for checkpoints and optional data. It also downloads the two validation
benchmark sets (about 63 MB) from Hugging Face into ``data/benchmarks/``; use ``bash setup.sh --skip-benchmarks`` to skip them.
Use ``bash setup.sh --skip-weights`` for an environment-only install.

Setup then asks whether to add the environments that reward-guided search and the refolding
metrics use: an AlphaFold 2 environment, and an AlphaFold 3 / OpenFold3 environment with an optional
download of the OpenFold3 weights. If you say yes to either, it also asks for your CPU
architecture (``amd64`` or ``arm64``) and CUDA major version (``12`` or ``13``), downloads what is
published, and records the ``AF2_*`` and ``AF3_*`` settings in ``.env.local``. Skip the questions with
``bash setup.sh --skip-search-envs`` and run them later with ``bash scripts/setup_search_envs.sh``.
See the :doc:`quickstart` for what each answer does.

TMol and ligand support are included by default. For a manual install into a
Python 3.12 environment with CUDA 13, use the same CUDA wheel source as setup.
The pip bundled with a new venv still trusts the compromised eMudhra/emSign CA
(October 2026), so the first command after activation replaces it with pip 26.2.2
from a hash-pinned URL before anything else is downloaded:

.. code-block:: bash

   python3.12 -m venv .venv
   source .venv/bin/activate
   python -m pip install "pip @ https://files.pythonhosted.org/packages/3f/06/3e4420302363fbb71be3c7f35ce042f54bfca330e3e66b3a2e3c4cca579f/pip-26.2.2-py3-none-any.whl#sha256=fd7f4dfa8190e014013b9f08f3e973f4e4c26ffd179a204895533a7254813a70"
   python -m pip install "uv>=0.13.4"
   uv pip install --build-constraint <(printf 'torch==2.14.1\n') \
       --torch-backend cu132 -e '.[all-cuda13]'
   python -m pip check
   export DATA_PATH="${DATA_PATH:-$PWD/data_path}"
   ./scripts/download_assets.sh "$DATA_PATH" --skip-of3
   rfd4-generate inputs=data/benchmarks/monomer \
       subset_to_examples=monomer-short num_replicates=1 out_dir=./runs/smoke

For **CUDA 12**, replace the ``uv pip install`` command above with:

.. code-block:: bash

   uv pip install --no-sources --torch-backend cu126 -e '.[all-cuda12]'

Use only one CUDA extra in an environment. Both profiles support generation and
metrics; the optional BioIR backend requires CUDA 13.

Setup reads the Torch requirement from ``pyproject.toml`` and keeps the qualified
Torch 2.14 runtime when installing binary metric packages.

The installed commands ``rfd4-generate``, ``rfd4-evaluate``,
``rfd4-train``, and ``rfd4-train-ae`` use their Python environment
directly. Activate that environment first (``source .venv/bin/activate`` for
setup's venv), or use an explicit path such as ``.venv/bin/rfd4-generate``.
The command uses the interpreter where it was installed; it does not choose
a container or automatically activate another environment. Wheels include the
Hydra configs and monomer example inputs; clone the repository for the benchmark
sets. A wheel install reads local environment files from the working directory;
an editable install reads them from the checkout root.

AtomWorks is installed from the project metadata. The base package runs generation with the
default metrics. There are two extras; pick the one that matches your driver, because it selects the
TMol and cuEquivariance operator wheels (``tmol-ops-cu12`` or ``tmol-ops-cu13``).

.. list-table::
   :header-rows: 1
   :widths: 24 76

   * - Extra
     - Adds
   * - (none)
     - Generation, evaluation with the default metrics, Hugging Face checkpoints. Runs without the fused cuEquivariance
       kernels, so use a CUDA extra for speed.
   * - ``all-cuda13``
     - TMol with ``tmol-ops-cu13``, the fused cuEquivariance and CUTLASS DSL kernels, PoseBusters, the training and
       analysis tools (matplotlib, scikit-learn, boto3) and the BioIR runtime for ``inference_backend=bioir``.
   * - ``all-cuda12``
     - The same with ``tmol-ops-cu12`` and the CUDA 12 cuEquivariance operators, without BioIR (CUDA 13 only).

For example ``uv pip install "rfd4-proteina[all-cuda13]"``. System drivers and external metric tools are
configured separately.

TMol devices
~~~~~~~~~~~~

TMol metrics default to ``device="cuda"``. Install ``tmol[cu12]>=0.2.0`` or
``tmol[cu13]>=0.2.0`` for the selected CUDA major; the nonexistent package names
``tmol-ops-cuda12`` and ``tmol-ops-cuda13`` are not valid. Missing CUDA operators
raise an actionable error and never route automatically to CPU. Use
``device="cpu"`` to request CPU scoring explicitly. TMol's build-time headers do
not change RFProteina's qualified Torch 2.14 runtime.

Torch 2.14.1 resolves for cu126 (the CUDA 12 profile; PyTorch publishes no cu128/cu129
wheels for 2.14) and for both cu130 and cu132 on CUDA 13. ``make setup`` and the Apptainer
recipes use cu132; the Docker image uses cu130. A successful install does not by itself
qualify RFProteina generation, reward, or relaxation.

Deployment options
~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 22 30 48

   * - Target
     - Command
     - Notes
   * - Workstation or cluster node, CUDA 13 driver
     - ``make setup``
     - Default. Torch 2.14.1/cu132, ``all-cuda13`` extra. Creates ``.venv`` and caches the weights.
   * - CUDA 12 driver
     - ``uv pip install --no-sources --torch-backend cu126 -e '.[all-cuda12]'``
     - Torch 2.14.1/cu126 (PyTorch publishes no cu128/cu129 wheels for 2.14).
   * - CPU only
     - ``uv pip install --no-sources --torch-backend cpu -e .``
     - Preprocessing and explicit CPU metrics; no CUDA operators installed.
   * - Docker
     - ``docker build --build-arg RFD4_ACCELERATOR=cu13|cu12|cpu .``
     - Application image, see below. Needs only the NVIDIA driver and container toolkit on the host.
   * - Apptainer / Singularity
     - ``apptainer build rfd4.sif docker-daemon://rfd4-proteina:cu13``
     - Convert the Docker image; pass ``--nv`` at run time.
   * - PyPI
     - ``uv pip install rfd4-proteina``
     - Add the ``all-cuda12`` or ``all-cuda13`` extra to match the driver. ``uv`` is about five times faster than ``pip`` for this environment; ``pip install`` also works.

The CUDA profiles accept cuEquivariance 0.13.x and TMol 0.2.x with the CUDA-major operator package; the container images pin the exact versions.
Linux x86_64 is the qualified platform; the pinned wheels also exist for aarch64 but are not
qualified. The optional BioIR backend is a separate runtime that these installs do not include yet.

Docker
~~~~~~

The root ``Dockerfile`` builds an application image containing RFD4-Proteina and
its command-line tools. CUDA 13 is the default; CUDA 12 and CPU-only images must
be selected explicitly:

.. code-block:: bash

   docker build -t rfd4-proteina:cu13 .
   docker build --build-arg RFD4_ACCELERATOR=cu12 -t rfd4-proteina:cu12 .
   docker build --build-arg RFD4_ACCELERATOR=cpu -t rfd4-proteina:cpu .

``RFD4_ACCELERATOR`` is the only knob, and it selects the extra and the Torch wheel index:

.. list-table::
   :header-rows: 1
   :widths: 18 30 52

   * - Value
     - Installs
     - Operators
   * - ``cu13`` (default)
     - ``.[all-cuda13]``, Torch cu130
     - ``tmol-ops-cu13``, cuEquivariance cu13
   * - ``cu12``
     - ``.[all-cuda12]``, Torch cu126
     - ``tmol-ops-cu12``, cuEquivariance cu12
   * - ``cpu``
     - no metrics extra, Torch cpu
     - none; explicit CPU metrics only

Everything resolves from public PyPI and the public PyTorch index. The build passes
``--no-sources``, so ``[tool.uv.sources]`` in ``pyproject.toml`` is ignored and no Git or
SSH dependency is reachable: the image needs no submodule and no internal credentials.
TMol and its operator packages are installed from published wheels only, never built from
source. CUDA libraries come from the PyTorch wheels, so the image uses a slim Python base
rather than a CUDA base image. Dependencies are installed before the source is copied, so
rebuilding after a code change reuses the cached dependency layer. The image does not silently
fall back to CPU. The ``cu13`` profile additionally installs ``bionemo-ir`` from PyPI at the version
``pyproject.toml`` pins; ``cu12`` and ``cpu`` stay on the native runtime. See
:doc:`how-to/advanced/backends` for what BioIR does and does not support.

Mount the Hugging Face cache and output directory when running a released checkpoint:

.. code-block:: bash

   docker run --rm --gpus all \
       -v "${HF_HOME:-$HOME/.cache/huggingface}:/opt/rfd4/cache/huggingface" \
       -v "$PWD/runs:/workspace/runs" \
       rfd4-proteina:cu13 \
       rfd4-generate inputs=/opt/rfd4/data/benchmarks/monomer/monomer-short.cif.zst \
       hf_ckpt_path=nvidia/RFD4-Proteina num_replicates=1 out_dir=/workspace/runs/smoke ~metrics compile_lengths=null

For native checkpoints, mount them read-only and pass ``hf_ckpt_path=null``,
``ckpt_path=/checkpoints``, ``ckpt_name=...``, and
``autoencoder_ckpt_path=/checkpoints/...``. The CPU image is intended for
preprocessing and explicit CPU metrics; model generation requires a qualified
NVIDIA runtime.

Environment files and paths
---------------------------

RFD4-Proteina reads machine-specific settings from ``.env.local`` in the repository
root, so you don't have to export them in every shell. ``make setup`` creates it
from ``.env.local.example`` and sets ``DATA_PATH``, the directory holding the
released checkpoints. Git ignores this file, so your paths are never committed.

You edit it later to change ``DATA_PATH`` or to add optional settings, such as
refolding-oracle paths (``AF3_IMAGE``, ``AF3_MODEL_DIR``, ``AF3_JAX_CACHE_DIR``),
``FOLDSEEK_EXEC``, or mirror and cache settings. Use shell-style ``KEY=value``
lines, and quote paths that contain spaces. A variable exported in your shell
overrides the same variable in the file.

.. note::

   Older setups used a file named ``.env``. The loader reads ``.env.local`` if it
   exists and ignores ``.env`` entirely; the two are never merged. If you have a
   ``.env``, keep using it or rename it to ``.env.local``, but don't create both.

Set ``DATA_PATH`` to the directory for optional model data (OpenFold3, AlphaFold 2, tutorial data) and any native checkpoints.
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

``make setup`` caches the released model bundle (denoiser and matching autoencoder) from the Hugging
Face repository and revision named by ``hf_ckpt_path`` and ``hf_ckpt_revision`` in
``configs/inference.yaml`` (``nvidia/RFD4-Proteina`` by default). ``HF_HOME`` controls the cache
location. Optional tutorial data and refolding assets are downloaded into ``DATA_PATH``, defaulting
to ``./data_path``. Refolding images and OpenFold3 weights use a separate pinned Hub revision.
A failed download stops setup with an error. Everything comes from Hugging Face except the
AlphaFold 2 parameters, which are DeepMind's public release.

To download manually after a pip or ``--skip-weights`` installation:

.. code-block:: bash

   ./scripts/download_assets.sh ./data_path --skip-of3

If you changed ``DATA_PATH`` in your environment file, pass that directory to the
script instead. For private Hub downloads, provide ``HF_TOKEN`` or log in with ``hf auth login`` using
an account with access. The same downloader works without credentials once the repository is public.
Internal checkouts keep their existing development setup override; after
activating that environment, use the manual download command above, or select
``bash setup.sh --native`` explicitly.
The default model and revision are defined in ``configs/inference.yaml``.
Override ``hf_ckpt_path`` and ``hf_ckpt_revision`` when running
generation to select another local or Hub bundle.
Add ``--with-test-dataset`` for the small training tutorial dataset.
HF manages resumable downloads and reuses files from the selected revision.
The script does not install a CCD mirror.

Setup downloads only the model weights. To add OpenFold3, rerun the download
script without ``--skip-of3`` on a host with Apptainer installed. For refolding,
set ``AF3_IMAGE``, ``AF3_MODEL_DIR``, and
``AF3_JAX_CACHE_DIR`` in your environment file, following the download script's
printed paths. The oracle runs in a separate Apptainer container. Its host launcher
requires Python 3 (standard library only), alongside the native model environment.
Use the command line argument ``metrics/refolding_oracles=of3`` to use OpenFold3
for refolding your designs.

The downloader selects a CUDA 12 or 13 image and verifies its SHA-256 checksum; pass ``--cuda 12``
or ``--cuda 13`` to choose explicitly. Prebuilt images support ``amd64``. Docker/OCI archives and
build recipes are also available in the
`runtime directory <https://huggingface.co/nvidia/RFD4-Proteina/tree/main/openfold3/runtime>`_.

Run your first design
---------------------

Activate the environment created by setup and run the installed command:

.. code-block:: bash

   source .venv/bin/activate
   rfd4-generate inputs=data/benchmarks/monomer \
       subset_to_examples=monomer-short num_replicates=1 out_dir=./runs/smoke

``rfd4-generate`` runs in whichever Python environment is active, so activate
``.venv`` first (as above) or call ``.venv/bin/rfd4-generate`` directly. To submit
runs to a SLURM cluster instead, see :ref:`slurm-installation`.

.. _slurm-installation:

SLURM clusters
--------------

On the login node, you can launch from the environment where you installed the
repo with pip. Launchers use the active virtualenv/Conda environment,
falling back to the checkout ``.venv`` or Python with RFD4-Proteina installed.
The same environment must be available on the compute nodes.

Scheduler settings
~~~~~~~~~~~~~~~~~~

No scheduler profile is required: partition and account are omitted
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

Submit a run
~~~~~~~~~~~~

To submit 32 designs of the short monomer example across four jobs:

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
under the run directory for information if a job fails. Designs and metrics are in
``inference_samples/`` and ``inference_metrics/`` respectively.
``num_replicates`` counts designs per input example across the jobs; it is not
multiplied by ``--njobs``. Use ``skip_existing=true`` with the same output directory
to resume. For a binder campaign, replace ``inputs`` with your annotated CIF;
see :doc:`how-to/input_prep/index` for conditioning.

The command above computes structure metrics. To score refolding/designability,
rerun the data download without ``--skip-of3`` on a host with Apptainer, configure
the printed OpenFold3 paths in ``.env.local``, and add
``metrics/refolding_oracles=of3 metrics_tags='[design,unconditional_refolding]'``.
Use the task-specific refolding tag for binders or motif scaffolds as described in
:doc:`how-to/evaluate-with-metrics`. Refolding runs in its own container even when generation
uses the native environment.

Optional development tools
-------------------------------

To run the example notebooks or add your own metrics, install the extra
development dependencies:

.. code-block:: bash

   bash setup.sh --venv
   source .venv/bin/activate

This installs RFD4-Proteina in editable mode and its development dependencies, scaffolds
custom metric configuration, and registers a notebook kernel. Use the installed
``rfd4-*`` commands from the activated environment.
