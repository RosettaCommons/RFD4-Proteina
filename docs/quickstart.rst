Quickstart
==========

There are three ways to run RFD4-Proteina. All of them need Python 3.12 and a
:ref:`supported NVIDIA GPU <supported-hardware>` with a CUDA 13-compatible driver.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Option
     - Choose it when
   * - :ref:`1. GitHub (git clone and make setup) <quickstart-make-setup>`
     - You want the full workflow: annotated inputs, reward-guided search, refolding
       metrics, examples and training. This is the recommended route.
   * - :ref:`2. PyPI <quickstart-pip>`
     - You want the model inside your own Python environment or application.
   * - :ref:`3. Hugging Face Diffusers <quickstart-diffusers>`
     - You want the model as a Diffusers pipeline.

.. _quickstart-make-setup:

1. Clone from GitHub and run ``make setup``
-------------------------------------------

.. code-block:: bash

   git clone https://github.com/RosettaCommons/RFD4-Proteina.git
   cd RFD4-Proteina
   make setup
   source .venv/bin/activate

Setup creates ``.venv``, caches the released model (flow and autoencoder) from Hugging Face and
configures ``DATA_PATH``. It then asks whether you want the environments that reward-guided
search and the refolding metrics score designs with. Both are optional, and you can run the
questions later with ``bash scripts/setup_search_envs.sh``.
See the dedicated :doc:`installation` guide for CUDA build prerequisites and environment settings.
The example input (``monomer-short.cif.zst``) used below is already in the repository, so nothing else
needs to be downloaded to generate your first design.

#. **Set up an AlphaFold 2 environment for search?** AlphaFold 2 refolds protein-only designs.
   Answer yes to download its runtime image (needs ``apptainer``). A follow-up asks whether to
   download DeepMind's public parameters (CC BY 4.0, about 5.3 GB).
#. **Set up an AlphaFold 3 / OpenFold3 environment for search?** This refolds designs with ligands,
   nucleic acids or modified residues. Answer yes to download the runtime image
   from Hugging Face (about 4 GB, needs ``apptainer``). A follow-up asks whether to download the Apache-licensed
   OpenFold3 weights (about 1.4 GB). AlphaFold 3's own weights are gated by Google DeepMind, so
   decline that download if you will point ``AF3_MODEL_DIR`` at your own copy.
#. **Which CPU architecture, and which CUDA major version?** If you said yes to either
   environment, setup asks for ``amd64`` or ``arm64`` and for CUDA ``12`` or ``13``. It suggests
   your machine's values.

The answers are written to a marked block in ``.env.local`` (``AF2_*`` and ``AF3_*`` variables),
which every entry point reads. Prebuilt images support ``amd64`` with CUDA ``12`` or ``13``;
``arm64`` requires a separately built runtime.

Skip the questions with ``bash setup.sh --skip-search-envs``. Choose an oracle at run time with a
Hydra override such as ``metrics/refolding_oracles=of3`` or ``metrics/refolding_oracles=af2``;
:doc:`how-to/evaluate-with-metrics` and :doc:`how-to/search` describe the oracles and how search uses them.

Generate a design:

.. code-block:: bash

   rfd4-generate inputs=data/benchmarks/monomer/monomer-short.cif.zst \
       num_replicates=1 out_dir=./runs/first \
       ~metrics compile_lengths=null

This generates one 100-residue protein under ``runs/first/inference_samples/``.
``compile_lengths=null`` skips ``torch.compile`` warm-up so this first run finishes quickly; drop it
for larger runs where the warm-up cost is worth paying back. Increase ``num_replicates`` for more
designs, or replace the example path with your own annotated CIF input. See
:doc:`how-to/run-the-model` for sampling settings and :doc:`how-to/evaluate-with-metrics` for
scoring. For batch jobs, see :ref:`slurm-installation`. For other CUDA versions, CPU-only installs,
Docker and Apptainer, see :doc:`installation`.

.. _quickstart-pip:

2. Install from PyPI
--------------------

The wheel is small (about 1 MB): it ships the code, the configs and the monomer example input, not the
benchmark inputs. The antibody torsion reference data is downloaded when that metric is first used. Clone the repository
(option 1) to use the benchmark inputs.

Create an environment, install the package for your CUDA major version, and generate. ``uv`` is recommended: the
environment is about 7 GB and ``uv`` installs it in about two minutes, where plain ``pip`` takes about ten.

.. code-block:: bash

   pip install uv                      # or: curl -LsSf https://astral.sh/uv/install.sh | sh
   uv venv --python 3.12 && source .venv/bin/activate
   uv pip install "rfd4-proteina[all-cuda13]"   # CUDA 12: all-cuda12 (no BioIR); rfd4-proteina alone is the base package

   rfd4-generate hf_ckpt_path=nvidia/RFD4-Proteina \
       inputs=design.yaml out_dir=./runs/pip ~metrics compile_lengths=null

``inputs`` takes an annotated CIF or a condition-spec YAML; see :doc:`how-to/input_prep/yaml-json`.
The released weights are fetched from the Hugging Face Hub and cached. The model is also
available from Python:

.. code-block:: python

   from rfd4_proteina import Proteina

   model = Proteina.from_pretrained("nvidia/RFD4-Proteina")

Search and the refolding metrics also need an AlphaFold 2 or AlphaFold 3 / OpenFold3 runtime,
which a clone sets up for you (option 1).

.. _quickstart-diffusers:

3. Use the Diffusers pipeline
-----------------------------

RFD4-Proteina is also available as a `Hugging Face Diffusers
<https://huggingface.co/docs/diffusers>`_ pipeline. The pipeline takes prepared feature tensors;
the ``rfd4-proteina-prep`` package turns an annotated CIF into those tensors and writes the
generated structure back to a CIF.

.. warning::

   The pipeline needs Diffusers support that is not in a released Diffusers version yet. It lives on
   branch ``add-rfd4-proteina-torch`` of ``huggingface/diffusers-new-model-addition-RFD4-Proteina``
   (pull request 4). ``from diffusers import RFD4ProteinaPipeline`` fails on a released Diffusers until
   that support is merged.

.. code-block:: bash

   pip install diffusers git+https://huggingface.co/nvidia/RFD4-Proteina-prep@c9959efdebd75ac7fe9129fbbf0e291a6e0487d5

.. code-block:: python

   import torch
   from huggingface_hub import hf_hub_download
   from diffusers import RFD4ProteinaPipeline
   from rfd4_proteina_prep import prepare_cif, write_cif

   pipe = RFD4ProteinaPipeline.from_pretrained(
       "nvidia/RFD4-Proteina-Diffusers", dtype=torch.bfloat16
   ).to("cuda:0")

   # An annotated CIF says what to hold fixed and what to design: here a PD-L1 target
   # and a 50-120 residue binder.
   input_cif = hf_hub_download(
       "nvidia/RFD4-Proteina-Diffusers", "examples/protein_binder/input.cif"
   )
   features, atoms = prepare_cif(input_cif, seed=42, device=pipe.device)

   required = ["padding_mask", "is_in_context_conditioning", "supports_local_latent", "token_type",
               "atom37_mask", "coords_nm"]
   output = pipe(
       **{name: features[name] for name in required},
       conditioning_features={name: value for name, value in features.items() if name not in required},
       generator=torch.Generator("cuda:0").manual_seed(42),
   )
   write_cif(output, features, atoms, "design.cif")

``output.coords_nm`` holds the generated atom37 coordinates in nanometers. The model repository
also has inputs and designs for antibodies, nanobodies, enzyme motifs, small-molecule binders,
inverse folding and partial diffusion. For reward-guided search and the refolding metrics, use
option 1.

Next steps
----------

:doc:`The full workflow <concepts/workflow>`
maps out what a run actually does, from the input file to the metrics that come
back, with a diagram you can click through to any stage.

Where to go next:

* :doc:`Example notebooks <examples/index>`: runnable example notebooks
  that create annotated CIFs and run RFD4-Proteina.
* :doc:`Preparing inputs <how-to/input_prep/index>` shows the different ways to 
  provide input to RFD4-Proteina to specify your design tasks.
* :doc:`Running the model <how-to/run-the-model>` covers the generate command, the
  settings worth changing, and what to do when a run misbehaves.
* :doc:`Metric guidance <concepts/metric-guidance>` explains how to steer
  RFD4-Proteina's sampling trajectory toward a differentiable objective.
* :doc:`Reward guided search <concepts/search>` explains how search branches
  trajectories, scores them, and keeps the best ones.
