Quickstart
==========

There are three ways to run RFD4-Proteina. All of them need Python 3.12 and a
:ref:`supported NVIDIA GPU <supported-hardware>` with a driver compatible with CUDA 12 or CUDA 13.
Choose the package extra and PyTorch wheels for your CUDA version; the native
``make setup`` workflow defaults to CUDA 13. For CUDA 12, use the
:ref:`PyPI instructions below <quickstart-pip>` or the manual install in :doc:`installation`.

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
The example input (``5o45_cropped.cif.zst``) used below is already in the repository, so nothing else
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

Generate a design. Describe it in a YAML condition spec: this one asks for a 50-120 residue binder
to PD-L1, using an input that is already in the repository. Save it as ``binder.yaml`` in the
repository root:

.. code-block:: yaml

   binder:
     input: docs/examples/inputs/5o45_cropped.cif.zst
     contig: "50-120, /0, A17-131"
     conditions: {C_CRDGRP: {1: [{select: "sel('A/*/*/[N,CA,C,O]')"}]},
                  C_SEQ: {true: [{select: "sel('A')"}]}}

and run it:

.. code-block:: bash

   rfd4-generate inputs=binder.yaml num_replicates=2 out_dir=./runs/first '~metrics'

JSON with the same keys works too. This writes two designs under ``runs/first/inference_samples/``.
``'~metrics'`` skips scoring so no refolding environment is needed (the quotes stop zsh from reading
``~metrics`` as a home directory). Add ``compile_lengths=null`` to a first smoke test to skip the
``torch.compile`` warm-up; leave it out for larger runs. To design something else, edit the spec
(:doc:`how-to/input_prep/yaml-json`) or pick a recipe from the :doc:`how-to/cookbook`;
:doc:`how-to/binder-in-10-lines` walks through this example. See :doc:`how-to/run-the-model` for sampling
settings and :doc:`how-to/evaluate-with-metrics` for scoring. For batch jobs, see :ref:`slurm-installation`.
For other CUDA versions, CPU-only installs, Docker and Apptainer, see :doc:`installation`.

For the fastest possible first run add ``inference_experiment=quickstart``: it turns metrics off, skips the
``torch.compile`` warm-up and generates one design, with no other flags needed.

.. _quickstart-pip:

2. Install from PyPI
--------------------

The wheel is small (about 1 MB): it ships the code, the configs and the monomer example input, not the
benchmark inputs. The antibody torsion reference data is downloaded when that metric is first used. Clone the repository
(option 1) to use the benchmark inputs.

Create an environment, install the package for your CUDA major version, and generate. ``uv`` is recommended: the
environment is about 7 GB and ``uv`` installs it in about two minutes, where plain ``pip`` takes about ten.
Older pip releases still trust the compromised eMudhra/emSign CA (October 2026), so the first command
installs pip 26.2.2 from a hash-pinned URL before ``uv`` is downloaded:

.. code-block:: bash

   pip install "pip @ https://files.pythonhosted.org/packages/3f/06/3e4420302363fbb71be3c7f35ce042f54bfca330e3e66b3a2e3c4cca579f/pip-26.2.2-py3-none-any.whl#sha256=fd7f4dfa8190e014013b9f08f3e973f4e4c26ffd179a204895533a7254813a70"
   pip install "uv>=0.13.4"            # or: curl -LsSf https://astral.sh/uv/install.sh | sh
   uv venv --python 3.12 && source .venv/bin/activate

Choose **one** install command to match your driver:

.. code-block:: bash

   # CUDA 12
   uv pip install --torch-backend cu126 "rfd4-proteina[all-cuda12]"

   # CUDA 13
   uv pip install --torch-backend cu132 "rfd4-proteina[all-cuda13]"

Both support generation and metrics. The optional BioIR backend requires CUDA 13.
Installing ``rfd4-proteina`` without an extra gives the base package.

Generate a design:

.. code-block:: bash

   curl -LO https://raw.githubusercontent.com/RosettaCommons/RFD4-Proteina/main/docs/examples/inputs/5o45_cropped.cif.zst
   cat > binder.yaml <<'EOF'
   binder:
     input: 5o45_cropped.cif.zst
     contig: "50-120, /0, A17-131"
     conditions: {C_CRDGRP: {1: [{select: "sel('A/*/*/[N,CA,C,O]')"}]},
                  C_SEQ: {true: [{select: "sel('A')"}]}}
   EOF
   rfd4-generate inference_experiment=quickstart inputs=binder.yaml out_dir=./runs/pip

This designs a 50-120 residue binder to PD-L1. The ``quickstart`` preset skips scoring and the ``torch.compile``
warm-up so a first run finishes quickly; drop it for larger runs. ``inputs`` takes an annotated CIF or a
condition-spec YAML; see :doc:`how-to/input_prep/yaml-json`.
The released weights are fetched from the Hugging Face Hub and cached. The model is also
available from Python:

.. code-block:: python

   import rfd4_proteina as rfd4

   model = rfd4.Proteina.from_pretrained("nvidia/RFD4-Proteina")
   rfd4.generate(model, overrides=["inference_experiment=quickstart", "inputs=binder.yaml", "out_dir=runs/py"])

Search and the refolding metrics also need an AlphaFold 2 or AlphaFold 3 / OpenFold3 runtime,
which a clone sets up for you (option 1).

.. _quickstart-diffusers:

3. Use the Diffusers pipeline
-----------------------------

RFD4-Proteina is also available as a `Hugging Face Diffusers
<https://huggingface.co/docs/diffusers>`_ pipeline. The pipeline takes prepared feature tensors;
the ``rfd4-proteina-prep`` package turns an annotated CIF into those tensors and writes the
generated structure back to a CIF.

Start with an :doc:`annotated CIF <how-to/input_prep/index>`, or first convert a
:doc:`YAML/JSON condition spec <how-to/input_prep/yaml-json>` to an annotated CIF.
``prepare_cif()`` reads the resulting CIF; it does not accept YAML or JSON directly.
The input guide's :ref:`Diffusers conversion steps <diffusers-condition-spec>` show
how to write one CIF per design from a spec. The native ``rfd4-generate`` command
accepts those specs directly.

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

* :doc:`Cookbook <how-to/cookbook>`: copy-paste YAML specs and overrides for each design task.
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
