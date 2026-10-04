Quickstart
==========

Install RFD4-Proteína, download the checkpoints, and generate your first design.
Use Linux x86_64, Python 3.12.0–3.12.10, and an NVIDIA GPU with a CUDA 13-compatible driver.

Install
-------

.. code-block:: bash

   pip install rfd4-proteina

The installed command works from your working directory; a repository checkout is not
required. To edit the code, use the :ref:`GitHub source setup <native-python-installation>`
instead. Both installations provide the same Python package and commands.

Download the weights and input
------------------------------

Download the `flow checkpoint <https://rfd4-proteina-public-e04a.cwobject.com/flow_checkpoints/chk_epoch_00000169_step_000000340000-ema.ckpt>`_
and `autoencoder checkpoint <https://rfd4-proteina-public-e04a.cwobject.com/ae_checkpoints/2026_09_04_release_ae_z16_400k-ema.ckpt>`_
(about 8 GB total). Save them as ``weights/flow.ckpt`` and ``weights/ae.ckpt``.
Save the :download:`example input <../data/benchmarks/monomer/monomer-short.cif>` as ``design.cif``.

Generate a design
-----------------

.. code-block:: bash

   rfd4-generate inputs=design.cif \
       ckpt_path=./weights ckpt_name=flow.ckpt \
       autoencoder_ckpt_path=./weights/ae.ckpt \
       num_replicates=1 out_dir=./runs/first \
       compile_lengths=null '~metrics'

This generates one 100-residue protein under ``runs/first/inference_samples/``.
The first run disables compilation and optional metrics. Increase ``num_replicates``
for more designs, or replace ``design.cif`` with your own annotated input.
See :doc:`workflow/running` for compilation and sampling settings, and
:doc:`workflow/metrics` for scoring. For batch jobs, see :ref:`slurm-installation`.

Use a Hugging Face model or local bundle
-----------------------------------------

For models distributed as RFD4 bundles, use one model location instead of the two
checkpoint paths. A bundle contains the flow model, matching autoencoder and
configuration. Set ``MODEL`` to a downloaded bundle directory, or to an actual
Hugging Face model repository ID in the form ``organization/model``:

.. code-block:: bash

   MODEL=./model
   rfd4-generate pretrained_model_name_or_path="$MODEL" inputs=design.cif \
       num_replicates=1 out_dir=./runs/first \
       compile_lengths=null '~metrics'

A Hub repository is downloaded and cached automatically. Add
``pretrained_revision=FULL_COMMIT_HASH`` to pin a Hub revision. Input preparation,
sampling options and output paths are identical for local and Hub models.

For Python model loading:

.. code-block:: python

   import rfd4_proteina as rfd4

   model = rfd4.Proteina.from_pretrained("./model", device="cuda")

The same method accepts a Hub repository ID and an optional ``revision``. Use
``local_files_only=True`` for offline loading. The CLI handles the full design
workflow, including input preparation, sampling and CIF output; ``from_pretrained``
loads the model for Python use. Model hosting on the Hub does not run inference
remotely: generation uses your GPU.

Next steps
----------

:doc:`The full workflow <workflow>`
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
