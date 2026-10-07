Quickstart
==========

Run ``make setup`` to install RFD4-Proteína and download its checkpoints, then generate your first design.
Use Linux x86_64, Python 3.12, and a :ref:`supported NVIDIA GPU <supported-hardware>` with a
CUDA 13-compatible driver.

Install
-------

.. code-block:: bash

   git clone https://github.com/RosettaCommons/RFD4-Proteina.git
   cd RFD4-Proteina
   make setup
   source .venv/bin/activate

Setup creates ``.venv``, downloads the released flow and autoencoder checkpoints
(about 8 GB), and configures ``DATA_PATH``. See :doc:`installation` for CUDA build
prerequisites and environment settings. The checkout includes the example input.

Generate a design
-----------------

.. code-block:: bash

   rfd4-generate inputs=data/benchmarks/monomer/monomer-short.cif \
       num_replicates=1 out_dir=./runs/first \
       compile_lengths=null

This generates one 100-residue protein under ``runs/first/inference_samples/``.
Increase ``num_replicates`` for more designs, or replace the example path with
your own annotated input.
See :doc:`workflow/running` for compilation and sampling settings, and
:doc:`workflow/metrics` for scoring. For batch jobs, see :ref:`slurm-installation`.

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
