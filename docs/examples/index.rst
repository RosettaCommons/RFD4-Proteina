Examples
========

Complete, runnable tutorials. Notebooks are committed without outputs: read one here for the steps and
code, and run it on a GPU machine with the repository checked out to see the tables and 3D views.
Each notebook also includes a public-package Colab setup that needs no checkout or credentials.
Choose a GPU (A100 recommended) and Python 3.12 under **Runtime → Change runtime type**.
Colab's ``2026.07`` runtime provides Python 3.12; see the
`available runtime versions <https://research.google.com/colaboratory/runtime-version-faq.html>`_.
Run the installation cell, choose **Runtime → Restart session** when prompted, then rerun
the cell and wait for **Ready** before continuing.

The Colab quickstart generates one design and downloads its outputs. The Python API tutorial
builds an annotated input, and the VHH tutorial samples without a refolding oracle. The enzyme
and PPI tutorials support input preparation and inspection on Colab; their later scoring steps
require a separately configured refolding runtime (and Slurm for the PPI tutorial).

.. nbgallery::

   colab_quickstart
   python_api_tutorial
   enzyme_design_tutorial
   ppi_specificity_tutorial
   antibody_design_tutorial
