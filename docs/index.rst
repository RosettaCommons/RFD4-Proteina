RFD4-Proteina
=============

.. raw:: html

   <div class="hero">
     <div class="hero-copy">
       <p class="hero-lede">A next generation sequence structure codesign model for proteins, RNA,
       DNA and small molecule complexes.</p>
       <p>Capable of satisfying robust conditioning and generating plausible sequences without
       further refinement.</p>
       <p>Developed by the Baker Lab × NVIDIA GenAIR, with the Institute for Protein Design
       at the University of Washington.</p>

       <p class="hero-ack">We gratefully acknowledge the Jen-Hsun and Lori Huang Foundation for
       providing a computational grant used for this study.</p>

       <!-- Both marks ship in two variants and the theme swaps them on `only-light`/`only-dark`.
            IPD publishes no reverse lockup, so the dark one is generated from the official 2025
            colour lockup with the type reversed to white and the chevron untouched. -->
       <div class="hero-credits">
         <img class="only-light" src="_static/logo-nvidia.svg" alt="NVIDIA">
         <img class="only-dark" src="_static/logo-nvidia-dark.svg" alt="NVIDIA">
         <span class="hero-credit-rule" aria-hidden="true"></span>
         <img class="only-light hero-credit-ipd" src="_static/logo-ipd.png"
              alt="Institute for Protein Design, University of Washington">
         <img class="only-dark hero-credit-ipd" src="_static/logo-ipd-dark.png"
              alt="Institute for Protein Design, University of Washington">
         <span class="hero-credit-rule" aria-hidden="true"></span>
         <img class="hero-credit-rosetta"
              src="_static/Full-color_Rosetta-Commons_Logo_NoBackground.png" alt="RosettaCommons">
       </div>
     </div>
     <figure class="hero-figure">
       <img src="_static/hero-protein.png"
            alt="A designed protein shown as a teal ribbon, wrapped around a small molecule ligand
                 drawn in pink at its binding pocket.">
     </figure>
   </div>

   <a class="wf-card-primary" href="quickstart.html">
     <span class="wf-card-title">Quickstart &rarr;</span>
     <span class="wf-card-body">Install, then generate your first designs on a GPU.</span>
   </a>

   <div class="wf-cards">
     <a class="wf-card" href="installation.html">
       <span class="wf-card-title">Installation</span>
       <span class="wf-card-body">Set up the environment, checkpoints, and cluster config.</span>
     </a>
     <a class="wf-card" href="examples/index.html">
       <span class="wf-card-title">Examples</span>
       <span class="wf-card-body">Runnable notebooks that take one design problem end to end.</span>
     </a>
     <a class="wf-card" href="how-to/input_prep/index.html">
       <span class="wf-card-title">Input Preparation</span>
       <span class="wf-card-body">Build and validate an annotated CIF from your design intent.</span>
     </a>
     <a class="wf-card" href="how-to/index.html">
       <span class="wf-card-title">How-to guides</span>
       <span class="wf-card-body">Task-oriented instructions: the config overrides and CLI flags
       for a specific job.</span>
     </a>
     <a class="wf-card" href="concepts/index.html">
       <span class="wf-card-title">Concepts</span>
       <span class="wf-card-body">How the pieces fit together: annotated CIFs, generation modes,
       search, and metric guidance.</span>
     </a>
     <a class="wf-card" href="reference/index.html">
       <span class="wf-card-title">Reference docs</span>
       <span class="wf-card-body">Lookup tables and syntax for configs and CLI options.</span>
     </a>
     <a class="wf-card" href="contributing/index.html">
       <span class="wf-card-title">Contributing</span>
       <span class="wf-card-body">How to propose changes, code style, and review expectations.</span>
     </a>
   </div>

What you can design
--------------------

The released checkpoint supports all of these design tasks.

.. list-table::
   :header-rows: 1

   * - Task
     - Status
   * - Unconditional generation
     - Supported
   * - Motif scaffolding
     - Supported
   * - Enzyme design
     - Supported, with a :doc:`worked example <examples/enzyme_design_tutorial>`
   * - Small molecule binder design
     - Supported
   * - Protein binder design
     - Supported
   * - DNA binder design
     - Supported

State-of-the-art performance across relevant biomolecular tasks
---------------------------------------------------------------

.. raw:: html

   <figure class="fig-card">
     <div class="fig-plot">
       <span class="fig-ylabel">Average per-target success rate, unclustered</span>
       <div class="fig-img">
         <img src="_static/benchmark-success-rate.png"
              alt="Grouped bar chart of average per-target success rate on three tasks. Enzyme
                   design: RFD4-Proteina 0.36, Proteina-Complexa 0.11, RFD3 0.09. Small molecule
                   binders: RFD4-Proteina 0.62, DISCO model 0.27, RFD3 0.24. Protein binder design:
                   RFD4-Proteina 0.15, Proteina-Complexa 0.04, RFD3 0.04. Error bars on every bar.">
         <!-- Positioned at the measured centre of each bar cluster: 23.2%, 52.0%, 80.7% of the
              image width. The task names were separate text boxes on the slide, so they are not
              part of the raster. -->
         <div class="fig-xgroups">
           <span style="left: 23.2%">Enzyme design</span>
           <span style="left: 52.0%">Small molecule binders</span>
           <span style="left: 80.7%">Protein binder design</span>
         </div>
       </div>
     </div>
     <figcaption>Within each task, bars are RFD4-Proteina, the strongest open baseline, and
     RFD3.</figcaption>
   </figure>

.. toctree::
   :maxdepth: 2
   :caption: Start here
   :hidden:

   quickstart
   installation
   examples/index
   how-to/index

.. toctree::
   :maxdepth: 1
   :caption: More
   :hidden:

   concepts/index
   reference/index
   contributing/index
