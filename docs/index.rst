RFD4-Proteína
=============

.. raw:: html

   <div class="hero">
     <div class="hero-copy">
       <p class="hero-lede">A next generation sequence structure codesign model for proteins, RNA,
       DNA and small molecule complexes.</p>
       <p>Capable of satisfying robust conditioning and generating plausible sequences without
       further refinement.</p>
       <p>RFD4-Proteína comes out of a year long collaboration between NVIDIA GenAI Research and
       the Institute for Protein Design at the University of Washington.</p>

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
       </div>

       <p class="hero-actions">
         <a class="hero-btn" href="quickstart.html">Get started</a>
       </p>
     </div>
     <figure class="hero-figure">
       <img src="_static/hero-protein.png"
            alt="A designed protein shown as a teal ribbon, wrapped around a small molecule ligand
                 drawn in pink at its binding pocket.">
     </figure>
   </div>

   <div class="wf-cards">
     <a class="wf-card" href="quickstart.html">
       <span class="wf-card-title">Quickstart</span>
       <span class="wf-card-body">Install, then generate your first designs on a GPU.</span>
     </a>
     <a class="wf-card" href="examples.html">
       <span class="wf-card-title">Examples</span>
       <span class="wf-card-body">Runnable notebooks that take one design problem end to end.</span>
     </a>
     <a class="wf-card" href="workflow.html">
       <span class="wf-card-title">Workflow</span>
       <span class="wf-card-body">The map of a run, from input to output. Click any piece.</span>
     </a>
   </div>

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
     <figcaption>Within each task, bars are RFD4-Proteína, the strongest open baseline, and
     RFD3.</figcaption>
   </figure>

.. toctree::
   :maxdepth: 2
   :caption: Start here
   :hidden:

   quickstart
   workflow
   examples

.. toctree::
   :maxdepth: 1
   :caption: More
   :hidden:

   installation
   development
   glossary
   api_reference
   contributor_guide
