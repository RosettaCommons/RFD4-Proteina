Concepts
========

How RFD4-Proteina's pieces fit together: the input format, the ways to run the model, and how
search and guidance steer it. For step-by-step tasks, start at :doc:`../quickstart` instead; for
lookup tables and syntax, see :doc:`../reference/index`.

.. toctree::
   :maxdepth: 1
   :hidden:

   workflow

.. raw:: html

   <div class="wf-cards">
     <a class="wf-card" href="workflow.html">
       <span class="wf-card-title">Workflow</span>
       <span class="wf-card-body">The full pipeline as an interactive diagram, from annotated CIF
       through generation, search, refolding, and metrics.</span>
     </a>
     <a class="wf-card" href="annotated-cif.html">
       <span class="wf-card-title">Annotated CIF</span>
       <span class="wf-card-body">What the format is, why it keeps structure and intent in one
       file, and how conditioning attaches at different granularities.</span>
     </a>
     <a class="wf-card" href="conditioning.html">
       <span class="wf-card-title">Conditioning</span>
       <span class="wf-card-body">Why conditions behave the way they do: active masks, conditions
       that imply other conditions, and why charges and bonds matter.</span>
     </a>
     <a class="wf-card" href="partial-diffusion.html">
       <span class="wf-card-title">Partial diffusion and inverse folding</span>
       <span class="wf-card-body">Start denoising from a structure you already have instead of
       from noise.</span>
     </a>
     <a class="wf-card" href="metric-guidance.html">
       <span class="wf-card-title">Metric guidance</span>
       <span class="wf-card-body">Differentiate a scalar objective and steer the sampling
       trajectory toward it.</span>
     </a>
     <a class="wf-card" href="search.html">
       <span class="wf-card-title">Reward guided search</span>
       <span class="wf-card-body">Branch generation at checkpoints and keep the candidates that
       score best.</span>
     </a>
     <a class="wf-card" href="metrics.html">
       <span class="wf-card-title">Metrics</span>
       <span class="wf-card-body">What each metric family measures, how binder and target
       selections resolve, and where a run can stop early.</span>
     </a>
     <a class="wf-card" href="refolding.html">
       <span class="wf-card-title">Refolding</span>
       <span class="wf-card-body">Why an independent structure prediction model checks each design,
       and how a design can carry more than one refold.</span>
     </a>
   </div>
