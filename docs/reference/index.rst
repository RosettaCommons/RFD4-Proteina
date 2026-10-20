Reference
=========

Lookup tables, syntax references, and generated API docs for RFD4-Proteina. Reach for this section
once you already know roughly what you're doing and need to look something up; for task-oriented
walkthroughs, start at :doc:`../concepts/workflow` instead.

.. toctree::
   :maxdepth: 1
   :hidden:

   glossary
   annotations
   checkpoints
   cli-options
   yaml-json
   metrics
   search-parameters
   output-format
   troubleshooting

.. raw:: html

   <div class="wf-cards">
     <a class="wf-card" href="glossary.html">
       <span class="wf-card-title">Glossary</span>
       <span class="wf-card-body">Conditioning and inference terminology used throughout the docs.</span>
     </a>
     <a class="wf-card" href="annotations.html">
       <span class="wf-card-title">Annotations</span>
       <span class="wf-card-body">Every condition annotation the model reads, its CIF column,
       values, and training definition. See the Conditioning concept page for why they behave
       the way they do.</span>
     </a>
     <a class="wf-card" href="checkpoints.html">
       <span class="wf-card-title">Checkpoints</span>
       <span class="wf-card-body">The model-weight checkpoints RFD4-Proteina loads, and how to
       override them.</span>
     </a>
     <a class="wf-card" href="cli-options.html">
       <span class="wf-card-title">CLI options</span>
       <span class="wf-card-body">Every CLI option for <code>rfd4-generate</code>,
       <code>rfd4-evaluate</code>, <code>rfd4-train</code>, and <code>rfd4-train-ae</code>, with
       types, defaults, and examples.</span>
     </a>
     <a class="wf-card" href="yaml-json.html">
       <span class="wf-card-title">YAML/JSON condition spec</span>
       <span class="wf-card-body">Every field and condition annotation accepted in a YAML/JSON
       condition-spec input file.</span>
     </a>
     <a class="wf-card" href="metrics.html">
       <span class="wf-card-title">Metrics</span>
       <span class="wf-card-body">Every metric, oracle, pass criterion, and pre-oracle filter that
       ships, and the tag that turns each one on.</span>
     </a>
     <a class="wf-card" href="search-parameters.html">
       <span class="wf-card-title">Search parameters</span>
       <span class="wf-card-body">Every parameter accepted under the <code>search</code> block for
       reward guided (beam) search, including per-checkpoint fields, reward terms, and threshold
       criteria.</span>
     </a>
     <a class="wf-card" href="output-format.html">
       <span class="wf-card-title">Output format</span>
       <span class="wf-card-body">How generated designs and refolds are addressed, tagged, and
       annotated.</span>
     </a>
     <a class="wf-card" href="troubleshooting.html">
       <span class="wf-card-title">Troubleshooting</span>
       <span class="wf-card-body">An index of every FAQ and troubleshooting table across the
       docs, linking to each one in place.</span>
     </a>
   </div>

See `API reference`_ below for the reusable Python components behind the Hydra and CLI workflows.

API reference
-------------

These pages are generated from the package's Google style docstrings. You
configure RFD4-Proteina mostly through Hydra and its command line entry points,
so what follows documents the reusable Python components sitting behind those
workflows, grouped by pipeline stage rather than alphabetically.

Data and conditioning
~~~~~~~~~~~~~~~~~~~~~~

Parsing structures into tokens, annotating and cropping them, and resolving
conditions onto a batch.

.. toctree::
   :maxdepth: 2

   api/conditions
   api/transforms

Generation, search, and evaluation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Composing inputs, running reward-guided search, scoring trajectories, and
aggregating metrics once samples are written.

.. toctree::
   :maxdepth: 2

   api/search
   api/rewards
   api/metrics
   api/inference_utils
