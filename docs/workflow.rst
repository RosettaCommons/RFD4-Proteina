:html_theme.sidebar_secondary.remove:

Workflow
========

.. container:: wf-intro

   RFD4-Proteína has a lot of pieces. To run a basic example, start with the
   :doc:`quickstart`.

   But we also ship the tooling to run complete design campaigns. That includes
   refolding with structure prediction oracles, which empirically correlate with
   wet lab success [#bennett]_, ddG computation with TMol, a GPU accelerated
   Rosetta [#rosetta]_, and many more structural and sequence based metrics. You
   can also :doc:`search <workflow/search>` over those metrics, biasing the
   distribution you sample from toward the properties you want.

   The diagram below lays out the general flow from input to output, and shows
   the ecosystem around the base model. Click any piece to open its
   documentation. Contributions welcome.

.. raw:: html

   <svg viewBox="0 0 900 620" xmlns="http://www.w3.org/2000/svg" class="wf" role="img"
        aria-labelledby="wf-title wf-desc">
     <title id="wf-title">RFD4-Proteína design and guidance workflows</title>
     <desc id="wf-desc">Author an annotated CIF, then choose one of three linked model modes:
       standard generation from noise; partial diffusion or inverse folding from a supplied
       structure; or metric guidance from a differentiable objective. Optional refolding checks
       designs. Search loops candidates back, while metrics compare the input, design, and refold
       and drive named filtering gates.</desc>

     <defs>
       <marker id="wf-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5" markerHeight="5"
               orient="auto-start-reverse">
         <path d="M 0 0 L 10 5 L 0 10 z" class="wf-arrowhead"/>
       </marker>

       <!-- flat line icons, 24x24, stroked in the parent's colour -->
       <symbol id="ico-claude" viewBox="0 0 24 24">
         <path d="M12 3 L13.6 9.4 L20 11 L13.6 12.6 L12 19 L10.4 12.6 L4 11 L10.4 9.4 Z"/>
       </symbol>
       <symbol id="ico-yaml" viewBox="0 0 24 24">
         <rect x="5" y="3" width="14" height="18" rx="2"/>
         <path d="M8.5 8.5 H15.5 M8.5 12 H15.5 M8.5 15.5 H13"/>
       </symbol>
       <symbol id="ico-python" viewBox="0 0 24 24">
         <path d="M9 7 L4 12 L9 17 M15 7 L20 12 L15 17"/>
       </symbol>
       <symbol id="ico-cif" viewBox="0 0 24 24">
         <path d="M6 3 H14 L18 7 V21 H6 Z"/><path d="M14 3 V7 H18"/>
         <circle cx="12" cy="14" r="2.4"/><path d="M12 16.4 V19"/>
       </symbol>
       <!-- a padlock: conditions are what you hold fixed -->
       <symbol id="ico-lock" viewBox="0 0 24 24">
         <rect x="5" y="10.5" width="14" height="10" rx="2"/>
         <path d="M8.2 10.5 V7.6 A3.8 3.8 0 0 1 15.8 7.6 V10.5"/>
       </symbol>
       <!-- A 2-3-2 network. Edges are drawn thinner than the rings so the two layers of the
            drawing stay separable at 26px instead of blurring into one blob. -->
       <symbol id="ico-model" viewBox="0 0 24 24">
         <path class="wf-ico-hair"
               d="M5 7.5 L11 4 M5 7.5 L11 12 M5 16.5 L11 12 M5 16.5 L11 20
                  M13 4 L19 7.5 M13 12 L19 7.5 M13 12 L19 16.5 M13 20 L19 16.5"/>
         <circle cx="4" cy="7.5" r="2"/><circle cx="4" cy="16.5" r="2"/>
         <circle cx="12" cy="4" r="2"/><circle cx="12" cy="12" r="2"/><circle cx="12" cy="20" r="2"/>
         <circle cx="20" cy="7.5" r="2"/><circle cx="20" cy="16.5" r="2"/>
       </symbol>
       <symbol id="ico-refold" viewBox="0 0 24 24">
         <path d="M20 12 A8 8 0 1 1 17.2 6"/><path d="M17.6 2.6 V6.4 H13.8"/>
       </symbol>
       <!-- one trunk fanning into three candidates -->
       <symbol id="ico-branch" viewBox="0 0 24 24">
         <path d="M4 12 H10 M10 12 L16.5 5.5 M10 12 H16.5 M10 12 L16.5 18.5"/>
         <circle cx="18.4" cy="5.5" r="1.7"/><circle cx="18.4" cy="12" r="1.7"/>
         <circle cx="18.4" cy="18.5" r="1.7"/>
       </symbol>
       <symbol id="ico-design" viewBox="0 0 24 24">
         <path d="M6 3 H14 L18 7 V21 H6 Z"/><path d="M14 3 V7 H18"/>
         <path d="M9 12 C 11 9.5, 13 14.5, 15 12"/>
       </symbol>
       <symbol id="ico-table" viewBox="0 0 24 24">
         <rect x="4" y="4" width="16" height="16" rx="2"/>
         <path d="M4 9.5 H20 M9.5 9.5 V20"/>
       </symbol>
       <symbol id="ico-gate" viewBox="0 0 24 24">
         <path d="M4 12 H9 M15 12 H20"/><path d="M9 5.5 V18.5 M15 5.5 V18.5"/>
       </symbol>
     </defs>

     <!-- ================================================== metrics rails, drawn first so the
          cards paint over their ends. Three of them, because metrics are read against three
          different things: your input, the design, and the refold. -->
     <path d="M108 190 V130 H244" class="wf-rail wf-rail-return"/>
     <path d="M200 430 H244" class="wf-rail wf-rail-return"/>
     <path d="M200 522 H244" class="wf-rail wf-rail-return"/>

     <!-- ============================================================ 1. ANNOTATED CIF -->
     <text x="244" y="26" class="wf-eyebrow wf-ink-specify">YOU SPECIFY</text>

     <a href="workflow/inputs.html" class="wf-link">
       <rect x="244" y="36" width="402" height="112" rx="8" class="wf-node wf-node-specify"/>
       <use href="#ico-cif" x="266" y="52" width="22" height="22" class="wf-ico wf-ico-specify"/>
       <text x="300" y="68" class="wf-node-label">ANNOTATED CIF</text>
       <text x="300" y="82" class="wf-node-sub">structure and intent in one file</text>
       <text x="253" y="96" class="wf-inner-eyebrow">THREE WAYS TO MAKE ONE</text>
     </a>

     <!-- Sub-boxes are drawn after their card so they paint on top of it, and as siblings rather
          than children, because an <a> inside an <a> is not something to rely on. A sub-box with no
          page of its own gets `pointer-events: none`, so a click on it falls through to the card
          behind and still opens the card's page. A sub-box that does have a page is a link, marked
          by a magenta border. -->
     <a href="workflow/design2cif.html" class="wf-link">
       <rect x="302" y="101" width="90" height="38" rx="6" class="wf-chip wf-chip-link"/>
       <use href="#ico-claude" x="312" y="112" width="16" height="16" class="wf-ico wf-ico-specify"/>
       <text x="334" y="125" class="wf-chip-label wf-ink-specify">Agent</text>
     </a>

     <a href="workflow/condition-spec.html" class="wf-link">
       <rect x="400" y="101" width="90" height="38" rx="6" class="wf-chip wf-chip-link"/>
       <use href="#ico-yaml" x="410" y="112" width="16" height="16" class="wf-ico wf-ico-specify"/>
       <text x="432" y="125" class="wf-chip-label wf-ink-specify">YAML</text>
     </a>

     <a href="examples/annotated_cif_tutorial.html" class="wf-link">
       <rect x="498" y="101" width="90" height="38" rx="6" class="wf-chip wf-chip-link"/>
       <use href="#ico-python" x="508" y="112" width="16" height="16" class="wf-ico wf-ico-specify"/>
       <text x="530" y="125" class="wf-chip-label wf-ink-specify">Python</text>
     </a>

     <path d="M445 148 L 445 190" class="wf-rail wf-rail-specify" marker-end="url(#wf-arrow)"/>

     <!-- ============================================================ 3. THE MODEL -->
     <g>
       <rect x="244" y="190" width="402" height="168" rx="8" class="wf-node wf-node-model"/>
       <use href="#ico-model" x="264" y="204" width="26" height="26" class="wf-ico"/>
       <text x="300" y="222" class="wf-node-label">RFD4-PROTEÍNA</text>
       <text x="253" y="245" class="wf-inner-eyebrow">THREE WAYS TO RUN IT</text>
     </g>

     <a href="workflow/running.html" class="wf-link wf-mode-standard"
        aria-label="Standard generation workflow">
       <rect x="249" y="251" width="190" height="44" rx="6" class="wf-chip"/>
       <text x="267" y="271" class="wf-chip-label">standard generation</text>
       <text x="267" y="286" class="wf-chip-sub">from noise</text>
     </a>

     <a href="workflow/partial-diffusion.html" class="wf-link wf-mode-partial"
        aria-label="Partial diffusion and inverse folding workflow">
       <rect x="451" y="251" width="190" height="44" rx="6" class="wf-chip"/>
       <text x="469" y="271" class="wf-chip-label">partial diffusion</text>
       <text x="469" y="286" class="wf-chip-sub">or inverse folding</text>
     </a>

     <a href="workflow/metric-guidance.html" class="wf-link wf-mode-metric"
        aria-label="Metric guidance workflow">
       <rect x="249" y="303" width="392" height="44" rx="6" class="wf-chip"/>
       <text x="267" y="323" class="wf-chip-label">metric guidance</text>
       <text x="267" y="338" class="wf-chip-sub">differentiable objective</text>
     </a>

     <!-- gate: pre-oracle filters -->
     <path d="M445 358 L 445 402" class="wf-rail wf-rail-return" marker-end="url(#wf-arrow)"/>
     <rect x="428" y="371" width="34" height="18" rx="4" class="wf-gate-box"/>
     <use href="#ico-gate" x="437" y="373" width="16" height="16" class="wf-ico wf-ico-return"/>
     <text x="474" y="376" class="wf-gate-name">PRE-ORACLE FILTERS</text>
     <text x="474" y="388" class="wf-note">which designs are worth refolding</text>

     <!-- ============================================================ REFOLD (a stage, not a page) -->
     <rect x="244" y="402" width="402" height="56" rx="8" class="wf-node wf-node-optional"/>
     <use href="#ico-refold" x="266" y="419" width="20" height="20" class="wf-ico"/>
     <text x="300" y="427" class="wf-node-label">REFOLD</text>
     <text x="364" y="426" class="wf-inner-eyebrow">OPTIONAL</text>
     <text x="300" y="444" class="wf-node-sub">supports AlphaFold2, AlphaFold3 and OpenFold3</text>

     <!-- ============================================================ 4. SEARCH -->
     <path d="M646 430 H676" class="wf-rail wf-rail-loop" marker-end="url(#wf-arrow)"/>
     <path d="M676 248 H650" class="wf-rail wf-rail-loop" marker-end="url(#wf-arrow)"/>
     <a href="workflow/search.html" class="wf-link">
       <rect x="676" y="190" width="208" height="268" rx="8" class="wf-node wf-node-loop"/>
       <use href="#ico-branch" x="694" y="206" width="18" height="18" class="wf-ico wf-ico-specify"/>
       <text x="718" y="220" class="wf-node-label">SEARCH</text>
       <text x="694" y="252" class="wf-node-sub">branch at checkpoints,</text>
       <text x="694" y="266" class="wf-node-sub">keep what scores best,</text>
       <text x="694" y="280" class="wf-node-sub">feed the survivors back</text>
       <rect x="694" y="304" width="34" height="18" rx="4" class="wf-gate-box wf-gate-box-loop"/>
       <use href="#ico-gate" x="703" y="306" width="16" height="16" class="wf-ico wf-ico-specify"/>
       <text x="740" y="317" class="wf-gate-name wf-gate-name-loop">SEARCH PRUNING</text>
       <text x="694" y="346" class="wf-note">prunes on any metric,</text>
       <text x="694" y="359" class="wf-note">a refold included</text>
     </a>

     <!-- gate: pass criteria -->
     <path d="M445 458 L 445 502" class="wf-rail wf-rail-return" marker-end="url(#wf-arrow)"/>
     <rect x="428" y="471" width="34" height="18" rx="4" class="wf-gate-box"/>
     <use href="#ico-gate" x="437" y="473" width="16" height="16" class="wf-ico wf-ico-return"/>
     <text x="474" y="476" class="wf-gate-name">PASS CRITERIA</text>
     <text x="474" y="488" class="wf-note">which of those counts as a hit</text>

     <!-- ============================================================ 5. METRICS -->
     <!-- Named metrics rather than the four family names: a reader recognises "scRMSD" and can
          tell at a glance whether their thing is in here. The families are on the page itself. -->
     <a href="workflow/metrics.html" class="wf-link">
       <rect x="16" y="190" width="184" height="352" rx="8" class="wf-node wf-node-return"/>
       <use href="#ico-table" x="34" y="204" width="18" height="18" class="wf-ico wf-ico-return"/>
       <text x="58" y="218" class="wf-node-label">METRICS</text>
     </a>

     <g class="wf-subchip">
       <rect x="34" y="240" width="148" height="24" rx="5" class="wf-chip"/>
       <text x="46" y="256" class="wf-chip-label">input motif RMSD</text>

       <rect x="34" y="268" width="148" height="24" rx="5" class="wf-chip"/>
       <text x="46" y="284" class="wf-chip-label">scRMSD</text>

       <rect x="34" y="296" width="148" height="24" rx="5" class="wf-chip"/>
       <text x="46" y="312" class="wf-chip-label">TMol ddG</text>

       <rect x="34" y="324" width="148" height="24" rx="5" class="wf-chip"/>
       <text x="46" y="340" class="wf-chip-label">H-bond counts</text>

       <rect x="34" y="352" width="148" height="24" rx="5" class="wf-chip"/>
       <text x="46" y="368" class="wf-chip-label">RASA adherence</text>

       <rect x="34" y="380" width="148" height="24" rx="5" class="wf-chip"/>
       <text x="46" y="396" class="wf-chip-label">PoseBusters</text>

       <rect x="34" y="408" width="148" height="24" rx="5" class="wf-chip"/>
       <text x="46" y="424" class="wf-chip-label">…</text>
     </g>

     <a href="workflow/custom-metrics.html" class="wf-link">
       <rect x="34" y="436" width="148" height="24" rx="5" class="wf-chip wf-chip-link"/>
       <text x="46" y="452" class="wf-chip-label wf-ink-specify">custom metrics</text>
     </a>

     <!-- ============================================================ 6. OUTPUTS -->
     <text x="244" y="492" class="wf-eyebrow wf-ink-return">YOU GET BACK</text>
     <a href="workflow/outputs.html" class="wf-link">
       <rect x="244" y="502" width="402" height="88" rx="8" class="wf-node wf-node-return"/>
       <use href="#ico-design" x="266" y="518" width="20" height="20" class="wf-ico wf-ico-return"/>
       <text x="298" y="533" class="wf-node-label">OUTPUTS</text>

       <rect x="259" y="548" width="180" height="30" rx="6" class="wf-chip"/>
       <use href="#ico-design" x="269" y="555" width="16" height="16" class="wf-ico wf-ico-return"/>
       <text x="291" y="568" class="wf-chip-label">annotated designs</text>

       <rect x="451" y="548" width="180" height="30" rx="6" class="wf-chip"/>
       <use href="#ico-table" x="461" y="555" width="16" height="16" class="wf-ico wf-ico-return"/>
       <text x="483" y="568" class="wf-chip-label">metrics table</text>
     </a>
   </svg>

.. toctree::
   :maxdepth: 2
   :titlesonly:
   :caption: Describing the problem

   workflow/inputs
   workflow/design2cif

.. toctree::
   :maxdepth: 2
   :titlesonly:
   :caption: Running the model

   workflow/running
   workflow/backends
   workflow/partial-diffusion
   workflow/metric-guidance
   workflow/search

.. toctree::
   :maxdepth: 2
   :titlesonly:
   :caption: Judging the results

   workflow/metrics
   workflow/outputs

.. rubric:: References

.. [#bennett] Bennett, N.R., Coventry, B., Goreshnik, I., Huang, B., Allen, A.,
   et al. *Improving de novo protein binder design with deep learning.* Nature
   Communications **14**, 2625 (2023).
   https://doi.org/10.1038/s41467-023-38328-5

.. [#rosetta] Alford, R.F., Leaver-Fay, A., Jeliazkov, J.R., O'Meara, M.J.,
   DiMaio, F.P., et al. *The Rosetta all-atom energy function for macromolecular
   modeling and design.* Journal of Chemical Theory and Computation **13** (6),
   3031 (2017). https://doi.org/10.1021/acs.jctc.7b00125
