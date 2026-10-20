:orphan:

:html_theme.sidebar_secondary.remove:

Architecture
============

.. container:: wf-intro

   These diagrams take RFD4-Proteina apart one level at a time, from the whole generation loop down
   to a single attention head. The top level is a map of the system. Every level below it is a
   `neural circuit diagram <https://vtabbott.io/articles/diagrams-explained-quickly/>`__ generated
   from an algebraic expression of the code, drawn at the dimensions of the reference model (pair
   width 384, triangle hidden width 256).

   Hover a box or an operation to see its formula, a description and links to the lines of code it
   stands for. Click a box whose details offer "Open diagram" to open its own diagram. The
   breadcrumbs, the tree and the browser's back button take you back up.

   :doc:`The latent explorer <expanded-chemistry>` takes the autoencoder's latents apart.

.. raw:: html

   <div class="arch" id="arch-explorer">
     <div class="arch-toolbar">
       <nav class="arch-crumbs" aria-label="Diagram level"></nav>
       <p class="arch-live" aria-live="polite"></p>
       <div class="arch-tools">
         <div class="arch-zoom" role="group" aria-label="Zoom">
           <button type="button" data-zoom="out" aria-label="Zoom out">−</button>
           <button type="button" data-zoom="fit" aria-label="Fit the diagram to the frame">Fit</button>
           <button type="button" data-zoom="in" aria-label="Zoom in">+</button>
         </div>
         <a class="arch-open" target="_blank" rel="noopener" hidden>Open ↗</a>
         <button type="button" class="arch-expand">Expand</button>
       </div>
     </div>
     <div class="arch-body">
       <nav class="arch-tree" aria-label="All diagrams"></nav>
       <div class="arch-stage" tabindex="-1">
         <div class="arch-map">
           <svg viewBox="0 0 1100 580" xmlns="http://www.w3.org/2000/svg" class="wf arch-map-svg" role="group"
                aria-labelledby="arch-map-title arch-map-desc">
             <title id="arch-map-title">RFD4-Proteina at a glance</title>
             <desc id="arch-map-desc">An annotated CIF is featurized into design features that do not depend on
               the noisy state. A flow-matching sampler runs 60 steps
               from Gaussian noise; at every step the denoiser embeds the noisy state, runs its trunk R times in a
               recycling loop and predicts velocities, and an integration step moves the state. The decoder turns
               the final CA positions and latents into a sequence and atoms. The autoencoder is trained first and
               defines the latents that flow training regresses.</desc>
             <defs>
               <marker id="arch-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5" markerHeight="5"
                       orient="auto-start-reverse">
                 <path d="M 0 0 L 10 5 L 0 10 z" class="wf-arrowhead"/>
               </marker>
               <marker id="arch-arrow-loop" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5" markerHeight="5"
                       orient="auto-start-reverse">
                 <path d="M 0 0 L 10 5 L 0 10 z" class="arch-loop-arrow"/>
               </marker>
               <!-- a circular arrow: something that runs again on its own output -->
               <symbol id="arch-ico-cycle" viewBox="0 0 24 24">
                 <path d="M19.5 12 A7.5 7.5 0 1 1 16.8 6.2"/><path d="M17.4 2.4 V6.6 H13.2"/>
               </symbol>
               <symbol id="arch-ico-cif" viewBox="0 0 24 24">
                 <path d="M6 3 H14 L18 7 V21 H6 Z"/><path d="M14 3 V7 H18"/>
                 <circle cx="12" cy="14" r="2.4"/><path d="M12 16.4 V19"/>
               </symbol>
               <symbol id="arch-ico-model" viewBox="0 0 24 24">
                 <path class="wf-ico-hair"
                       d="M5 7.5 L11 4 M5 7.5 L11 12 M5 16.5 L11 12 M5 16.5 L11 20
                          M13 4 L19 7.5 M13 12 L19 7.5 M13 12 L19 16.5 M13 20 L19 16.5"/>
                 <circle cx="4" cy="7.5" r="2"/><circle cx="4" cy="16.5" r="2"/>
                 <circle cx="12" cy="4" r="2"/><circle cx="12" cy="12" r="2"/><circle cx="12" cy="20" r="2"/>
                 <circle cx="20" cy="7.5" r="2"/><circle cx="20" cy="16.5" r="2"/>
               </symbol>
             </defs>

             <!-- ======================================================== rails, drawn first -->
             <path d="M115 116 V152" class="wf-rail wf-rail-specify" marker-end="url(#arch-arrow)"/>
             <path d="M837 212 H872" class="wf-rail wf-rail-return" marker-end="url(#arch-arrow)"/>
             <path d="M972 262 V296" class="wf-rail wf-rail-return" marker-end="url(#arch-arrow)"/>
             <!-- training: the encoder supplies the latents flow training regresses -->
             <path d="M292 510 H392" class="wf-rail wf-rail-return" marker-end="url(#arch-arrow)"/>
             <text x="300" y="502" class="arch-dim">lat = x₁</text>

             <!-- ======================================================== inputs -->
             <text x="20" y="28" class="wf-eyebrow wf-ink-specify">YOU SPECIFY</text>
             <a href="../workflow/inputs.html" class="wf-link">
               <rect x="20" y="40" width="190" height="76" rx="8" class="wf-node wf-node-specify"/>
               <use href="#arch-ico-cif" x="34" y="54" width="22" height="22" class="wf-ico wf-ico-specify"/>
               <text x="64" y="68" class="wf-node-label">ANNOTATED CIF</text>
               <text x="34" y="98" class="wf-node-sub">structure and intent in one file</text>
             </a>
             <a href="#view=embeddings" data-arch="embeddings" class="wf-link">
               <rect x="20" y="152" width="190" height="144" rx="8" class="wf-node wf-node-specify"/>
               <text x="34" y="178" class="wf-node-label">DESIGN FEATURES</text>
               <text x="34" y="198" class="wf-node-sub">729 token, 77 pair channels,</text>
               <text x="34" y="213" class="wf-node-sub">CATH labels, token types</text>
               <text x="34" y="236" class="wf-node-sub">fixed during a generation,</text>
               <text x="34" y="251" class="wf-node-sub">except 37 coordinate-group</text>
               <text x="34" y="266" class="wf-node-sub">channels, relabelled at</text>
               <text x="34" y="281" class="wf-node-sub">every network evaluation</text>
             </a>

             <!-- ======================================================== the sampler -->
             <text x="240" y="28" class="wf-eyebrow arch-loop-label">FLOW-MATCHING SAMPLER</text>
             <a href="#view=generation" data-arch="generation" class="wf-link">
               <rect x="240" y="40" width="597" height="396" rx="10" class="arch-node-loop"/>
               <use href="#arch-ico-cycle" x="254" y="52" width="22" height="22" class="wf-ico arch-ico-loop"/>
               <text x="284" y="68" class="wf-node-label">T = 60 STEPS</text>
               <text x="420" y="68" class="wf-node-sub">from Gaussian noise at t = 0 to data at t = 1</text>
             </a>
             <!-- the design features enter the denoiser; drawn over the sampler's fill so the arrowhead shows -->
             <path d="M210 212 H258" class="wf-rail wf-rail-specify" marker-end="url(#arch-arrow)"/>
             <!-- the sampling loop, drawn over the sampler's fill: the new state goes back into the denoiser -->
             <path d="M787 262 V412 H500 V392" class="arch-loop-rail" marker-end="url(#arch-arrow-loop)"/>
             <text x="520" y="428" class="arch-loop-sub">x at t + Δt becomes the input of the next step</text>

             <!-- the denoiser -->
             <a href="#view=denoiser" data-arch="denoiser" class="wf-link">
               <rect x="258" y="88" width="470" height="304" rx="8" class="wf-node wf-node-model"/>
               <use href="#arch-ico-model" x="272" y="100" width="24" height="24" class="wf-ico"/>
               <text x="306" y="116" class="wf-node-label">DENOISER</text>
               <text x="394" y="116" class="wf-node-sub">LocalLatentsTransformer · 178 M parameters</text>
             </a>
             <a href="#view=embeddings" data-arch="embeddings" class="wf-link">
               <rect x="272" y="150" width="112" height="52" rx="6" class="wf-chip wf-chip-link"/>
               <text x="284" y="172" class="wf-chip-label wf-ink-specify">Embeddings</text>
               <text x="284" y="189" class="wf-chip-sub">s₀ 768 · z₀ 384</text>
             </a>
             <a href="#view=conditioning" data-arch="conditioning" class="wf-link">
               <rect x="272" y="212" width="112" height="52" rx="6" class="wf-chip wf-chip-link"/>
               <text x="284" y="234" class="wf-chip-label wf-ink-specify">Conditioning</text>
               <text x="284" y="251" class="wf-chip-sub">times, CATH → c</text>
             </a>
             <path d="M384 176 H398" class="wf-rail" marker-end="url(#arch-arrow)"/>
             <path d="M384 238 H398" class="wf-rail" marker-end="url(#arch-arrow)"/>

             <!-- the recycling loop around the trunk -->
             <a href="#view=denoiser" data-arch="denoiser" class="wf-link">
               <rect x="400" y="138" width="210" height="238" rx="8" class="arch-node-loop"/>
               <use href="#arch-ico-cycle" x="410" y="146" width="20" height="20" class="wf-ico arch-ico-loop"/>
               <text x="436" y="160" class="arch-loop-label">RECYCLE × R</text>
               <text x="412" y="178" class="arch-loop-sub">R(i) = 1, 2, 3, 5, 6, 5, 3, 2, 1</text>
             </a>
             <a href="#view=trunk" data-arch="trunk" class="wf-link">
               <rect x="414" y="190" width="182" height="54" rx="6" class="wf-chip wf-chip-link"/>
               <text x="426" y="212" class="wf-chip-label wf-ink-specify">Trunk</text>
               <text x="426" y="230" class="wf-chip-sub">14 layers · 7 pair updates</text>
             </a>
             <a href="#view=layer" data-arch="layer" class="wf-link">
               <rect x="414" y="254" width="88" height="40" rx="6" class="wf-chip wf-chip-link"/>
               <text x="420" y="279" class="wf-chip-label wf-ink-specify">Layer</text>
             </a>
             <a href="#view=pair-update" data-arch="pair-update" class="wf-link">
               <rect x="508" y="254" width="88" height="40" rx="6" class="wf-chip wf-chip-link"/>
               <text x="514" y="279" class="wf-chip-label wf-ink-specify">Pair update</text>
             </a>
             <a href="#view=attention" data-arch="attention" class="wf-link">
               <rect x="414" y="302" width="88" height="40" rx="6" class="wf-chip wf-chip-link"/>
               <text x="420" y="327" class="wf-chip-label wf-ink-specify">Attention</text>
             </a>
             <a href="#view=triangle" data-arch="triangle" class="wf-link">
               <rect x="508" y="302" width="88" height="40" rx="6" class="wf-chip wf-chip-link"/>
               <text x="514" y="327" class="wf-chip-label wf-ink-specify">Triangles</text>
             </a>
             <text x="414" y="364" class="arch-dim">s, z feed the next cycle</text>
             <path d="M610 240 H624" class="wf-rail" marker-end="url(#arch-arrow)"/>
             <a href="#view=heads" data-arch="heads" class="wf-link">
               <rect x="626" y="212" width="90" height="56" rx="6" class="wf-chip wf-chip-link"/>
               <text x="638" y="235" class="wf-chip-label wf-ink-specify">Heads</text>
               <text x="638" y="253" class="wf-chip-sub">v̂ CA, v̂ lat</text>
             </a>
             <path d="M728 240 H742" class="wf-rail" marker-end="url(#arch-arrow)"/>

             <!-- the integration step -->
             <a href="#view=integrator" data-arch="integrator" class="wf-link">
               <rect x="744" y="212" width="86" height="50" rx="6" class="wf-chip wf-chip-link"/>
               <text x="756" y="234" class="wf-chip-label wf-ink-specify">Step</text>
               <text x="756" y="251" class="wf-chip-sub">vf_tsr</text>
             </a>

             <!-- ======================================================== outputs -->
             <text x="872" y="28" class="wf-eyebrow wf-ink-return">COMES BACK</text>
             <a href="#view=ae-decoder" data-arch="ae-decoder" class="wf-link">
               <rect x="872" y="160" width="200" height="102" rx="8" class="wf-node wf-node-return"/>
               <text x="886" y="186" class="wf-node-label">DECODER</text>
               <text x="886" y="206" class="wf-node-sub">CA positions and latents</text>
               <text x="886" y="221" class="wf-node-sub">→ 33 residue classes</text>
               <text x="886" y="236" class="wf-node-sub">→ atom37 coordinates</text>
             </a>
             <a href="../workflow/outputs.html" class="wf-link">
               <rect x="872" y="298" width="200" height="72" rx="8" class="wf-node wf-node-return"/>
               <text x="886" y="324" class="wf-node-label">DESIGN</text>
               <text x="886" y="344" class="wf-node-sub">sequence + all-atom CIF</text>
             </a>

             <!-- ======================================================== training -->
             <text x="20" y="470" class="wf-eyebrow">TRAINED FIRST</text>
             <a href="#view=autoencoder" data-arch="autoencoder" class="wf-link">
               <rect x="20" y="482" width="272" height="80" rx="8" class="wf-node"/>
               <text x="34" y="506" class="wf-node-label">AUTOENCODER</text>
               <text x="34" y="524" class="wf-node-sub">256 M parameters, frozen afterwards</text>
             </a>
             <a href="#view=ae-encoder" data-arch="ae-encoder" class="wf-link">
               <rect x="34" y="532" width="110" height="24" rx="5" class="wf-chip wf-chip-link"/>
               <text x="44" y="549" class="wf-chip-label wf-ink-specify">Encoder</text>
             </a>
             <a href="#view=ae-layer" data-arch="ae-layer" class="wf-link">
               <rect x="152" y="532" width="126" height="24" rx="5" class="wf-chip wf-chip-link"/>
               <text x="162" y="549" class="wf-chip-label wf-ink-specify">AE layer</text>
             </a>
             <text x="394" y="470" class="wf-eyebrow">THEN, FOR THE FLOW</text>
             <a href="#view=training" data-arch="training" class="wf-link">
               <rect x="394" y="482" width="443" height="80" rx="8" class="wf-node"/>
               <text x="408" y="506" class="wf-node-label">TRAINING STEP</text>
               <text x="408" y="526" class="wf-node-sub">encode → interpolate with noise → denoise with R ~ Poisson(3)</text>
               <text x="408" y="543" class="wf-node-sub">→ squared velocity error per modality + insertion loss</text>
             </a>
           </svg>
         </div>
         <iframe class="arch-frame" hidden title="Interactive neural circuit diagram"></iframe>
         <p class="arch-stage-hint"><span class="arch-hint-mouse">drag to pan · Ctrl + scroll to zoom · hover for details · click a box whose details offer "Open diagram" to go deeper · Tab into the diagram, arrow keys move between blocks, Enter opens</span><span class="arch-hint-touch">drag to pan · pinch to zoom · tap a box for details · tap "Open diagram" to go deeper</span></p>
       </div>
     </div>
     <section class="arch-panel"></section>
   </div>
   <script src="../_static/architecture/manifest.js"></script>
   <script src="../_static/architecture/explorer.js"></script>

Reading the diagrams
--------------------

A neural circuit diagram draws arrays as bundles of wires, one wire per axis, and operations as
shapes across them. A wire that passes an operation is an axis it is broadcast over; a wire that
ends at an operation is an axis it consumes. Dashed lines separate the arrays of a product. A bold
box is a block of the model: hover it for its formula, description and code. When its details
offer "Open diagram", click it to open the block as its own diagram; clicking any other box pins
its details. A bracket with a count is a loop: ``T`` for the 60 sampling steps, ``R`` for the
recycles, ``P`` for the 7 repetitions of [layer, pair update, layer] in the trunk and ``L`` for
the 12 layers of each half of the autoencoder. Coloured labels at the top and bottom of a box are
tape slots: values written once and read elsewhere, such as the conditioning ``c``, which every
recycling cycle reads and the trunk passes through its layers.

Axis letters and value symbols are read differently. An axis label is a letter with its size as a
subscript, as p\ :sub:`384` for the pair channels, except on the token axis ``x``, whose size is the
length of the design. The legend beside every diagram lists each axis with its identifier in the
code where it has one and a descriptive name otherwise. Tape-slot
labels are coloured and carry no size, and they and the formulas name values, not axes: ``s`` is
the single representation, ``z`` the pair representation and ``c`` the conditioning. The two
modalities are marked ``CA`` for the CA positions and ``lat`` for the latents, as in ``Step_CA``
and ``v̂ lat``, so ``z`` always means the pair representation. The CA position of a token is
atom37 slot 1: the Cα of an amino acid, C1′ of a nucleotide and the atom itself for an atomized
token, which carries no latent. `Diagrams Explained (Quickly)
<https://vtabbott.io/articles/diagrams-explained-quickly/>`__ introduces the notation in full.

.. list-table:: Sizes of the diagrams
   :header-rows: 1
   :widths: 10 12 28 50

   * - Axis
     - Size
     - Name in the code
     - What it indexes
   * - ``x``
     - design
     - tokens
     - residues, nucleotides and atomized ligand atoms of the design
   * - ``m``
     - 768
     - ``token_dim``
     - the single representation
   * - ``p``
     - 384
     - ``pair_repr_dim``
     - the pair representation
   * - ``u``
     - 256
     - ``dim_cond``
     - the conditioning vector
   * - ``h``, ``d``
     - 12, 64
     - ``nheads``, ``dim_head``
     - attention heads
   * - ``f``
     - 3072
     - ``dim_inner``
     - the SwiGLU of every trunk and autoencoder layer
   * - ``e``
     - 256
     - ``tri_mult_c``
     - the triangle multiplications
   * - ``l``
     - 16
     - ``latent_dim``
     - autoencoder latents per residue or nucleotide
   * - ``T``
     - 60
     - ``nsteps``
     - sampling steps
   * - ``R``
     - 1 to 6
     - ``n_recycle``
     - trunk passes per network evaluation, set per sampling step
   * - ``P``
     - 7
     - ``nlayers`` / ``update_pair_repr_every_n``
     - repetitions of [layer, pair update, layer] in the trunk
   * - ``L``
     - 12
     - ``nlayers`` of the autoencoder
     - layers of the encoder and of the decoder

Where to read next
------------------

.. container:: wf-intro

   A companion page takes the autoencoder's latents apart.

.. raw:: html

   <nav class="arch-guide" aria-label="Where to read next">
     <div class="wf-cards">
       <a class="wf-card" href="expanded-chemistry.html">
         <span class="arch-guide-kicker">Latent space</span>
         <span class="wf-card-title">Thirty-two numbers, six molecular contexts</span>
         <span class="wf-card-body">Shift, walk and add noise to latent channels of the release autoencoder and see the
           decoded protein, DNA, RNA, phosphorylated, noncanonical and glycan contexts.</span>
       </a>
     </div>
   </nav>

.. toctree::
   :hidden:
   :maxdepth: 2

   expanded-chemistry
