Glossary
========

This glossary defines the conditioning and inference terms used throughout
RFD4-Proteína. For structure parsing nomenclature such as entities, instances,
chains, and polymer or non-polymer units, see the `AtomWorks glossary
<https://rosettacommons.github.io/atomworks/latest/glossary.html>`_.

Conditioning
------------

Conditioning gives the model information or constraints to preserve while it
generates the rest of the sequence and structure. In an annotated CIF, one or
more standard annotation columns carry each condition. See
:doc:`Preparing inputs <workflow/inputs>` for the complete schema.

Indexed and unindexed residues
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

An **indexed** residue has its position in the designed chain exposed to the
model. Its ``C_IDX`` mask is on. Most residues are indexed by default.

An **unindexed** residue has its ``C_IDX`` mask off. The model knows the
residue exists but is free to place it within the generated chain. This is
useful for motif scaffolding when the motif's absolute sequence location should
not be fixed.

``C_IDXGRP`` assigns residues to an **index group**. Residues in the same group
retain known relative offsets even when their absolute placement is free.
Atomized tokens cannot be unindexed.

RASA
~~~~

**Relative accessible surface area (RASA)** describes how exposed an atom or
residue is relative to a reference maximum. ``C_RAS`` provides binned RASA
conditioning:

* ``0`` is unspecified, so no RASA condition is applied
* ``1`` is buried, with RASA in ``[0.0, 0.1)``
* ``2`` is intermediate, with RASA in ``[0.1, 0.2)``
* ``3`` is exposed, with RASA at least ``0.2``

The training transform computes solvent accessible surface area with the Shrake
and Rupley algorithm, a 1.4 Å probe, ProtOr radii, and 100 sphere points. RASA
conditioning steers surface exposure and the formation of buried ligand pockets.

Common condition aliases
~~~~~~~~~~~~~~~~~~~~~~~~

``C_CRD``
   Fixes selected atom coordinates. Unmasked coordinates are generated.

``C_SEQ``
   Reveals selected residue identities. Unmasked sequence positions are
   designed.

``C_IDX``
   Reveals residue positions in a chain. Turning the mask off makes residues
   unindexed.

``C_IDXGRP``
   Groups residues whose relative sequence offsets are known.

``C_HOT``
   Marks hotspot atoms on the target interface that a designed binder should
   engage.

``C_RAS``
   Requests buried, intermediate, or exposed surface placement.

``C_HB``
   Requests donor, acceptor, or combined hydrogen bond roles for selected atoms.

``C_SEC``
   Requests helix, sheet, or loop secondary structure at selected residues.

``C_CENTROID``
   Supplies a target center of mass for a chain without fixing its individual
   atoms.

``C_CRDGRP``
   Groups atoms with known relative coordinates but possibly unknown global
   placement. Group ``0`` is the absolutely conditioned frame, ``-1`` is
   unconditioned, and every id ``>= 1`` is one internal coordinate group.

``C_INTCRD``
   Internal coordinate conditioning. It gives a group's centered, randomly
   rotated coordinates, so the model knows the internal structure and designs
   the global placement.

Input formats
-------------

Annotated CIF
   The preferred input format. It stores coordinates, chemistry, and model
   conditions in a standard mmCIF file and can be edited with the Python API.

Generation and search
---------------------

Flow matching
   The generative process used by RFD4-Proteína to transform noise into a
   biomolecular sequence and structure over a fixed number of sampling steps.

Vanilla sampling
   One uninterrupted generation trajectory. This is the fastest baseline.

Reward guided search
   Generation that periodically branches trajectories, scores rolled out
   candidates with a reward model, and keeps the strongest ones.

Beam search
   The reward guided strategy that ships with the model. ``search.checkpoints``
   controls when candidates get scored, ``n_branch`` controls children per
   surviving beam, and ``beam_width`` controls how many candidates survive each
   pruning step.

Reward
   A weighted sum of metric outputs named in each checkpoint's ``rewards``.
   Positive weights maximize a metric and negative weights minimize it.
   Optional clipping limits a metric's influence past a threshold.

Oracle
   An external evaluator used during generation or final evaluation. For
   example, the AF3 refolding oracle predicts a generated sequence and scores
   confidence or agreement with the design.

Metrics tag
   A named group of metrics selected through configuration. Top level
   ``metrics_tags`` controls output evaluation; search instead lists individual
   metric providers in each checkpoint's ``metrics`` field.
