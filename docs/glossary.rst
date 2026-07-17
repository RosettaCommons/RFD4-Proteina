Glossary
========

This glossary defines the conditioning and inference terms used throughout
RFD4-Proteína. For structure-parsing nomenclature such as entities, instances,
chains, and polymer/non-polymer units, see the `AtomWorks glossary
<https://rosettacommons.github.io/atomworks/latest/glossary.html>`_.

Conditioning
------------

Conditioning gives the model information or constraints that should be
preserved while the remaining sequence and structure are generated. In an
annotated CIF, each condition is represented by one or more standard annotation
columns. See the :doc:`annotated-CIF guide <guides/annotated_cif_inputs>` for
the complete schema.

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

* ``0`` — unspecified; no RASA condition is applied
* ``1`` — buried, with RASA in ``[0.0, 0.1)``
* ``2`` — intermediate, with RASA in ``[0.1, 0.2)``
* ``3`` — exposed, with RASA at least ``0.2``

The training transform computes solvent-accessible surface area with the
Shrake-Rupley algorithm, a 1.4 Å probe, ProtOr radii, and 100 sphere points.
RASA conditioning can steer surface exposure and the formation of buried
ligand pockets.

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
   Marks target-interface hotspot atoms that a designed binder should engage.

``C_RAS``
   Requests buried, intermediate, or exposed surface placement.

``C_HB``
   Requests donor, acceptor, or combined hydrogen-bond roles for selected
   atoms.

``C_CENTROID``
   Supplies a target center of mass for a chain without fixing its individual
   atoms.

``C_CRDGRP``
   Groups atoms with known relative coordinates but potentially unknown global
   placement.

``C_INTCRD``
   Represents internal-coordinate conditioning.

Input formats
-------------

Annotated CIF
   The preferred input format. It stores coordinates, chemistry, and model
   conditions in a standard mmCIF file and can be edited with the AtomWorks
   PyMOL plugin.

Legacy input specification
   A JSON or YAML ``InputSpecification`` that describes contigs, selections,
   motifs, and conditions separately from the structure file. It remains
   supported for existing workflows.

Generation and search
---------------------

Flow matching
   The generative process used by RFD4-Proteína to transform noise into a
   biomolecular sequence and structure over a fixed number of sampling steps.

Vanilla sampling
   One uninterrupted generation trajectory. This is the fastest baseline.

Reward-guided search
   Generation that periodically branches trajectories, scores rolled-out
   candidates with a reward model, and retains the strongest candidates.

Beam search
   The provided reward-guided strategy. ``step_checkpoints`` controls when
   candidates are scored, ``n_branch`` controls children per surviving beam,
   and ``beam_width`` controls how many candidates survive each pruning step.

Reward
   A weighted sum of metric outputs selected by ``reward_metrics_tags``.
   Positive weights maximize a metric; negative weights minimize it. Optional
   clipping limits a metric's influence after a threshold.

Oracle
   An external evaluator used during generation or final evaluation. For
   example, the AF3 refolding oracle predicts a generated sequence and scores
   confidence or agreement with the design.

Metrics tag
   A named group of metrics selected through configuration. Search uses
   ``reward_metrics_tags`` independently of the top-level ``metrics_tags`` used
   for final outputs.
