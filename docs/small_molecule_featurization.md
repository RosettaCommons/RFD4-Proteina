# Small-molecule featurization

How small molecules (ligands, cofactors, metals) are featurized in the data pipeline today. 

Covers what the features are, where they live, and which transforms produce them.

## TL;DR

Small molecules are **atomized**: every atom of a non-polymer becomes its own token.

Per-atom scalars (element, formal charge, chirality) and pair-wise bond features
(bond mask, Kekulé bond order, aromaticity) are added as token-level features in the
atom37 representation. Metals are separated into their own molecule and their bonds
typed as COORDINATION at parse time (inside atomworks).

## Where data flows

```
parse   (atomworks types metal bonds as COORDINATION; metals become their own molecule)
      → AtomizeByCCDName / FlagNonPolymersForAtomization
      → AddChiralityFeature
      → ConvertToAtom37Format
      → ComputeTokenLevelBondFeaturesLikeAF3
      → DropoutCoordinationBonds       (training only)
      → DropoutAromaticFlag            (training only)
      → ConvertToTorch
```

* Pipeline assembly: `proteinfoundation/datapipes/pipeline.py` (`get_pipeline`).
* Bond-matrix helpers (used by `ComputeTokenLevelBondFeaturesLikeAF3`) live in
  `proteinfoundation/datapipes/bonds.py`.
* Feature extraction (`Feature` subclasses that read these batch fields and build
  the model-input tensors) lives in
  `proteinfoundation/nn/feature_factory_atom37.py`.

## Atom37 slot convention

The atom37 representation reserves **slot 0** for the residue-level CA position
(used by polymers, unused by atomized tokens) and writes each atomized atom's
per-atom scalars (charge, element, atom-name, ...) into **slot 1**. Anywhere you
see `batch["charge"][:, :, 1]` or `batch["element"][:, :, 1]` in the feature
factory, that's the atom37 convention for fetching the per-atom value of an
atomized token.

See `ConvertToAtom37Format` in `datapipes/pipeline.py`.

## Per-atom (sequence) features

Produced by transforms in `transforms/` and turned into model-input tensors by the
listed `Feature` class.

| Batch key | Source transform | Feature class (atom37 factory) | Shape | Notes |
|---|---|---|---|---|
| `element` | parse → `ConvertToAtom37Format` | `AtomicElementSeqFeat` | `[b, n, 16]` | Learned `Embedding(max_atomic_num=128)` over atomic number; zeros for non-atomized tokens. |
| `charge` | parse → `ConvertToAtom37Format` | `AtomChargeSeqFeat` | `[b, n, 16]` | 11-class formal charge embedding (-4…+6); table initialized from sin/cos of the actual charge values, then learnable. |
| `chirality` | `AddChiralityFeature` | `ChiralitySeqFeat` | `[b, n, MAX_OVERLAPPING_CENTERS * 3]` (= `[b, n, 18]`) | Block-diagonal feature: each chiral center's neighbor positions are replaced by the matching idealized polyhedron template (tetrahedral / square-planar / TBP / octahedral), unit-rotated to align with the actual neighbors. Atoms participating in multiple centers fill consecutive blocks; remaining blocks cycle the occupied ones. At training time blocks are randomly shuffled and a random rotation is applied per center; both are skipped at inference for determinism. |
| `coord_group` | upstream (`C_CRDGRP`) | `CoordGroupSeqFeat` | `[b, n, 37]` | Random-permuted coordinate-group ID, used as a permutation-invariant "same atomized fragment" signal. |

## Pair (bond-graph) features

Produced by `ComputeTokenLevelBondFeaturesLikeAF3` (which calls into
`datapipes/bonds.py`) and read by these `Feature` classes:

| Batch key | Feature class | Shape | Notes |
|---|---|---|---|
| `bond_mask` | `BondMaskPairFeat` | `[b, n, n, 1]` | Boolean: any bond between any atom of token `i` and any atom of token `j`. |
| `bond_order` | `BondOrderPairFeat` | `[b, n, n, 5]` | One-hot Kekulé bond order: 0=none, 1=single, 2=double, 3=triple, 4=coordination. |
| `is_aromatic` | `IsAromaticPairFeat` | `[b, n, n, 1]` | Boolean aromaticity flag (any of `AROMATIC_{SINGLE,DOUBLE,TRIPLE}` or generic `AROMATIC`). |

The bond-graph helpers (`get_bond_order_matrix`, `get_is_aromatic_matrix`,
`get_bond_distance_matrix`) all delegate to a shared
`_build_symmetric_bond_matrix(atom_array, per_bond_values, dtype, fill)`.

## Metals

CCD-style structures often place metal ions on the same chain as the coordinating
ligand and use SINGLE bonds for the coordination. OMOL convention puts metals in
their own molecule with COORDINATION bonds. This is reconciled at parse time:

* **atomworks parse** (`_add_bonds`): retypes every bond touching a metal atom to
  `BondType.COORDINATION` regardless of source (struct_conn metalc, CCD component
  bonds, or inferred), and `add_id_and_entity_annotations` excludes COORDINATION
  bonds by default so a coordinated metal is its own molecule/entity. (This
  replaced the old in-pipeline `IsolateMetalsToOwnChain` transform.)

* **`DropoutCoordinationBonds`** (`transforms/metals.py`, training only):
  two-level stochastic regularizer.
  * Level 1 (`p_all`, default 0.5): with that probability hide **all**
    coordination bonds in the example.
  * Level 2 (`p_individual`, default 0.1): otherwise, drop each individual
    coordination bond independently. Drops are symmetrized so `(i,j)` and
    `(j,i)` always vanish together.
  Modifies `bond_order`, `bond_mask`, and `is_aromatic` in lockstep so all
  three pair features stay consistent.

## Aromatic dropout

**`DropoutAromaticFlag`** (`datapipes/pipeline.py`, training only): with
probability `p` (default 0.1), zeros every entry of `is_aromatic` for the
example. Forces the model not to depend hard on the aromatic annotation, which
is sometimes ambiguous in input data.

## Chirality: idealized-template formulation

The chirality feature (`AddChiralityFeature` →
`build_chirality_block_diagonal` in
`proteinfoundation/transforms/chirality_feature.py`) encodes the **stereochemistry
of each chiral center** — and *only* the stereochemistry, not the actual neighbor
positions. The construction is:

1. **Identify stereogenic centers.** From the parsed `chiral_type` annotation,
   collect every atom that is stereogenic (gated on
   `ChiralType.is_stereogenic()`) and has a supported `(geometry, n_neighbors)`
   pair. Geometries supported today (`IDEAL_POLYHEDRA` in
   `src/foundry/utils/chirals.py`):

   | Geometry | Idealized vertices |
   |---|---|
   | tetrahedral (n=4) | `(±1, ±1, ±1)/√3` (alternating signs) |
   | square-planar (n=4) | `±x̂`, `±ŷ` |
   | trigonal-bipyramidal (n=5) | three equatorial 120°, two axial `±ẑ` |
   | octahedral (n=6) | `±x̂`, `±ŷ`, `±ẑ` |

2. **Precompute one template per stereoisomer.** For each `(geometry, n)`, take
   the n! labelings of the ideal polyhedron's vertices and keep one
   representative per *proper-rotation* orbit (so labelings related by a pure
   rotation collapse, but mirror images do not). For tetrahedral n=4 this gives
   the two enantiomers; octahedral n=6 gives 30 stereoisomer reps; etc. These
   live in `STEREOISOMER_TEMPLATES`.

3. **Classify the actual center.** Center the n bonded neighbors on the
   stereogenic atom, unit-normalize, and find the template (within the
   `(geometry, n)` set) with the lowest residual under `Rotation.align_vectors`.
   That template's vertices, rotated back into the local atom frame, become the
   per-center "idealized neighbor positions". Replacing the actual coordinates
   with the idealized ones is what makes the feature **encode chirality only**,
   independent of bond lengths or any small geometric jitter.

4. **Pack into per-atom blocks.** The model needs a fixed-shape per-atom
   tensor. We allocate `MAX_OVERLAPPING_CENTERS = 6` (the largest supported
   coordination number; one block per potential overlapping center the atom
   participates in) blocks of 3 floats each, giving the
   `[n_atoms, 18]` chirality feature. For each chiral center, we find the
   first block where every neighbor still has a free 3-slot, then write the
   idealized neighbor position into that slot for each neighbor atom. Atoms that
   end up with empty blocks fill them by cycling the occupied ones (so the
   model sees a consistent shape regardless of how many centers an atom
   participates in).

5. **Training-time symmetry breaking.** Block ordering and the cycling-fill
   order are randomly shuffled, and a fresh random rotation is applied to the
   idealized neighbor positions per center, so the network can't latch onto a
   canonical block index or coordinate frame. At inference these are skipped
   (no shuffle, identity rotation) so the feature is deterministic.

The same template library (and the same `classify_stereoisomer` /
`near_tied_templates` calls) backs the `tetrahedral` / `non_tetrahedral`
chiral-accuracy metric in `models/rf3/src/rf3/metrics/chiral.py`. A predicted
center is correct if its single best-match template is among the GT's set of
near-tied templates — that handles geometries with near-degenerate templates
(e.g. axial/equatorial-ambiguous TBP) without rejecting them outright.

## Configs

* Training experiment: `configs/experiment/ncorley/omol_baseline.yaml`
* Network feature list: `configs/nn/local_latents_score_nn_160M_atom37.yaml`
  (`feats_seq` / `feats_pair_repr`)
