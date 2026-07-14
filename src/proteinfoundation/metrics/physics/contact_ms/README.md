# Contact Molecular Surface (CMS)

Distance-weighted interfacial surface area metric for protein-protein and
protein-small molecule interfaces.

```
CMS = Σ  area_i × exp(−weight × dist_i²)
```

where `area_i` is the surface dot area on the query side and `dist_i` is the
distance to the nearest dot on the reference side.

## Why CMS?

CMS captures how tightly two molecular surfaces pack together, rather than
simply measuring how much surface area is buried (like SASA) or returning a
single complementarity value (like Shape Complementarity). It also provides
**per-atom contributions**, enabling residue-level design feedback.

<img width="784" height="442" alt="CMS vs SASA vs Shape Complementarity" src="https://github.com/user-attachments/assets/a05ec752-9b92-4e95-a6dc-14b60129693a" />

*Figure from Brian Coventry's PhD thesis (University of Washington, 2021,
Section 6.2) — illustrating why CMS is more informative than SASA or Shape
Complementarity for protein interface design.*

## Background

CMS is based on the Connolly molecular surface algorithm and weights each
surface dot by proximity to the partner molecule. The original C++
implementation is by Longxing Cao; this Python port makes it accessible to
the broader protein-design community.

**References:**

- Brian Coventry, *Design of protein-protein interactions with a novel
  sampling algorithm and scoring function*, PhD thesis, University of
  Washington, 2021, Section 6.2.
- Lawrence & Colman (1993), *Shape Complementarity at Protein/Protein
  Interfaces*, J. Mol. Biol. 234:946–950.

## Using CMS as a training reward

CMS is `area × quality`, so a model optimising CMS could learn to create
huge low-quality interfaces instead of compact high-quality ones. The
mitigation depends on the use case:

### Protein-protein interactions (PPI)

**Restrict CMS to hotspot residues.** Select a fixed set of target residues
(e.g. known hotspot positions) and sum `per_atom_cms` only over those atoms.
This prevents the model from inflating CMS by expanding the interface.

```python
cms, per_atom_cms, _ = calculate_contact_ms(
    binder_xyz, binder_radii, target_xyz, target_radii, device="cuda",
)

# Sum CMS only over pre-selected hotspot target atoms.
# All target atoms still participate in surface generation (required for
# correct Connolly surface), but only hotspot atoms contribute to the score.
hotspot_cms = per_atom_cms[hotspot_atom_mask].sum()
```

### Small-molecule binders

Reward hacking is **not** a concern here — fully-enveloped small molecules
are exactly what you want. Just make sure CMS is not counted against atoms
that are part of a chemical linker or scaffold rather than the pharmacophore.

For normalisation, divide by the maximum possible CMS (total target surface
area):

```python
max_cms, _, _ = calculate_maximum_possible_contact_ms(
    target_xyz, target_radii, device="cuda",
)
cms_fraction = cms / max_cms  # 0–1 scale
```

## Pipeline stages

Both the NumPy and PyTorch backends implement the same 7-stage pipeline:

1. **Setup** — concatenate binder + target atoms, mark distant atoms as
   blockers (they contribute to the Connolly surface but not to CMS scoring)
2. **Neighbours** — build sorted same-molecule neighbour and cross-molecule
   burial lists via pairwise distance checks
3. **Atom pairs** — find overlapping extended-sphere pairs, queue toroid
   pairs, locate probe-sphere triplet positions
4. **Convex dots** — subdivide exposed sphere caps via latitude arcs +
   longitude circles, collision-check against neighbours
5. **Toroidal dots** — generate saddle-surface dots between bonded atom
   pairs along the torus circle
6. **Concave dots** — generate inward-facing dots inside probe spheres at
   atom triplets, with half-plane rejection and low-probe collision checks
7. **Scoring** — for each buried query dot, find the nearest reference dot
   and compute `area × exp(−w × d²)`; scatter-add per-atom contributions

## Backends

- **TorchCMS** (`gpu_cms.py`) — primary backend. Runs on both CUDA and CPU.
  Faster than the NumPy version in both cases (~400ms vs ~1.5s per structure
  on CPU; ~25-30ms on GPU). Supports `bfloat16` mixed precision on CUDA for
  additional speed-up in trig subdivision.
- **MolecularSurfaceCalculator** (`surface_calculator.py`) — NumPy/SciPy
  reference implementation. Useful for golden-value regression tests and
  environments without PyTorch.

## Quick start

### Using the public API

```python
from proteinfoundation.metrics.physics.contact_ms import (
    calculate_contact_ms,
    calculate_maximum_possible_contact_ms,
    extract_cms_inputs,
)

# Extract inputs from a biotite AtomArray
binder_xyz, binder_radii = extract_cms_inputs(atom_array, binder_mask)
target_xyz, target_radii = extract_cms_inputs(atom_array, target_mask)

# NumPy backend (default)
cms, per_atom, calc = calculate_contact_ms(
    binder_xyz, binder_radii, target_xyz, target_radii,
)

# PyTorch backend on GPU
cms, per_atom, _ = calculate_contact_ms(
    binder_xyz, binder_radii, target_xyz, target_radii, device="cuda",
)

# Maximum possible CMS (total surface area) for normalisation
max_cms, per_atom_max, _ = calculate_maximum_possible_contact_ms(
    target_xyz, target_radii, device="cuda",
)
```

### Using TorchCMS directly

```python
import torch
from proteinfoundation.metrics.physics.contact_ms import TorchCMS

calc = TorchCMS(device="cuda")
t = lambda x: torch.as_tensor(x, dtype=torch.float32, device="cuda")

# Target-side CMS
cms, per_atom = calc.compute(
    t(binder_xyz), t(binder_radii), t(target_xyz), t(target_radii),
)

# Both sides in one pipeline run (avoids recomputing the surface)
(t_cms, t_pa), (b_cms, b_pa) = calc.compute_both_sides(
    t(binder_xyz), t(binder_radii), t(target_xyz), t(target_radii),
)

# Maximum possible CMS for a single molecule
max_cms, per_atom_max = calc.compute_max_cms(t(target_xyz), t(target_radii))
```

### Using raw radii (without biotite)

```python
from proteinfoundation.metrics.physics.contact_ms import (
    calculate_contact_ms,
    get_radii_from_names,
)

# Prepare radii from residue/atom names — do NOT supply your own!
# CMS requires specific radii from Lawrence & Colman (1993).
binder_radii = get_radii_from_names(binder_res_names, binder_atom_names)
target_radii = get_radii_from_names(target_res_names, target_atom_names)

cms, per_atom, calc = calculate_contact_ms(
    binder_xyz, binder_radii, target_xyz, target_radii,
)

# Binder-side CMS (reuses the same surface, no recomputation)
binder_cms, binder_per_atom = calc.compute_contact_ms(target_side=False)
```

### As a proteinfoundation metric

```python
from proteinfoundation.metrics.physics.contact_ms import ContactMolecularSurfaceMetric

metric = ContactMolecularSurfaceMetric(
    compute_binder_side=True,
    compute_max_target_cms=True,
    device="auto",  # "cuda" if available, else "cpu"
)
results = metric.compute(predicted_atom_array=structure)
# results["contact_ms"]            — target-side CMS (Å²)
# results["contact_ms_binder"]     — binder-side CMS (Å²)
# results["max_contact_ms_target"] — total target surface area (Å²)
# results["contact_ms_fraction"]   — CMS / max CMS
```

## Small-molecule support

CMS supports protein-small molecule interfaces out of the box. Radii for
common drug-like atoms (C, N, O, S, P, F, Cl, Br) are provided via universal
prefix fallbacks in `radii.py`. Ligand residues with names starting with
`LG` have a dedicated lookup table for aromatic, charged, and polar atom
types.

## Module structure

| File                    | Description                                      |
|-------------------------|--------------------------------------------------|
| `__init__.py`           | Package exports and quick-start docstring        |
| `api.py`                | Public functions: `calculate_contact_ms`, etc.   |
| `metric.py`             | `ContactMolecularSurfaceMetric` class            |
| `gpu_cms.py`            | `TorchCMS` — PyTorch backend (primary)           |
| `surface_calculator.py` | `MolecularSurfaceCalculator` — NumPy backend     |
| `data_structures.py`    | Shared constants and NumPy struct-of-arrays      |
| `geometry.py`           | Pure math helpers (subdivision, collision)        |
| `radii.py`              | Dict-based radii lookup (Lawrence & Colman 1993) |

## Type annotations

All public and internal function signatures use
[jaxtyping](https://github.com/google/jaxtyping) shape annotations for
both `torch.Tensor` and `np.ndarray` parameters. Dimension names follow a
consistent convention:

| Dim  | Meaning                                   |
|------|-------------------------------------------|
| `Nb` | Number of binder atoms                    |
| `Nt` | Number of target atoms                    |
| `N`  | Total atoms (Nb + Nt) or single molecule  |
| `Nq` | Number of query-side atoms                |
| `K`  | Max neighbour slots (padded)              |
| `G`  | Good (active, non-dead) atoms             |
| `P`  | Atom pairs                                |
| `C`  | Convex-eligible atoms                     |
| `D`  | Surface dots                              |
| `M`  | Latitude subdivisions                     |

Inline shape comments (e.g. `# (N, K, 3)`) at the end of lines use the
same dimension names.
