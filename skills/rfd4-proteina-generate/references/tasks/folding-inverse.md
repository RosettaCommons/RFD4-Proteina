# Folding and inverse folding

Two mirror-image conditioning problems, both expressed through an **annotated
CIF** (build it with **rfd4-proteina-inputs**). Back to the [skill](../../SKILL.md).

- **Folding** — condition on the *sequence* (`C_SEQ`) and let the model generate
  the structure. "I know what it is; predict where it goes."
- **Inverse folding** — condition on the *backbone atoms only* (`N`, `CA`, `C`,
  `O`) and redesign the sequence. "I know the fold; give me a sequence for it."

## Why the input format matters

The distinction is entirely in what you condition, which lives inside the CIF:

- Backbone-only coordinate conditioning (`N/CA/C/O`) does **not** imply sequence
  — that is exactly how you fix a fold and redesign its sequence (inverse
  folding).
- Fixing **sidechain** `C_CRD` implies `C_SEQ` (atom37 slots fingerprint the
  residue), so it would **leak the native identity**. For inverse folding, never
  fix sidechain atoms — restrict coordinate conditioning to the backbone.
- Folding is the complement: apply `C_SEQ` (known sequence) without pinning
  coordinates, so the model builds the structure.

These are per-residue / per-atom decisions; get them right in
**rfd4-proteina-inputs**, which interrogates exactly this ("know where it is, or
what it is, or both?").

## Run

Once you have the annotated CIF:

```bash
# Inverse folding: backbone-only conditioned CIF, redesign sequence
python src/rfproteina/generate.py \
    inputs=my_backbone.cif \
    num_replicates=8 batch_size=8 out_dir=./runs/inverse_fold

# Folding: sequence-conditioned CIF, generate structure
python src/rfproteina/generate.py \
    inputs=my_sequence.cif \
    num_replicates=8 batch_size=8 out_dir=./runs/fold
```

## Task-specific overrides & metrics

- **`unconditional_refolding`** works as a general refolding check on the
  generated structure (AF3 oracle; needs a container).
- If you conditioned coordinates, `3d_coordinate_conditioning` reports coordinate
  RMSD of the conditioned region.
- Consider low diversity `(0.1,0.1)` for high-designability inverse-folding
  sequences.

## Gotchas

- Fixing a sidechain atom during inverse folding also reveals residue identity
  and accidentally locking the residue's identity. Let **rfd4-proteina-inputs**
  build and validate the CIF; do not hand-edit.
- There is no dedicated "folding" JSON benchmark — these are annotated-CIF
  conditioning problems built from your own structure.
