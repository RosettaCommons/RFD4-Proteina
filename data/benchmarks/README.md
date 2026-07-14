# Protein Foundation Benchmarks

This directory contains structural benchmark cases used for evaluation and training. Most structures have been converted to CIF using RCSB PDB structures as the canonical ground truth. This document describes the conversion process, which cases were not converted and why, and guidelines for building new benchmarks.

---


## Guidelines for Creating New Benchmarks

### All benchmarks

**DO NOT** save in the legacy, deprecated `.pdb` format!!. This causes **many** downstream problems,
is highly destructive (especially in cases involving non-polymers, non-canonicals, or pTMs), and does not support many annotations. `.pdb` format files should not be trusted in any benchmarks going forward.
Please save any new benchmark structure as a CIF file, ideally using `atomworks.io.utils.io_utils.to_cif_file`. 

### Benchmark Structures Derived from the RCSB PDB 

Follow these steps to create a new benchmark case correctly. The goal is to produce a CIF that is a faithful subset of the RCSB asymmetric unit, with coordinates that exactly match the deposited structure.

#### 1. Parse the RCSB structure with these canonical settings

```python
from atomworks.io import parse
from atomworks.io.config import ParseConfig

config = ParseConfig(
    remove_waters=True, # Can keep waters if strictly necessary, e.g. a catalytic water
    convert_mse_to_met=True,
    remove_ccds=(),       # or can remove crystallization aids
    build_assembly=None,  # asymmetric unit only
    hydrogen_policy="remove",
)

result = parse(pdb_id, config=config)
structure = result.asym_unit  # always start from asym_unit
```

#### 2. Subset as an AtomArray — do not modify bonds

Subset the parsed `AtomArray` directly using boolean masks. Do **not** reconstruct bonds, modify connectivity, or use PDB-level tools for subsetting. AtomWorks tracks bonds internally; manipulating them separately corrupts the structure.

```python
import numpy as np

# Example: keep only chain A, residues 10-80
mask = (
    (structure.chain_id == "A") &
    (structure.res_id >= 10) &
    (structure.res_id <= 80)
)
subset = structure[mask]
```

#### 3. Name the file with the PDB ID

Include the 4-character PDB ID as an underscore-delimited token in the filename.


---

## Conversion of legacy benchmarks

The conversion script can be run on an individual file or on the full benchmark set:

```bash
# Re-run on a single file (full traceback on failure)
python projects/proteinfoundation/convert_pdb_benchmarks_to_cif.py \
    --file projects/proteinfoundation/benchmarks/path/to/file.pdb

# Full batch run (updates all configs and writes conversion_failures.md)
python projects/proteinfoundation/convert_pdb_benchmarks_to_cif.py
```

The report is regenerated each run and written to `benchmarks/conversion_failures.md`.

## Directory Overview

| Directory | Contents | Converted? |
|-----------|----------|------------|
| `mcsa_40/` | Catalytic site atlas structures (40 cases) | ✅ Yes |
| `dna_pdbs/` | DNA/RNA structures | ⚠️ Partial |
| `motif_data_aa/` | Motif scaffolding templates | ✅ Yes |
| `complexa_targets/alpha_proteo_targets/` | Target structures for ComplexA binder design | ⚠️ Partial |
| `complexa_targets/bindcraft_targets/` | De novo binder designs (model-predicted) | ⚠️ Special (see below) |
| `rfbench/` | RosettaFold benchmark cases | ⚠️ Partial |
| `rfbench/sm_binder/` | Small-molecule binder benchmarks (protein excluded) | ✅ Yes |
| `rfd3_ppi_paper_benchmark/` | RFdiffusion3 PPI paper benchmark | ✅ Yes |
| `ss_conditioned_proteina/` | Secondary structure conditioned ProteinA cases | ⚠️ Failed |
| `tip_atom_no_ligand_training_examples/` | AlphaFold model examples (no RCSB source) | ⚠️ No PDB ID |
| `tip_atom_plinder_training_examples/` | PLINDER training examples | ⚠️ Failed |
| `de_novo_proteins/` | De novo designed proteins with RCSB PDB IDs | ❌ Never run |
| `public_complexa_eval/` | Config files only (no structures) | N/A |

---

## Conversion Method

The source of ground truth for all converted structures is the **RCSB PDB asymmetric unit** (or assembly), parsed with AtomWorks. The conversion proceeds as follows:

1. **PDB ID detection.** The filename must contain exactly one underscore-delimited 4-character alphanumeric token that matches a known PDB ID in the local PDB mirror. Files with zero or multiple matches are skipped.

2. **Parse the RCSB reference.** AtomWorks `parse()` is called on the RCSB structure with these settings:
   - `remove_waters=False`
   - `convert_mse_to_met=True`
   - `remove_ccds=()` (all ligands kept)
   - `build_assembly=None` (asymmetric unit)
   - `hydrogen_policy="remove"`

3. **Subset to match the benchmark.** Each benchmark residue is aligned to the RCSB reference by matching residue-level coordinates (tolerance 1e-3 Å, then 1e-4 for tiebreaking). Bond patterns resolve ambiguities. The asymmetric unit is tried first; if it fails, each biological assembly is tried.

4. **Center coordinates.** Output coordinates are centered at the geometric mean of the matched atoms.

5. **Write CIF.** Output is written to the same directory with a `.cif` suffix. The original `.pdb` is deleted on success.

### Special Handling

- **`complexa_targets/bindcraft_targets/`** — Many of these are de novo designed proteins with no RCSB counterpart. Structures without a detectable PDB ID are parsed directly and written as CIF without any RCSB lookup or alignment. **Two exceptions** (`9hag` → `BBF-14_9hag.cif`, `2lag` → `IFNAR2_2lag.cif`) were manually subsetted from their RCSB PDB entries to match the target region used in binder design.
- **`rfbench/sm_binder/`** — These are only used for their small molecules in any benchmarks, and the proteins do not match the associated RCSB PDB entry. Protein atoms are excluded; only the ligand is retained.
- **mcsa_40/M0731_1mt5.pdb** - This PDB ID failed tmol-based sanitization, so it was subsetted directly from the RCSB reference.
- **mcsa_40/M0255_1mg5.pdb** & **mcsa_40/M0375_3ts9.pdb** - These PDB IDs contained crystallization aids (ACT and PO4) which had been removed from the reparsed PDB, so they were subsetted directly from the RCSB reference.
- **mcsa_40/M0907_1rbl.pdb** - This benchmark example had two issues: 1) It contained the crystallization aid FMT so could not be subsetted from the reparsed PDB. This was handled by falling back to the RCSB reference. 2) The mapping algorithm failed to find a low-rmsd state. This is because the closest matching conformers actually came from two different regions of the extremely symmetric `asym_unit`, leading to a high overall RMSD. This was handled manually, leading to a near-zero rmsd.
---

## What Was Not Converted

**Failures are not necessarily safe to use.** Files that were not converted may have been modified in ways that are destructive -- for example, many PDB files do not contain bonding information, or only contain single bonds. This can be very problematic for some models!!

### `de_novo_proteins/` — Already annotated CIFs

This directory was not included in the conversion script as it already contains annotated CIF files deposited with atomworks. It is assumed to be safe but was not manually re-derived from the RCSB reference.

### No PDB ID Found (36 files)

These files contain no underscore-delimited 4-character PDB ID that matches the mirror. They are de novo designs, synthetic constructs, AlphaFold models, or generic DNA sequences with no RCSB counterpart.

**AlphaFold models** (`tip_atom_no_ligand_training_examples/`):
- `AF-A0A0B1TUI3-F1-model_v4.cif`
- `AF-A0A1A7PJN8-F1-model_v4.cif`
- `AF-A0A2U2CUX3-F1-model_v4.cif`
- `AF-A0A316SJY4-F1-model_v4.cif`
- `AF-A0A7C9K389-F1-model_v4.cif`

**De novo / synthetic structures** (`rfbench/`, `dna_pdbs/`):
- `rfbench/theozyme/theo_{0..4}.pdb` — theozyme templates (no RCSB source)
- `rfbench/serine_hydrolase/siteC*.pdb`, `siteD.pdb` — synthetic motifs
- `rfbench/ppi/egfr_n_v1.pdb`, `flu_HA.pdb`, `il12r_v1.pdb`, `insulin_target.pdb`, `pdl1_v1.pdb` — de novo binder targets
- `rfbench/gaa.pdb`, `norn_58res.pdb`, `sh_native.pdb`, `sh_theozyme.pdb`, and various `_ORI_cm*` variants
- `rfbench/sym_gp_A.pdb`, `sym_gp_AB.pdb`, `ca_motif2_2_sym.pdb`, `10_res_atomized_3*.pdb`
- `rfbench/partial_T_example.pdb`, `pipeline_test_0_input.pdb`, `mol_frag_combo_*.pdb`
- `dna_pdbs/CGAGAACATAGTCG.pdb` — synthetic DNA sequence

### Parse Errors (4 files)

These files could not be parsed (malformed charge fields, empty files, or non-standard format), and their RCSB references also failed:

- `complexa_targets/alpha_proteo_targets/1tnf_cropped_fixed.pdb`
- `rfbench/1yzr_no_covalent_ORI_cm2.pdb`
- `rfbench/rsv0_4jhw.pdb`
- `rfbench/rsv5_5tpn.pdb`

### Subset/Alignment Failures (38 files)

The RCSB alignment could not match all residues in the benchmark file to the reference structure. Common causes: repacked sidechains, mutations, non-standard residues, or heavy crops that don't appear in the RCSB entry.

**ComplexA targets** (repacked/modified, use `*_cropped.cif` versions instead):
- `alpha_proteo_targets/1tnf_repacked.pdb`, `3di3_cropped_fixed.pdb`, `3di3_repacked.pdb`, `5o45_cropped_fixed.pdb`, `5o45_repacked.pdb`

**DNA/RNA**:
- `dna_pdbs/1tgh.pdb`

**rfbench cases** (various modifications):
- `rfbench/1a00_d2.pdb`, `1a9o_c3.pdb`, `1dpx.pdb`, `1yzr_no_covalent.pdb`, `1yzr_no_covalent_ORI_cm1.pdb`
- `rfbench/2j0l.pdb`, `3l0f_covale.pdb`, `4qil_rna.pdb`, `4qil_rna_v2.pdb`, `4yl0.pdb`, `4yl0_2.pdb`
- `rfbench/6m91.pdb`, `7s4x_frags.pdb`, `lip_6tp8.pdb`, `ra_5an7.pdb`, `ra_5an7_no_cov.pdb`, `ra_5an7_no_cov_ORI_cm1.pdb`
- `rfbench/ppi/5o45_pdl1.pdb`, `trka_1www_C_optH.pdb`

**ss_conditioned_proteina** (all 5 files failed — likely model-generated coordinates):
- `2o5n_ppi_3d_conditioned.cif`, `2o5n_ppi_3d_conditioned_from_train.cif`
- `3wzs_light_2d_and_seq_conditioned.cif`, `3wzs_light_2d_and_seq_conditioned_from_train.cif`
- `6e7s_unconditioned.cif`

**tip_atom_plinder_training_examples** (all 6 files failed):
- `2yio__1__1.A__1.K.cif`, `5eni__1__1.A__1.B.cif`, `5mnw__1__1.A__1.B.cif`
- `6qlm__2__1.B__1.G.cif`, `7fr3__1__1.A__1.C.cif`, `7z5m__2__1.B__1.L.cif`

### Skipped (1 file)

Files with `relaxed` in the name are skipped because energy-minimized coordinates will not match the RCSB reference:

- `rfbench/ppi/3DI3_clean_relaxed_B.pdb`

---
