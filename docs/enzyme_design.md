# Enzyme Design

Design novel enzyme scaffolds around catalytic active sites using unindexed motif scaffolding with ligand conditioning.

**Back to [README](../README.md)**

---

## Overview

Enzyme design generates a new protein scaffold that positions catalytic residues and cofactors/substrates in the correct geometry. This uses **unindexed motif scaffolding**: you specify the active-site residues and ligands, and the model builds a scaffold around them without preserving the original sequence ordering.

## Quick Start

### 1. Prepare Input

Create a JSON file specifying catalytic residues and ligands from the MCSA (Mechanism and Catalytic Site Atlas):

```json
{
    "my_enzyme": {
        "dialect": 2,
        "input": "template_enzyme.pdb",
        "unindex": "A64,A86,A90,A114,A137,A145",
        "ligand": "BCA",
        "length": "180-180",
        "select_fixed_atoms": {
            "A64": "O,C",
            "A86": "CB,CA,N,C",
            "A90": "CE1,ND1,NE2,CG,CD2",
            "A114": "N,CA",
            "A137": "NE1,CD1,CE2,CG,CD2,CZ2",
            "A145": "OD2,CG,CB,OD1",
            "BCA": "C6B,C5B,C7B,C4B,O2B,C2B,C3B,C1B,S1P,O1B"
        }
    }
}
```

**Key fields:**

| Field | Description |
|-------|-------------|
| `input` | Path to template enzyme structure |
| `unindex` | Catalytic residues (positions unknown to model — will be placed freely) |
| `ligand` | Cofactor/substrate residue names to include |
| `length` | Target scaffold length |
| `select_fixed_atoms` | Which atoms of each residue/ligand to fix in 3D space |

### 2. Run Generation

```bash
python proteinfoundation/generate.py \
    inputs=my_enzyme.json \
    ckpt_path=./checkpoints \
    ckpt_name=model.ckpt \
    num_replicates=100
```

### 3. Evaluate

Key metrics for enzyme design:

| Metric | Good Value | Description |
|--------|-----------|-------------|
| `motif_rmsd` | < 1.0 Å | Active site geometry preservation |
| `rf3.plddt` | > 80 | Overall fold confidence |
| `molprobity.clashscore` | < 5 | Stereochemistry quality |
| `tmol.total_score` | Low | Rosetta energy (lower is better) |

## Examples

### Minimal Active Site (2-3 Residues)

Simple enzyme with a small catalytic triad:

```json
{
    "small_active_site": {
        "dialect": 2,
        "input": "1chm.pdb",
        "unindex": "A232,A262,A358",
        "ligand": "CMS",
        "length": "180-180",
        "select_fixed_atoms": {
            "A232": "NE2,CD2,CE1",
            "A262": "OE2,CD",
            "A358": "OE1,CD",
            "CMS": "O3,C1,N1,N3,C2"
        }
    }
}
```

### Large Active Site with Multiple Cofactors

Complex enzyme with metal ion and nucleotide cofactors:

```json
{
    "complex_enzyme": {
        "dialect": 2,
        "input": "1pfk.pdb",
        "unindex": "A11,A72,A125,A127,A129,A171",
        "ligand": "ADP,FBP,MG",
        "length": "180-180",
        "select_fixed_atoms": {
            "A11": "N,CA",
            "A72": "NH2,CZ,NE,NH1",
            "A125": "OG1,CB,CA,CG2",
            "A127": "OD2,CG",
            "A129": "OD2,CG",
            "A171": "NH1,CZ,NE,NH2",
            "ADP": "O2B,PB,O1B,O3B,O3A,PA,O1A,O2A,O5',C5',C4',O4'",
            "FBP": "O2P,P1,O1P,O3P,O1,C1,C2,O2,C3,O5",
            "MG": "MG"
        }
    }
}
```

### Backbone-Only Motif Conditioning

Fix only backbone atoms (redesign sidechains):

```json
{
    "backbone_motif": {
        "dialect": 2,
        "input": "template.pdb",
        "unindex": "A100,A141,A144,A164",
        "ligand": "DAK",
        "length": "180-180",
        "select_fixed_atoms": {
            "A100": "BKBN",
            "A141": "BKBN",
            "A144": "ALL",
            "A164": "ALL"
        },
        "redesign_motif_sidechains": true
    }
}
```

## Benchmark: MCSA 41

The repository includes the MCSA-41 benchmark — 41 diverse enzyme active sites:

```bash
# Standard MCSA benchmark (with ligands)
python proteinfoundation/generate.py \
    inputs=benchmarks/mcsa_41.json \
    ckpt_path=./checkpoints \
    ckpt_name=model.ckpt \
    num_replicates=100

# Without ligands
python proteinfoundation/generate.py \
    inputs=benchmarks/mcsa_41_no_ligand.json \
    ckpt_path=./checkpoints \
    ckpt_name=model.ckpt \
    num_replicates=100

# With fixed ligand coordinates
python proteinfoundation/generate.py \
    inputs=benchmarks/mcsa_41_fixed_ligand.json \
    ckpt_path=./checkpoints \
    ckpt_name=model.ckpt \
    num_replicates=100
```

## Tips

1. **Atom selection**: Fix functional atoms (catalytic groups) not just Cα. Use `TIP` for sidechain tips or specify atoms explicitly.
2. **Unindex vs contig**: Use `unindex` for enzyme design (residues placed freely). Use `contig` only if you know the exact scaffold topology.
3. **Ligand completeness**: Include all relevant cofactor atoms. Partial ligands are supported.
4. **Length**: Match the natural enzyme length (±20%). Too short may not accommodate the active site.
5. **Motif RMSD filter**: Filter designs with `motif_rmsd < 1.0 Å` to ensure catalytic geometry is preserved.
6. **Search**: Use best-of-N search with `motif_rmsd`-weighted rewards to focus on active-site accuracy.

