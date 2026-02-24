# Small Molecule Binder Design

Design proteins that bind small molecule ligands using tip-atom and hydrogen-bond conditioning.

**Back to [README](../README.md)**

---

## Overview

Small molecule binder design generates a protein with a binding pocket shaped to accommodate a specific ligand. The model uses **tip-atom conditioning** to place protein residues near the ligand and **hydrogen bond conditioning** to specify donor/acceptor interactions.

## Quick Start

### 1. Prepare Input

Create a JSON specifying the ligand and desired hydrogen bond interactions:

```json
{
    "my_sm_binder": {
        "dialect": 2,
        "input": "ligand_in_pocket.pdb",
        "contig": "150-150",
        "ligand": "FAD",
        "select_hbond_donor": {
            "B412": "N6A,O2B,O3B,O2A,O2P,O4',O3',O2',N3"
        },
        "select_hbond_acceptor": {
            "B412": "N1A,N7A,N3A,O4B,O2B,O3B,O5B,O1A,O2A,N10,N5,N1,O2,O4,N3"
        }
    }
}
```

**Key fields:**

| Field | Description |
|-------|-------------|
| `input` | Path to structure containing the ligand in a binding pocket context |
| `contig` | Protein length to generate (e.g., `"150-150"` for exactly 150 residues) |
| `ligand` | Ligand residue name(s) to include in the design |
| `select_hbond_donor` | Ligand atoms that should donate H-bonds to the protein |
| `select_hbond_acceptor` | Ligand atoms that should accept H-bonds from the protein |

### 2. Run Generation

```bash
python proteinfoundation/generate.py \
    inputs=my_sm_binder.json \
    ckpt_path=./checkpoints \
    ckpt_name=model.ckpt \
    nsamples=100
```

### 3. Evaluate

Key metrics for small molecule binder design:

| Metric | Good Value | Description |
|--------|-----------|-------------|
| `rf3.plddt` | > 80 | Overall fold confidence |
| `hbond.num_hbonds` | > 3 | Number of protein-ligand H-bonds |
| `rasa.mean_binder_rasa` | < 0.3 | Ligand should be buried |
| `tmol.total_score` | Low | Rosetta energy |

## Examples

### Hydrogen Bond–Conditioned Design

Specify which ligand atoms should form H-bonds:

```json
{
    "hbond_binder": {
        "dialect": 2,
        "input": "ligand_context.pdb",
        "contig": "150-150",
        "ligand": "IAI",
        "select_hbond_donor": {
            "B392": "N9,N27"
        },
        "select_hbond_acceptor": {
            "B392": "N13,N12,O8,N5,N28,O31"
        }
    }
}
```

### RASA-Conditioned Pocket Design

Condition on solvent accessibility to control binding pocket shape:

```json
{
    "rasa_pocket": {
        "dialect": 2,
        "input": "ligand_context.pdb",
        "contig": "150-150",
        "ligand": "SAM",
        "select_buried": true,
        "select_hbond_donor": {
            "B386": "N6,O2',O3',N"
        },
        "select_hbond_acceptor": {
            "B386": "N,O,OXT,O4',O3',O2',N7,N3,N1"
        }
    }
}
```

### Tip-Atom Pocket (No Explicit H-bonds)

Let the model decide interactions, just specify the ligand:

```json
{
    "tip_atom_pocket": {
        "dialect": 2,
        "input": "ligand_context.pdb",
        "contig": "120-180",
        "ligand": "ATP"
    }
}
```

## Benchmark

```bash
# H-bond conditioned small molecule binders
python proteinfoundation/generate.py \
    inputs=benchmarks/sm_binder_hbonds.json \
    ckpt_path=./checkpoints \
    ckpt_name=model.ckpt \
    nsamples=100

# RASA-conditioned binders
python proteinfoundation/generate.py \
    inputs=benchmarks/rasa_IAI.json \
    ckpt_path=./checkpoints \
    ckpt_name=model.ckpt \
    nsamples=100
```

## Tips

1. **H-bond specification**: Identify donor/acceptor atoms from the ligand's chemistry. Donors have N-H or O-H groups; acceptors have lone pairs (N, O, F).
2. **Pocket template**: Provide a PDB with the ligand already placed in a reasonable binding context (e.g., from docking or a homolog crystal structure).
3. **Length**: 120-200 residues is typical for single-domain binders.
4. **Post-filtering**: Use Rosetta energy (via `tmol`) and H-bond count to filter designs.
5. **Refinement**: BindCraft refinement can optimize the binder sequence while keeping the ligand-contacting geometry fixed.

