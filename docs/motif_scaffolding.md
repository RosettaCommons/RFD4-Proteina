# Motif Scaffolding

Build novel protein scaffolds around fixed structural motifs — either with known (indexed) or unknown (unindexed) sequence positions.

**Back to [README](../README.md)**

---

## Overview

Motif scaffolding generates a complete protein that incorporates one or more structural motifs (e.g., epitopes, binding loops, functional sites). The model fills in the scaffold around the fixed motif coordinates.

There are two modes:

| Mode | When to Use | Input Key |
|------|-------------|-----------|
| **Indexed** | You know where motif residues sit in the final sequence (e.g., loop A16-35 flanked by 5-20 residue linkers) | `contig` |
| **Unindexed** | You only know the 3D positions of motif residues, not their sequence order (e.g., enzyme active sites, discontinuous epitopes) | `unindex` |

## Indexed Motif Scaffolding

### Input Format

```json
{
    "indexed_motif": {
        "dialect": 2,
        "input": "motif_data_aa/1prw_aa.pdb",
        "contig": "5-20,A16-35,10-25,A52-71,5-20",
        "select_fixed_atoms": true
    }
}
```

**Contig syntax:**
- `5-20` — generate 5 to 20 new residues
- `A16-35` — take residues 16–35 from chain A of input (fixed motif)
- Alternating generated/fixed segments builds the scaffold

### Examples

**Single-segment motif:**

```json
{
    "single_seg": {
        "dialect": 2,
        "input": "motif_data_aa/5tpn_aa.pdb",
        "contig": "10-40,A163-181,10-40",
        "select_fixed_atoms": true
    }
}
```

**Multi-segment motif (discontinuous):**

```json
{
    "multi_seg": {
        "dialect": 2,
        "input": "motif_data_aa/1bcf_aa.pdb",
        "contig": "8-15,A92-99,16-30,A123-130,16-30,A47-54,16-30,A18-25,8-15",
        "select_fixed_atoms": true
    }
}
```

**Backbone-only conditioning:**

```json
{
    "backbone_only": {
        "dialect": 2,
        "input": "motif.pdb",
        "contig": "10-30,A10-20,10-30",
        "select_fixed_atoms": {
            "A10-20": "BKBN"
        }
    }
}
```

### Run Indexed Benchmark

```bash
python proteinfoundation/generate.py \
    inputs=benchmarks/motif_scaffold_aa_tasks_indexed.json \
    ckpt_path=./checkpoints \
    ckpt_name=model.ckpt \
    nsamples=100
```

---

## Unindexed Motif Scaffolding

### Input Format

```json
{
    "unindexed_motif": {
        "dialect": 2,
        "input": "motif_data_aa/1prw_aa.pdb",
        "unindex": "A16-35,A52-71",
        "select_fixed_atoms": true,
        "length": "60-105"
    }
}
```

**Key differences from indexed:**
- Uses `unindex` instead of `contig`
- `length` specifies total scaffold length
- The model decides where to place motif residues in the sequence

### Examples

**Discontinuous active site:**

```json
{
    "active_site": {
        "dialect": 2,
        "input": "template.pdb",
        "unindex": "A244,A274,A320,A329,A375",
        "select_fixed_atoms": {
            "A244": "OG,CB",
            "A274": "ND1,CG,CE1,CB,CD2,NE2",
            "A320": "NE2,CD2,CE1",
            "A329": "NH2,CZ,NE,NH1",
            "A375": "OD2,CG,CB,OD1"
        },
        "length": "180-180"
    }
}
```

**Tip-atom only conditioning:**

```json
{
    "tip_atoms": {
        "dialect": 2,
        "input": "template.pdb",
        "unindex": {
            "A10": "TIP",
            "A25": "TIP",
            "A40": "TIP"
        },
        "length": "100-150"
    }
}
```

### Run Unindexed Benchmark

```bash
python proteinfoundation/generate.py \
    inputs=benchmarks/motif_scaffold_aa_tasks_unindexed.json \
    ckpt_path=./checkpoints \
    ckpt_name=model.ckpt \
    nsamples=100
```

---

## Tips

1. **Indexed vs unindexed**: Use indexed when you know the topology (which loops connect which motif segments). Use unindexed for truly discontinuous sites (enzyme active sites).
2. **Atom selection**: `ALL` fixes every atom, `BKBN` fixes backbone only (allowing sidechain redesign), `TIP` fixes sidechain tip atoms only.
3. **Length ranges**: Wider ranges (e.g., `5-20`) give the model more design flexibility. Fixed lengths (e.g., `10-10`) constrain the scaffold tightly.
4. **Multi-segment**: More segments make the problem harder. 1-3 segments work reliably; 4+ segments may need more samples.
5. **Search**: Use best-of-N with `motif_rmsd`-weighted rewards for challenging scaffolding problems.

