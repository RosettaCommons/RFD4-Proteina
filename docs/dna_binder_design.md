# DNA/RNA Binder Design

Design novel protein binders against DNA or RNA targets.

**Back to [README](../README.md)**

---

## Overview

DNA/RNA binder design generates a protein that binds to a specified nucleic acid duplex or single strand. The model conditions on the nucleic acid structure and generates a complementary protein interface, similar to transcription factors or other DNA/RNA-binding proteins.

## Quick Start

### 1. Prepare Input

```json
{
    "my_dna_binder": {
        "dialect": 2,
        "input": "dna_target.pdb",
        "contig": "B83-92,/0,C93-102,/0,50-80",
        "length": "70-100",
        "select_unfixed_sequence": false
    }
}
```

**Key fields:**

| Field | Description |
|-------|-------------|
| `input` | Path to DNA/RNA structure (PDB/CIF) |
| `contig` | Fixed nucleic acid chains + generated protein chain(s). `/0` separates chains. |
| `length` | Total design length |
| `select_unfixed_sequence` | `false` = fix nucleic acid sequence; `true` = allow sequence redesign |

**Contig syntax for DNA binders:**
- `B83-92,/0,C93-102` — two DNA strands from chains B and C
- `/0` — chain break
- `50-80` — generate a protein chain of 50-80 residues

### 2. Run Generation

```bash
python proteinfoundation/generate.py \
    inputs=my_dna_binder.json \
    ckpt_path=./checkpoints \
    ckpt_name=model.ckpt \
    num_replicates=100
```

## Examples

### Single-Chain DNA Binder

Design one protein chain that binds a DNA duplex:

```json
{
    "single_chain": {
        "dialect": 2,
        "input": "dna_cifs/7n5u.cif",
        "contig": "B83-92,/0,C93-102,/0,50-80",
        "length": "70-100",
        "select_unfixed_sequence": true
    }
}
```

### Homodimer DNA Binder

Design a homodimeric protein that binds DNA:

```json
{
    "dimer_binder": {
        "dialect": 2,
        "input": "dna_cifs/7n5u.cif",
        "contig": "B83-92,/0,C93-102,/0,50-80,/0,50-80",
        "length": "120-180",
        "select_unfixed_sequence": false
    }
}
```

### DNA Binder with Structure Conditioning

Fix both structure and sequence of the DNA, generate protein around it:

```json
{
    "structure_conditioned": {
        "dialect": 2,
        "input": "dna_cifs/7m5w.cif",
        "contig": "B139-148,/0,C149-158,/0,50-80",
        "length": "70-100",
        "select_unfixed_sequence": false
    }
}
```

## Benchmark

```bash
# DNA binder benchmark (multiple targets)
python proteinfoundation/generate.py \
    inputs=benchmarks/paper_dna.json \
    ckpt_path=./checkpoints \
    ckpt_name=model.ckpt \
    num_replicates=100
```

The benchmark includes targets from PDB structures `7n5u`, `7rte`, `7m5w`, `7rcu`, `1tgh` with single-chain, dimer, and structure-conditioned variants.

## Tips

1. **Chain notation**: DNA strands are typically chains B and C (or later letters). Check your PDB file for chain assignments.
2. **Sequence conditioning**: Set `select_unfixed_sequence: true` to allow the model to optimize protein sequence freely; `false` provides the input sequence as a hint.
3. **Dimer design**: Use two length ranges separated by `/0` (e.g., `50-80,/0,50-80`) to generate a homodimeric binder.
4. **Evaluation**: Use `rf3.iptm` for binding confidence and `rf3.plddt` for fold quality.

