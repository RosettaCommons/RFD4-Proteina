# Unconditional Generation

Generate novel protein structures from scratch without any conditioning.

**Back to [README](../README.md)**

---

## Overview

Unconditional generation creates entirely new protein structures and sequences with no input template. This is useful for exploring novel protein folds, generating diverse scaffolds, or benchmarking model quality.

## Quick Start

### 1. Prepare Input

```json
{
    "monomer_150": {
        "length": "150-150",
        "dialect": 2
    }
}
```

That's it — just specify the desired length.

### 2. Run Generation

```bash
python proteinfoundation/generate.py \
    inputs=my_generation.json \
    ckpt_path=./checkpoints \
    ckpt_name=model.ckpt \
    num_replicates=100
```

## Examples

### Fixed-Length Monomers

```json
{
    "short": {
        "length": "100-100",
        "dialect": 2
    },
    "medium": {
        "length": "150-150",
        "dialect": 2
    },
    "long": {
        "length": "250-250",
        "dialect": 2
    }
}
```

### Variable-Length Generation

```json
{
    "variable": {
        "length": "80-200",
        "dialect": 2
    }
}
```

### Secondary Structure–Conditioned

Generate proteins with specific secondary structure preferences:

```json
{
    "helical": {
        "length": "150-150",
        "select_buried": true,
        "dialect": 2
    }
}
```

### Multi-Chain (Unconditional Dimer)

```json
{
    "dimer": {
        "contig": "80-120,/0,80-120",
        "dialect": 2
    }
}
```

## Benchmarks

```bash
# Standard monomer benchmark (short, medium, long)
python proteinfoundation/generate.py \
    inputs=benchmarks/monomer.json \
    ckpt_path=./checkpoints \
    ckpt_name=model.ckpt \
    num_replicates=100

# Helical monomers
python proteinfoundation/generate.py \
    inputs=benchmarks/monomer_helical.json \
    ckpt_path=./checkpoints \
    ckpt_name=model.ckpt \
    num_replicates=100

# Ig-fold monomers
python proteinfoundation/generate.py \
    inputs=benchmarks/monomer_ig.json \
    ckpt_path=./checkpoints \
    ckpt_name=model.ckpt \
    num_replicates=100
```

## Evaluation

Key metrics for unconditional generation:

| Metric | Good Value | Description |
|--------|-----------|-------------|
| `rf3.plddt` | > 80 | Fold confidence (designability) |
| `rf3.ptm` | > 0.8 | Template modeling score |
| `sse.helix_fraction` | Varies | Secondary structure composition |
| `sidechain.valid_fraction` | > 0.95 | Sidechain quality |
| `novelty` | > 0.5 | TM-score novelty vs PDB |

## Tips

1. **Length**: Shorter proteins (50-100) are easier to generate; longer ones (200+) may need more diffusion steps.
2. **Diffusion steps**: Default is 400. Increase to 800+ for higher quality at the cost of speed.
3. **Self-conditioning**: Enabled by default (`self_cond: true`). This significantly improves generation quality.
4. **Guidance**: For stronger conditioning (when used with conditions), increase `guidance_w` above 1.0.
5. **Diversity**: Generate many samples (100+) and filter by `plddt > 80` for high-quality novel folds.

