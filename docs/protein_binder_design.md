# Protein Binder Design

Design novel protein binders against target proteins using hotspot-conditioned generation.

**Back to [README](../README.md)**

---

## Overview

PPI binder design generates a new protein chain that binds to a specified target protein. The model uses **hotspot conditioning**: you provide a target structure and specify interface residues (hotspots) that the binder should contact.

## Quick Start

### 1. Prepare Input

Create a JSON file specifying the target protein and hotspot residues:

```json
{
    "my_binder": {
        "dialect": 2,
        "input": "target.pdb",
        "contig": "A1-100,80-120",
        "select_fixed_atoms": true,
        "select_hotspots": "A25,A30,A45,A52"
    }
}
```

**Key fields:**

| Field | Description |
|-------|-------------|
| `input` | Path to target protein structure (PDB/CIF) |
| `contig` | `<target_residues>,<binder_length_range>` — target chain kept fixed, binder chain generated |
| `select_hotspots` | Target residues the binder should contact (atom-level selection) |
| `select_fixed_atoms` | Fix all atoms of indexed residues (default: `true`) |

### 2. Run Generation

```bash
python proteinfoundation/generate.py \
    inputs=my_binder.json \
    ckpt_path=./checkpoints \
    ckpt_name=model.ckpt \
    nsamples=100
```

### 3. Evaluate and Filter

Generated structures are automatically evaluated. Key metrics for binder design:

| Metric | Good Value | Description |
|--------|-----------|-------------|
| `rf3.iptm` | > 0.6 | Interface pTM — binding confidence |
| `rf3.min_ipae` | < 10 | Minimum interface PAE — tightest interface region |
| `rf3.plddt` | > 80 | Per-residue confidence |
| `hotspot_obedience` | > 0.8 | Fraction of hotspots contacted by binder |

## Examples

### Homodimer Binder

Design a binder against one chain of a homodimer:

```json
{
    "homodimer_binder": {
        "dialect": 2,
        "input": "homodimer.pdb",
        "contig": "A1-150,80-120",
        "select_hotspots": "A42,A55,A78,A91"
    }
}
```

### Multi-Hotspot Binder with RASA Conditioning

Add solvent accessibility conditioning to prefer certain surface properties:

```json
{
    "rasa_binder": {
        "dialect": 2,
        "input": "target.pdb",
        "contig": "A1-100,80-120",
        "select_hotspots": "A25,A30,A45",
        "select_buried": "A25,A30",
        "select_exposed": "A45"
    }
}
```

### Binder with Sequence Constraints

Fix the target sequence but allow redesign of specific regions:

```json
{
    "constrained_binder": {
        "dialect": 2,
        "input": "target.pdb",
        "contig": "A1-100,80-120",
        "select_hotspots": "A25,A30",
        "select_unfixed_sequence": true
    }
}
```

## Using Search to Improve Success Rate

For higher success rates, use **beam search** with reward-guided generation:

```bash
python proteinfoundation/generate.py \
    --config-name inference_test_beam \
    inputs=my_binder.json \
    ckpt_path=./checkpoints \
    ckpt_name=model.ckpt \
    search.strategy.beam_width=4 \
    search.strategy.n_branch=4
```

See [Search and Refinement](search_and_refinement.md) for details on search strategies and reward configuration.

## Using Refinement to Optimize Binders

After generation, optimize binder sequences with **BindCraft**:

```bash
python proteinfoundation/refine.py \
    --config-name refine_bindcraft \
    inputs=generated_binders/ \
    output_dir=refined_binders/
```

Or optimize in latent space with **gradient ascent**:

```bash
python proteinfoundation/refine.py \
    --config-name refine_gradient_ascent \
    inputs=generated_binders/ \
    autoencoder_ckpt_path=/path/to/autoencoder.ckpt
```

## Benchmark

The repository includes PPI benchmark targets in `benchmarks/`:

```bash
# Run PPI benchmark
python proteinfoundation/generate.py \
    inputs=benchmarks/complexa_paper_ppi.yaml \
    ckpt_path=./checkpoints \
    ckpt_name=model.ckpt \
    nsamples=100
```

## Tips

1. **Hotspot selection**: Choose 3-8 hotspots at the interface. Too many constrains the design; too few gives weak binding.
2. **Binder length**: 80-150 residues is typical. Shorter binders are harder to design.
3. **Multiple samples**: Generate 100+ samples and filter by metrics such as `iptm` and `min_ipae`.
4. **Search + refine**: Beam search + BindCraft refinement can significantly improve success rates.

