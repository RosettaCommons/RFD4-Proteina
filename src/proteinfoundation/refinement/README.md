# Gradient-Based Refinement Strategies

Standalone utilities for optimizing protein structures using gradient-based methods with differentiable reward models (typically AlphaFold2).

## Overview

Two complementary refinement strategies:

| Strategy | What it optimizes | Requires Autoencoder | Use Case |
|----------|------------------|---------------------|----------|
| **BindCraft** | Sequence only (structure fixed) | No | Fast sequence optimization |
| **Gradient Ascent** | Latents (structure + sequence) | Yes | Full structure-sequence co-optimization |

Both strategies:
- ✅ **Standalone**: Run on arbitrary PDB/CIF files via CLI
- ✅ **Integrated**: Automatically applied after generation in proteina.py
- ✅ **Unified interface**: Easy to swap via config

## Standalone Usage

### BindCraft Refinement

Optimize sequence with fixed structure:

```bash
# Single file
python proteinfoundation/refine.py \
    --config-name refine_bindcraft \
    inputs=my_binder.pdb \
    output_dir=refined/

# Directory of structures
python proteinfoundation/refine.py \
    --config-name refine_bindcraft \
    inputs=outputs/generated_binders/

# Fast mode (skip soft optimization)
python proteinfoundation/refine.py \
    --config-name refine_bindcraft \
    inputs=my_binder.pdb \
    refinement.strategy.mode=hard_only
```

### Gradient Ascent Refinement

Optimize in latent space (structure + sequence):

```bash
# Requires autoencoder checkpoint
python proteinfoundation/refine.py \
    --config-name refine_gradient_ascent \
    inputs=my_binder.pdb \
    autoencoder_ckpt_path=/path/to/autoencoder.ckpt

# Adjust iterations
python proteinfoundation/refine.py \
    --config-name refine_gradient_ascent \
    inputs=outputs/binders/ \
    refinement.strategy.n_iterations=100
```

## Integrated Usage (with Generation)

Add refinement after generation by including a refinement config:

```bash
# Generate with BindCraft refinement
python proteinfoundation/generate.py \
    --config-name inference_test_with_refinement \
    refinement=bindcraft_full

# Generate with Gradient Ascent refinement
python proteinfoundation/generate.py \
    --config-name inference_test_with_refinement \
    refinement=gradient_ascent

# No refinement (default)
python proteinfoundation/generate.py \
    --config-name inference_test
```

## Configuration

### BindCraft Modes

**Full mode** (`bindcraft_full.yaml`):
- Soft optimization (temperature annealing)
- Hard optimization (discrete sampling)
- Greedy search

**Hard-only mode** (`bindcraft_hard.yaml`):
- Skips soft optimization (faster)
- Hard optimization + greedy only

### Gradient Ascent Parameters

Key parameters in `gradient_ascent.yaml`:
- `n_iterations`: Number of optimization steps
- `line_search_steps`: Step sizes to test (adaptive)
- `update_bb_ca`: Whether to optimize backbone CA (default: false)
- `normalize_logits`: Apply sequence logit normalization

## Requirements

**BindCraft**:
- Differentiable reward model (e.g., AF2)
- No autoencoder needed

**Gradient Ascent**:
- Differentiable reward model (e.g., AF2)  
- Autoencoder checkpoint (REQUIRED)

## Implementation Status

| Strategy | Standalone Interface | BindCraft Phases | Gradient Ascent Core | AF2 Integration |
|----------|---------------------|------------------|----------------------|-----------------|
| BindCraft | ✅ | ⏳ Pending | N/A | ⏳ Pending |
| Gradient Ascent | ✅ | N/A | ⏳ Pending | ⏳ Pending |

## Architecture

Both strategies implement:

```python
class RefinementStrategy(ABC):
    @abstractmethod
    def refine_atom_arrays(
        atom_arrays: list[AtomArray],
        reward_model: RewardModel,
        autoencoder: AutoEncoder | None
    ) -> list[AtomArray]:
        """Standalone refinement interface."""
```

This allows:
1. **CLI usage**: refine.py calls `refine_atom_arrays` directly
2. **Proteina integration**: proteina.py wraps `refine_atom_arrays` for convenience
3. **Easy swapping**: Change one config line to switch strategies

