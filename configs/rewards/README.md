# Reward Model Configuration

Reward models aggregate metrics into scalar rewards for search-based protein design and refinement.

## Architecture

The reward system is fully modular:

1. **Metrics** (`rfproteina.metrics.*`): Compute individual scores from a generated structure
2. **RewardModel** (`rfproteina.rewards.RewardModel`): Aggregates any metrics with configurable weights

## Available Metrics

### Refolding Metrics

Refold the generated structure with an independent structure prediction model and compare.
Slower but provides strong validation of designability.

| Metric | Outputs | Differentiable | Description |
|--------|---------|---------------|-------------|
| **RF3Metric** | plddt, iptm, pae, ipae, min_ipae | Yes | Refolds with RF3 (PyTorch diffusion model) |
| **AF2Metric** | plddt, ptm, pae, ipae | Yes | Refolds with AlphaFold2 (JAX, via ColabDesign) |

### Force Field Metrics

Evaluate physical plausibility using energy functions. These use either the **Rosetta scoring
function** (via TMol, a GPU-accelerated PyTorch reimplementation of Rosetta) or **MACE**, a
machine-learning force field trained on quantum-mechanical calculations.

| Metric | Outputs | Differentiable | Backend | Description |
|--------|---------|---------------|---------|-------------|
| **TMolBlockPairMetric** | interface_ddg | Yes | TMol (Rosetta) | Region-pair binding ddG with soft repulsion (`soft_ddg` preset in `inline_metrics/tmol_block_pair_metrics.yaml`) |
| **ChargeComplementarityMetric** | charge_score, n_salt_bridges | Yes | NumPy | Salt bridge scoring across interfaces |
| **TMOLMetric** | interface H-bond/elec energies | Yes | TMol (Rosetta) | Interface energy decomposition |
| **MACEEnergyMetric** | median_energy_diff | Yes | MACE (ML force field) | Predicted vs ground-truth energy |

### Structure Quality Metrics

Fast geometric and chemical validation — no refolding or energy computation required.

| Metric | Outputs | Description |
|--------|---------|-------------|
| **PoseBustersMetric** | pass_rate, per-check results | Stereochemical validation of ligand molecules |
| **AtomArrayMetrics** | SS fractions, Rg, composition | General structure statistics |
| **BackboneMetrics** | chain breaks, clashes, floating atoms | Backbone geometry validation |
| **SidechainMetrics** | valid_sidechain_fraction | Sidechain validity |
| **SSEMetrics** | helix/strand/coil fractions | Secondary structure elements |
| **RASAMetrics** | mean_rasa, buried_fraction | Relative accessible surface area |
| **MolprobityMetrics** | clashscore, rama_favored, rotamer_outliers | MolProbity validation |
| **HbondMetrics** | num_hbonds, avg_hbond_energy | Hydrogen bond analysis |

## Adding Your Own Rewards

Adding a custom reward is simple. Every metric follows the same pattern:

**Input:** an `AtomArray` (the generated structure)
**Output:** a `dict[str, float]` (one or more named scores)

### Step 1: Write your metric class

Create a new file in `rfproteina/metrics/` (or anywhere importable):

```python
from biotite.structure import AtomArray
from rfproteina.metrics.metric import Metric


class MyCustomMetric(Metric):
    """My custom metric — computes some score from a structure."""

    def __init__(self, my_param: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.my_param = my_param

    @property
    def kwargs_to_compute_args(self) -> dict:
        return {"predicted_atom_array": "predicted_atom_array"}

    def compute(self, predicted_atom_array: AtomArray) -> dict[str, float]:
        # Your logic here — atom_array in, numbers out
        score = compute_something(predicted_atom_array, self.my_param)
        return {"my_score": float(score)}
```

That's it. The `compute()` method receives an `AtomArray` and returns a dictionary
of named float values.

### Step 2: Use it in a reward config

```yaml
# configs/rewards/my_reward.yaml
metrics:
  my_metric:
    _target_: rfproteina.metrics.my_custom_metric.MyCustomMetric
    my_param: 2.5

  rf3:
    _target_: rfproteina.metrics.rf3_metric.RF3Metric
    ckpt_path: /path/to/rf3_latest.pt

weights:
  my_metric.my_score: 3.0
  rf3.plddt: 1.0
  rf3.iptm: 2.0
```

### Step 3 (optional): Use it programmatically

`RewardModel` is a pure aggregator: it weights metric outputs that the model's metric
managers produce. `reward_metrics_tags` selects which (tag-gated) metrics fire during scoring.

```python
from rfproteina.rewards import RewardModel

reward_model = RewardModel(
    weights={
        "rf3.plddt": 1.0,
        "my_metric.my_score": 3.0,
    },
    reward_metrics_tags=["unconditional_refolding"],
)

# The search harness runs the tagged metrics per candidate, then aggregates:
reward = reward_model.aggregate_metrics_output(metrics_output)  # {"rf3.plddt": ..., ...} -> float
```

### Tips for custom metrics

- **Keep it simple**: `compute()` takes an `AtomArray`, returns `dict[str, float]`.
- **Use `kwargs_to_compute_args`** to map the input names from the pipeline to your `compute()` parameters.
- **Optional lifecycle hooks**: Override `setup()` and `teardown()` if your metric needs expensive resources (model loading, etc.) that should only live during validation.
- **Mix and match**: Your metric automatically works with all search strategies (beam search, best-of-N, vanilla) — no extra integration needed.

## Usage

### Python API

```python
from rfproteina.rewards import RewardModel

# Reward = weighted sum of metric outputs, keyed by "metric_name.output_key".
# The metrics come from the model's configured managers; reward_metrics_tags controls
# which fire during scoring.
reward_model = RewardModel(
    weights={
        "af2.ipae": -5.0,
        "af2.iptm": 2.0,
        "rasa.mean_rasa": 2.0,
        "molprobity.clashscore": -2.0,
    },
    reward_metrics_tags=["af2_refolding"],
)

# The search harness runs the tagged metrics per candidate, then aggregates to a scalar:
reward = reward_model.aggregate_metrics_output(metrics_output)
```

### With Hydra Config

Define the reward under `search.rewards` in an inference_experiment config; `generate.py`
builds the `RewardModel` from it (`weights` + `reward_metrics_tags`):

```yaml
search:
  strategy:
    _target_: rfproteina.search.beam_search.BeamSearchStrategy
    step_checkpoints: [0, 100, 200, 300, 400]
    n_branch: 2
    beam_width: 3
  rewards:
    reward_metrics_tags: [unconditional_refolding]
    weights:
      rf3.plddt: 1.0
      rf3.iptm: 2.0
```

## Available Configs

### Single-Source Rewards

- **`rf3.yaml`**: RF3 refolding confidence (pLDDT, ipTM, ipAE, min_ipAE)
- **`af2_binder.yaml`**: AlphaFold2 binder design
- **`structure_quality.yaml`**: Fast structure-only metrics (no refolding)

### Multi-Source Rewards

- **`multi_source.yaml`**: Combines AF2 + TMOL + structure metrics
