# Reward Model Configuration

Reward models aggregate metrics into scalar rewards for search-based protein design and refinement.

## Architecture

The reward system is fully modular:

1. **Metrics** (`proteinfoundation.metrics.*`): Compute individual scores from a generated structure
2. **RewardModel** (`proteinfoundation.rewards.RewardModel`): Aggregates any metrics with configurable weights

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

Create a new file in `proteinfoundation/metrics/` (or anywhere importable):

```python
from biotite.structure import AtomArray
from proteinfoundation.metrics.metric import Metric


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
    _target_: proteinfoundation.metrics.my_custom_metric.MyCustomMetric
    my_param: 2.5

  rf3:
    _target_: proteinfoundation.metrics.rf3_metric.RF3Metric
    ckpt_path: /path/to/rf3_latest.pt

weights:
  my_metric.my_score: 3.0
  rf3.plddt: 1.0
  rf3.iptm: 2.0
```

### Step 3 (optional): Use it programmatically

```python
from proteinfoundation.rewards import RewardModel
from proteinfoundation.metrics import RF3Metric
from my_module import MyCustomMetric

reward_model = RewardModel(
    metrics={
        "rf3": RF3Metric(ckpt_path="/path/to/rf3_latest.pt"),
        "my_metric": MyCustomMetric(my_param=2.5),
    },
    weights={
        "rf3.plddt": 1.0,
        "my_metric.my_score": 3.0,
    },
)
```

### Tips for custom metrics

- **Keep it simple**: `compute()` takes an `AtomArray`, returns `dict[str, float]`.
- **Use `kwargs_to_compute_args`** to map the input names from the pipeline to your `compute()` parameters.
- **Optional lifecycle hooks**: Override `setup()` and `teardown()` if your metric needs expensive resources (model loading, etc.) that should only live during validation.
- **Mix and match**: Your metric automatically works with all search strategies (beam search, best-of-N, vanilla) — no extra integration needed.

## Usage

### Python API

```python
from proteinfoundation.rewards import RewardModel
from proteinfoundation.metrics import RF3Metric, AF2Metric
from proteinfoundation.metrics.rasa_metrics import RASAMetrics
from proteinfoundation.metrics.molprobity_metrics import MolprobityMetrics

# Compose any metrics
reward_model = RewardModel(
    metrics={
        "af2": AF2Metric(model_name="model_1_ptm", num_recycles=3),
        "tmol": TMOLMetric(),
        "rasa": RASAMetrics(),
        "molprobity": MolprobityMetrics(),
    },
    weights={
        "af2.ipae": -5.0,
        "af2.iptm": 2.0,
        "tmol.num_hbonds": 3.0,
        "rasa.mean_rasa": 2.0,
        "molprobity.clashes": -2.0,
    },
    differentiable=False,  # Set to True only if all metrics support gradients
)

# Compute reward
reward = reward_model.compute_reward(predicted_atom_array=structure)

# Get detailed breakdown
components = reward_model.compute_components(predicted_atom_array=structure)
# Returns: {"af2.ipae": 8.5, "af2.iptm": 0.9, "tmol.num_hbonds": 15, ...}
```

### With Hydra Config

```python
import hydra
from omegaconf import DictConfig
from proteinfoundation.rewards import RewardModel
from proteinfoundation.metrics.metric import MetricManager

@hydra.main(config_path="configs/rewards", config_name="multi_source")
def main(cfg: DictConfig):
    # Instantiate metrics from config
    metric_manager = MetricManager.instantiate_from_hydra(cfg.metrics)

    # Create reward model
    reward_model = RewardModel(
        metrics=metric_manager,
        weights=dict(cfg.weights),
        differentiable=cfg.get("differentiable", False),
    )

    # Use for search
    from proteinfoundation.search import BestOfNStrategy
    strategy = BestOfNStrategy(n_samples=10)
    results = strategy.generate(model, batch, reward_model)
```

## Available Configs

### Single-Source Rewards

- **`rf3.yaml`**: RF3 refolding confidence (pLDDT, ipTM, ipAE, min_ipAE)
- **`af2_binder.yaml`**: AlphaFold2 binder design (differentiable!)
- **`structure_quality.yaml`**: Fast structure-only metrics (no refolding)

### Multi-Source Rewards

- **`multi_source.yaml`**: Combines AF2 + TMOL + structure metrics
