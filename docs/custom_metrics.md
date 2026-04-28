# Adding Custom Metrics

This guide explains how to write and register your own evaluation metrics without modifying any tracked files. Your code stays local unless you decide to contribute it back.

> **Broadly useful?** If your metric measures something general, consider contributing it. Move the class to a tracked file under `proteinfoundation/metrics/`, add a YAML entry to a tracked config file, and open a pull request. See [CLAUDE.md](../CLAUDE.md) for contribution guidelines.

---

## Three metric categories

There are three ways a custom metric can be run, depending on what inputs it needs.

### 1. Design metrics — operate on the raw generated structure

Run inline in `validation_step` / `evaluate.py Phase 1` on each generated design, before any folding happens.

**Available kwargs** (via `kwargs_to_compute_args`):

| Key | Type | Description |
|---|---|---|
| `predicted_atom_array` | `AtomArray` | The generated structure |
| `ground_truth_atom_array` | `AtomArray` | The input/reference structure |
| `extra_info` | `dict` | Metadata: `example_id`, `metrics_tags`, `output_base_dir`, `chain_of_interest` |

### 2. Refolding metrics — compare designed vs. refolded structure

Run after an oracle folds the design. Receive the designed structure as ground truth and the oracle-refolded structure as predicted. Both are bare `AtomArray` (not `AtomArrayStack`).

**Available kwargs**:

| Key | Type | Description |
|---|---|---|
| `ground_truth_atom_array` | `AtomArray` | The designed structure |
| `predicted_atom_array` | `AtomArray` | The oracle-refolded structure |
| `extra_info` | `dict` | Metadata: `example_id`, `metrics_tags` |

### 3. Pass criteria — threshold one metric to produce a pass/fail flag

Use the built-in `PassCriterionMetric` with `metric_key` + optional `min`/`max`. No custom Python needed. See [pass criteria config](#pass-criteria-config) below.

---

## How `kwargs_to_compute_args` works

`kwargs_to_compute_args` is a property on your metric class that routes kwargs into `compute()`:

```python
@property
def kwargs_to_compute_args(self) -> dict:
    return {
        # compute() param name    kwarg key path (tuple)
        "atom_array":  ("predicted_atom_array",),       # top-level key
        "example_id":  ("extra_info", "example_id"),    # nested key
        "out_dir":     ("extra_info", "output_base_dir"),
    }
```

- Single-element tuple `("predicted_atom_array",)` → reads `kwargs["predicted_atom_array"]`.
- Two-element tuple `("extra_info", "example_id")` → reads `kwargs["extra_info"]["example_id"]`.

---

## Step 1 — Write the Python class

Create `proteinfoundation/metrics/custom_metrics.py` (this path is git-ignored).

### Design metric example

```python
# proteinfoundation/metrics/custom_metrics.py
from biotite.structure import AtomArray
from foundry.metrics.metric import Metric


class ResidueCountMetric(Metric):
    """Count residues in the generated structure.

    Args:
        min_length: If set, also report whether the design meets this minimum.
    """

    def __init__(self, min_length: int | None = None, **kwargs):
        super().__init__(**kwargs)
        self.min_length = min_length

    @property
    def kwargs_to_compute_args(self) -> dict:
        return {"atom_array": ("predicted_atom_array",)}

    def compute(self, atom_array: AtomArray) -> dict[str, float]:
        n_residues = len(set(zip(atom_array.chain_id, atom_array.res_id)))
        result = {"n_residues": float(n_residues)}
        if self.min_length is not None:
            result["meets_min_length"] = float(n_residues >= self.min_length)
        return result
```

Key points:
- Always call `super().__init__(**kwargs)` so that tag filtering (`required_tags_any`, etc.) is picked up from the YAML config.
- Return a flat `dict[str, float]`. Keys become column names in `per_example_metrics.csv`, prefixed with the metric's config name (e.g. `residue_count.n_residues`).

### Refolding metric example

Receives bare `AtomArray` inputs — no unwrapping needed.

```python
from biotite.structure import AtomArray
from biotite.structure.residues import get_residues
from foundry.metrics.metric import Metric


class RefoldingResidueCountDelta(Metric):
    """Compare residue count between the designed and refolded structure."""

    @property
    def kwargs_to_compute_args(self) -> dict:
        return {
            "designed":  ("ground_truth_atom_array",),
            "refolded":  ("predicted_atom_array",),
        }

    def compute(self, designed: AtomArray, refolded: AtomArray) -> dict[str, float]:
        designed_count = len(get_residues(designed)[0])
        refolded_count = len(get_residues(refolded)[0])
        return {"residue_count_delta": float(refolded_count - designed_count)}
```

---

## Step 2 — Register in the config file

Create `configs/custom_metrics/custom_metrics.yaml` (also git-ignored). Use `# @package _global_` so Hydra merges your entries directly into the root config under the right MetricManager keys.

```yaml
# configs/custom_metrics/custom_metrics.yaml
# @package _global_

# ── Design metrics ───────────────────────────────────────────────────────────
# Omit this section if you have no custom design metrics.
design_metric_manager:
  metrics:
    residue_count:
      _target_: proteinfoundation.metrics.custom_metrics.ResidueCountMetric
      min_length: 50

# ── Refolding metrics ─────────────────────────────────────────────────────────
# Omit this section if you have no custom refolding metrics.
refolding_metric_manager:
  metrics:
    refolding_residue_delta:
      _target_: proteinfoundation.metrics.custom_metrics.RefoldingResidueCountDelta
      required_tags_any: [unconditional_refolding]

# ── Pass criteria ─────────────────────────────────────────────────────────────
# Omit this section if you have no custom pass criteria.
pass_criteria_metric_manager:
  metrics:
    my_custom_plddt:
      _target_: proteinfoundation.metrics.pass_criterion_metric.PassCriterionMetric
      metric_key: rf3_standard.plddt
      min: 75.0
      required_tags_any: [unconditional_refolding]
```

Hydra dict-merges populated sections into the existing MetricManager configs — new entries are added; existing built-in entries are preserved.

### Pass criteria config

`PassCriterionMetric` evaluates one value from the merged oracle+refolding metrics dict. The output key is `{config_name}.pass` (e.g. `my_custom_plddt.pass`). No Python class needed:

```yaml
pass_criteria_metric_manager:
  metrics:
    backbone_rmsd_tight:
      _target_: proteinfoundation.metrics.pass_criterion_metric.PassCriterionMetric
      metric_key: backbone_rmsd_unconditional.rmsd   # dot-notation into merged metrics
      max: 1.0
      required_tags_any: [unconditional_refolding]
```

---

## Step 3 — Run evaluation

The custom metrics file is picked up automatically when evaluate.py includes the optional entry in its defaults. No extra flags needed:

```bash
python proteinfoundation/evaluate.py \
    input_dir=/path/to/structures \
    out_dir=/path/to/output \
    +refolding_oracles=rf3_all \
    +refolding_metrics=standard \
    +pass_criteria=standard \
    refolding_oracles.rf3_standard.ckpt_path=/path/to/rf3_latest.pt \
    refolding_oracles.rf3_ppi.ckpt_path=/path/to/rf3_latest.pt
```

Results appear in `design_metrics.csv` and `refolding_metrics.csv`. Column names are prefixed with the config entry key:
- Design metric: `residue_count.n_residues`, `residue_count.meets_min_length`
- Refolding metric: `refolding_residue_delta.residue_count_delta`
- Pass criterion: `my_custom_plddt.pass`

---

## Quick reference

| | Design metric | Refolding metric | Pass criterion |
|---|---|---|---|
| **Config section** | `design_metric_manager.metrics` | `refolding_metric_manager.metrics` | `pass_criteria_metric_manager.metrics` |
| **Key inputs** | `predicted_atom_array` (bare `AtomArray`) | `ground_truth_atom_array` + `predicted_atom_array` (both bare `AtomArray`) | `extra_info["metrics"]` (merged oracle+refolding dict) |
| **When** | Before refolding | After oracle folds the structure | After refolding metrics |
| **Slow?** | No | No (oracle already ran) | No |

---

## Contributing back

```bash
# 1. Create a branch before making any changes
git switch -c feat/my-metric-name

# 2. Add your class to a tracked metrics file under proteinfoundation/metrics/

# 3. Add a YAML entry to an existing tracked config

# 4. Stage all your changes and commit
git add .
git commit -m "feat: describe your metric here"
git push -u origin HEAD

# 5. Navigate to the foundry-dev GitHub page and open a pull request in the web interface
```

See [CLAUDE.md](../CLAUDE.md) for the full commit message format and PR checklist.
