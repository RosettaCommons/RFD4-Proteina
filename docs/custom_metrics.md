# Adding Custom Metrics

This guide explains how to write and register your own evaluation metrics without modifying any tracked files. Your code stays local unless you decide to contribute it back.

Both `configs/metrics/custom_metrics/` and `proteinfoundation/metrics/custom_metrics/` are entirely git-ignored — nothing you put in them is ever tracked. `setup.sh` scaffolds an empty `configs/metrics/custom_metrics/all_custom_metrics.yaml` for you on first run; the Python-side directory is created the moment you drop your first metric module into it.

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

Add a module under `proteinfoundation/metrics/custom_metrics/` (the whole directory
is git-ignored). 

### Design metric example

```python
# proteinfoundation/metrics/custom_metrics/residue_count.py
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
# proteinfoundation/metrics/custom_metrics/refolding_residue_count_delta.py
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

Create `configs/metrics/custom_metrics/all_custom_metrics.yaml` (git-ignored;
`setup.sh` scaffolds it for you on first run). Use `# @package metrics` so Hydra
merges your entries directly into the overall metrics config under the right keys.

```yaml
# configs/metrics/custom_metrics/all_custom_metrics.yaml
# @package metrics

# ── Design metrics ───────────────────────────────────────────────────────────
# Omit this section if you have no custom design metrics.
inline_metrics:
  residue_count:
    _target_: proteinfoundation.metrics.custom_metrics.residue_count.ResidueCountMetric
    min_length: 50

# ── Refolding metrics ─────────────────────────────────────────────────────────
# Omit this section if you have no custom refolding metrics.
refolding_metrics:
  refolding_residue_delta:
    _target_: proteinfoundation.metrics.custom_metrics.refolding_residue_count_delta.RefoldingResidueCountDelta
    required_tags_any: [unconditional_refolding]

# ── Pass criteria ─────────────────────────────────────────────────────────────
# Omit this section if you have no custom pass criteria.
pass_criteria:
  my_custom_plddt:
    _target_: proteinfoundation.metrics.oracles.pass_criterion_metric.PassCriterionMetric
    metric_thresholds:
      rf3_standard.plddt:
        min: 75.0
    required_tags_any: [unconditional_refolding]
```

Hydra dict-merges each section (`inline_metrics`, `refolding_metrics`, `pass_criteria`, `refolding_oracles`, `pre_oracle_filters`) into the built-in config — new entries are added; existing built-in entries are preserved.

### Splitting across multiple files

If `all_custom_metrics.yaml` grows unwieldy, add sibling YAML files in the same
`configs/metrics/custom_metrics/` directory (each starting with `# @package metrics`)
and pull them in via a normal Hydra `defaults:` list:

```yaml
# configs/metrics/custom_metrics/all_custom_metrics.yaml
# @package metrics
defaults:
  - my_extra_metrics   # loads configs/metrics/custom_metrics/my_extra_metrics.yaml

inline_metrics:
  residue_count:
    _target_: proteinfoundation.metrics.custom_metrics.residue_count.ResidueCountMetric
```

```yaml
# configs/metrics/custom_metrics/my_extra_metrics.yaml
# @package metrics
refolding_metrics:
  refolding_residue_delta:
    _target_: proteinfoundation.metrics.custom_metrics.refolding_residue_count_delta.RefoldingResidueCountDelta
```

Both files dict-merge into the same `metrics` package. Only `all_custom_metrics.yaml` needs to be referenced from `configs/metrics/all_metrics.yaml`; everything else it pulls in is resolved transitively.

The same applies on the Python side — `proteinfoundation/metrics/custom_metrics/` is a package, so add as many modules as you like (`residue_count.py`, `refolding_residue_count_delta.py`, ...) and reference each via `proteinfoundation.metrics.custom_metrics.<module_name>.<ClassName>`.

### Pass criteria config

`PassCriterionMetric` evaluates one or more values from the merged oracle+refolding metrics dict via `metric_thresholds` (a dict of `metric_key: {min: ..., max: ...}`). The output key is `{config_name}.pass`. No custom Python class needed:

```yaml
pass_criteria:
  backbone_rmsd_tight:
    _target_: proteinfoundation.metrics.oracles.pass_criterion_metric.PassCriterionMetric
    metric_thresholds:
      backbone_rmsd_unconditional.rmsd:   # dot-notation into merged metrics
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
| **Config section** | `inline_metrics` | `refolding_metrics` | `pass_criteria` |
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
