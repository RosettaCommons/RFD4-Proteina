# Adding Custom Metrics

This guide explains how to write and register your own evaluation metrics without modifying any tracked files. Your code stays local unless you decide to contribute it back.

> **Broadly useful?** If your metric measures something general — backbone quality, sequence properties, structural features that aren't specific to your design problem — consider contributing it. Move the class to a new or existing file under `proteinfoundation/metrics/`, add a YAML entry to a new or existing yaml, and open a pull request. See [CLAUDE.md](../CLAUDE.md) and [DEVELOPMENT.md](DEVELOPMENT.md) for contribution guidelines. You don't need to be a git expert: the main step is creating a branch with `git switch -c feat/my-metric-name` before you start, then following the PR instructions in those files.

---

## Two kinds of metric

There are two places a custom metric can live, depending on what inputs it needs.

### 1. Design metrics — operate on the raw generated structure

These run directly on each generated design. They are the most common type and the easiest to write.

**Available inputs** (passed as keyword arguments by the outer `MetricManager`):

| Key | Type | Description |
|---|---|---|
| `predicted_atom_array` | `AtomArray` | The generated structure |
| `ground_truth_atom_array` | `AtomArray` | The input/reference structure |
| `extra_info` | `dict` | Metadata: `example_id`, `metrics_tags`, `output_base_dir`, `chain_of_interest` |
| `network_input` | `dict \| None` | Raw model inputs (rarely needed) |
| `prediction_metadata` | `Any \| None` | Per-design metadata from generation (rarely needed) |

Use `kwargs_to_compute_args` to select only the keys you need, and rename them to something semantically meaningful for your `compute()` method.

### 2. Refolding metrics — operate on RF3-refolded structures

These are nested *inside* an `RF3Metric.metric_manager`. RF3 first refolds each design, then passes the result to these inner metrics. Use them when your metric needs a comparison between the designed structure and its independent refolded prediction.

**Available inputs** (passed by the inner `RF3Metric.metric_manager`):

| Key | Type | Description |
|---|---|---|
| `ground_truth_atom_array` | `AtomArrayStack` (depth 1) | The original designed structure (used as "ground truth" for the refolding comparison) |
| `predicted_atom_array` | `AtomArrayStack` (depth 1) | The RF3-refolded structure |

> The naming is from the perspective of the refolding task: the designed structure is the "input" and the refolded structure is the "prediction". Use the `input_kwarg_names` config key in the YAML to remap these if your metric's `compute()` arguments have different names.
>
> The inner manager always passes single-element stacks. Use `unwrap_atom_array` from `proteinfoundation.metrics.metrics_utils` to extract the bare `AtomArray`, which will also assert the depth is 1.

---

## How `kwargs_to_compute_args` works

`kwargs_to_compute_args` is a property on your metric class that tells the framework how to route keyword arguments into your `compute()` method. It returns a `dict` mapping your compute parameter name → a tuple representing the path into the flat kwargs dictionary.

```python
@property
def kwargs_to_compute_args(self) -> dict:
    return {
        # compute() arg name   kwarg key path (tuple)
        "atom_array":      ("predicted_atom_array",),        # top-level key → compute arg renamed
        "example_id":      ("extra_info", "example_id"),     # nested key
        "out_dir":         ("extra_info", "output_base_dir"), # nested key
    }
```

- A single-element tuple like `("predicted_atom_array",)` reads `kwargs["predicted_atom_array"]`.
- A two-element tuple like `("extra_info", "example_id")` reads `kwargs["extra_info"]["example_id"]`.
- The dict key (left side) is the **name used in your `compute()` signature**, which can differ from the kwarg key. Giving it a meaningful name makes `compute()` easier to read.

If you omit `kwargs_to_compute_args` entirely (return `None`), all kwargs are forwarded directly to `compute()` — useful for quick experiments but can make signatures verbose.

---

## Step 1 — Write the Python class

Create `proteinfoundation/metrics/custom_metrics.py` (this path is git-ignored).

### Design metric example

A metric that counts residues and optionally filters by a minimum length threshold:

```python
# proteinfoundation/metrics/custom_metrics.py
from biotite.structure import AtomArray
from foundry.metrics.metric import Metric


class ResidueCountMetric(Metric):
    """Count residues in the generated structure.

    This is a minimal example. Replace with your own logic.

    Args:
        min_length: If set, also report whether the design meets this minimum.
            Defaults to ``None`` (no threshold check).
    """

    def __init__(self, min_length: int | None = None, **kwargs):
        super().__init__(**kwargs)
        self.min_length = min_length

    @property
    def kwargs_to_compute_args(self) -> dict:
        # Rename the kwarg to a more descriptive compute() argument name.
        return {
            "atom_array": ("predicted_atom_array",),
        }

    def compute(self, atom_array: AtomArray) -> dict[str, float]:
        """Return residue count (and optionally a pass/fail flag).

        Args:
            atom_array: The generated structure.

        Returns:
            Dictionary with ``n_residues`` and optionally ``meets_min_length``.
        """
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

A metric that runs *inside* an `RF3Metric` and compares a scalar property of the designed vs. refolded structure:

```python
from biotite.structure import AtomArrayStack
from biotite.structure.residues import get_residues
from foundry.metrics.metric import Metric
from proteinfoundation.metrics.metrics_utils import unwrap_atom_array


class RefoldingResidueCountDelta(Metric):
    """Compare residue count between the designed and refolded structure.

    Nested inside an RF3Metric.metric_manager. Receives the designed structure
    as 'ground truth' and the refolded structure as 'predicted'.

    This is a toy example — in practice you'd compute something more meaningful
    from the refolded structure (e.g. secondary-structure agreement, interface area).
    """

    @property
    def kwargs_to_compute_args(self) -> dict:
        # Inside RF3Metric, the designed structure is "ground_truth_atom_array"
        # and the refolded structure is "predicted_atom_array".
        return {
            "designed":  ("ground_truth_atom_array",),
            "refolded":  ("predicted_atom_array",),
        }

    def compute(self, designed: AtomArrayStack, refolded: AtomArrayStack) -> dict[str, float]:
        """Return the difference in residue count between designed and refolded structure."""
        # The inner RF3 manager passes single-element AtomArrayStacks; unwrap to bare AtomArrays.
        designed_aa = unwrap_atom_array(designed)
        refolded_aa = unwrap_atom_array(refolded)

        designed_count = len(get_residues(designed_aa)[0])
        refolded_count = len(get_residues(refolded_aa)[0])

        return {"residue_count_delta": float(refolded_count - designed_count)}
```

---

## Step 2 — Register in the config file

Create `configs/metrics/custom_metrics.yaml` (also git-ignored).

### Design metric config

```yaml
# configs/metrics/custom_metrics.yaml

# Example: count residues, require at least 50
residue_count:
  _target_: proteinfoundation.metrics.custom_metrics.ResidueCountMetric
  min_length: 50                          # passed to __init__
  required_tags_any: ["unconditional_refolding"]  # only run for these tag(s)
```

- Additional constructor arguments (beyond `required_tags_any` / `prohibited_tags` / `required_tags_all`) are specified as plain keys next to `_target_`.
- Omit the tag filter keys entirely if the metric should always run.

### Refolding metric config

Refolding metrics are nested inside an existing or new `RF3Metric` entry. You can add them to an existing RF3 config by overriding it in `custom_metrics.yaml`, but the simplest approach is to define a new RF3Metric entry that embeds your inner metric:

```yaml
# configs/metrics/custom_metrics.yaml

# Wrap an RF3Metric so we can run our custom refolding analysis.
# This reuses the same RF3 checkpoint as the built-in metrics.
rf3_custom:
  _target_: proteinfoundation.metrics.rf3_metric.RF3Metric
  ckpt_path: ${oc.env:DATA_PATH}/rf3_checkpoints/rf3-with-confidence-xaira-epoch-0711-fixed.ckpt
  num_steps: 50
  n_recycles: 5
  required_tags_any: ["unconditional_refolding"]
  metric_manager:
    _target_: foundry.metrics.metric.MetricManager
    metrics:
      refolding_residue_delta:
        _target_: proteinfoundation.metrics.custom_metrics.RefoldingResidueCountDelta
        # No input_kwarg_names needed here because our kwargs_to_compute_args
        # already uses the correct key names ("ground_truth_atom_array",
        # "predicted_atom_array") that the RF3 inner manager passes.
```

> **Note on `input_kwarg_names`**: This is not necessary for most custom `Metric`s, and is only used in the `StructureComparisonMetric` to handle the different naming conventions of the RF3 outputs and the design model outputs. The RF3 inner `MetricManager` passes `ground_truth_atom_array` (designed structure) and `predicted_atom_array` (refolded structure), while the outer `MetricManager` for the design model uses `predicted_atom_array` for its designed structure. In the YAML, `input_kwarg_names` lets you remap these for metrics whose `compute()` arguments use different names. If your `kwargs_to_compute_args` already uses the correct key names, you don't need `input_kwarg_names`.

---

## Step 3 — Run inference as usual

Once both files exist, `custom_metrics.yaml` is picked up automatically via the `optional custom_metrics` entry in `configs/metrics/all_metrics.yaml`. No extra flags needed:

```bash
python proteinfoundation/generate.py \
    inputs=design.json ckpt_path=./checkpoints ckpt_name=model.ckpt \
    num_replicates=50 metrics_tags=unconditional_refolding
```

Results appear in `per_example_metrics.csv` alongside the built-in metrics. Column names are prefixed with the top-level key from `custom_metrics.yaml`:
- Design metric: `residue_count.n_residues`, `residue_count.meets_min_length`
- Refolding metric: `rf3_custom.refolding_residue_delta.residue_count_delta`

---

## Quick reference

| | Design metric | Refolding metric |
|---|---|---|
| **Where** | Top-level in `custom_metrics.yaml` | Nested under `rf3_custom.metric_manager.metrics` |
| **Key inputs** | `predicted_atom_array` — bare `AtomArray` | `ground_truth_atom_array` (designed structure), `predicted_atom_array` (refolded structure) — both `AtomArrayStack` with depth 1 |
| **When to use** | Fast structural/sequence properties; no refolding needed | Comparing designed vs. independently refolded structure |
| **Slow?** | No | Yes (RF3 runs for each design) |

---

## Contributing back

If your metric is broadly useful, here's the minimal git workflow:

```bash
# 1. Create a branch before making any changes
git switch -c feat/my-metric-name

# 2. Add your class to a tracked metrics file, e.g. proteinfoundation/metrics/design_metrics.py

# 3. Add a YAML entry to an existing tracked config, e.g.:
#    configs/metrics/design_metrics.yaml

# 4. Stage all your changes and commit
git add .                                # stages everything new/modified in the working tree
git commit -m "feat: describe your metric here"   # replace with a real description
git push -u origin HEAD

# 5. Navigate to the foundry-dev GitHub page and open a pull request in the web interface
```

See [CLAUDE.md](../CLAUDE.md) for the full commit message format and PR checklist.
