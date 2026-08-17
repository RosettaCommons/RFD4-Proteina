# Adding custom metrics

Write and register your own evaluation metrics without touching a single tracked file. Your code
stays local unless you decide to contribute it back.

> Looking for what already exists? The [metrics reference](metrics.md) lists every metric
> that ships enabled along with its tag, plus ready made classes you can wire in with nothing but a
> YAML config. What you want is often already there, or one config away.

:::{admonition} On a configured compute cluster?
:class: ipd-cluster
Drop the `python` and call `./src/rfproteina/generate.py` directly, so its shebang runs it inside
the apptainer image. Do this on an interactive GPU. Full form in
[Running the model](running.md#running-inference).
:::

Both `configs/metrics/custom_metrics/` and `src/rfproteina/metrics/custom_metrics/` are entirely git
ignored, so nothing you put in them ever gets tracked. `setup.sh` scaffolds an empty
`configs/metrics/custom_metrics/all_custom_metrics.yaml` for you on first run, and the Python side
directory appears the moment you drop your first metric module into it.

> **Is yours broadly useful?** If your metric measures something general, contribute it. Move the
> class to a tracked file under `src/rfproteina/metrics/`, add a YAML entry to a tracked config, and
> open a pull request. See
> [CLAUDE.md](https://github.com/RosettaCommons/RFD4-Proteina/blob/main/CLAUDE.md) for the
> contribution guidelines.

## Writing your own

If no existing metric covers what you need, write a `Metric` subclass. The rest of this page walks
through the three categories, the `kwargs_to_compute_args` routing, registering your metric in a git
ignored config so nothing gets committed by accident, and contributing it back if it turns out to be
generally useful.

### Three metric categories

Which category you land in depends on what inputs your metric needs.

**1. Design metrics operate on the raw generated structure.** They run inline in `validation_step`
and in `evaluate.py` Phase 1, on each generated design, before any folding happens.

Available kwargs, through `kwargs_to_compute_args`:

| Key | Type | Description |
|---|---|---|
| `predicted_atom_array` | `AtomArray` | The generated structure |
| `ground_truth_atom_array` | `AtomArray` | The input or reference structure |
| `extra_info` | `dict` | Metadata, holding `example_id`, `metrics_tags`, `output_base_dir` and `chain_of_interest` |

**2. Refolding metrics compare the design against the refold.** They run after an oracle folds the
design, and they receive the designed structure as ground truth and the oracle refolded structure as
the prediction. Both arrive as bare `AtomArray`, not `AtomArrayStack`.

| Key | Type | Description |
|---|---|---|
| `ground_truth_atom_array` | `AtomArray` | The designed structure |
| `predicted_atom_array` | `AtomArray` | The oracle refolded structure |
| `extra_info` | `dict` | Metadata, holding `example_id` and `metrics_tags` |

**3. Pass criteria threshold a metric into a pass or fail flag.** Use the built in
`PassCriterionMetric` with a `metric_key` and an optional `min` or `max`. No custom Python needed.
See [pass criteria config](#pass-criteria-config) below.

### How `kwargs_to_compute_args` works

`kwargs_to_compute_args` is a property on your metric class that routes kwargs into `compute()`.

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

A single element tuple like `("predicted_atom_array",)` reads `kwargs["predicted_atom_array"]`. A
two element tuple like `("extra_info", "example_id")` reads `kwargs["extra_info"]["example_id"]`.

## Step 1, write the Python class

Add a module under `src/rfproteina/metrics/custom_metrics/`. The whole directory is git ignored.

### Design metric example

```python
# src/rfproteina/metrics/custom_metrics/residue_count.py
from biotite.structure import AtomArray
from rfproteina.metrics.metric import Metric


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

Two things to get right. Always call `super().__init__(**kwargs)` so tag filtering like
`required_tags_any` gets picked up from the YAML config. And return a flat `dict[str, float]`,
because those keys become column names in `per_example_metrics.csv`, prefixed with the metric's
config name, as in `residue_count.n_residues`.

### Refolding metric example

This one receives bare `AtomArray` inputs, so there is nothing to unwrap.

```python
# src/rfproteina/metrics/custom_metrics/refolding_residue_count_delta.py
from biotite.structure import AtomArray
from biotite.structure.residues import get_residues
from rfproteina.metrics.metric import Metric


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

## Step 2, register it in the config file

Create `configs/metrics/custom_metrics/all_custom_metrics.yaml`. It is git ignored, and `setup.sh`
scaffolds it for you on first run. Start it with `# @package metrics` so Hydra merges your entries
straight into the overall metrics config under the right keys.

Each entry can carry `required_tags_any`, `required_tags_all` or `prohibited_tags`, which gate
whether the metric runs at all. Every run compares these fields against whatever list you pass as
`metrics_tags` on the command line.

- `required_tags_any: [a, b]` runs the metric if `metrics_tags` holds **at least one** of `a` or `b`.
- `required_tags_all: [a, b]` runs it only if `metrics_tags` holds **both**.
- `prohibited_tags: [c]` skips it if `metrics_tags` holds `c`, even when the required tags match.

**These fields are optional, but omitting them is a choice, not a no-op.** A metric with none of the
three has no gate and runs on **every** generated design, whatever `metrics_tags` you pass. That is
fine for something cheap and universally applicable, like counting residues. Add a gate whenever
your metric assumes something about the input that is not always true. A metric that reads hotspot
conditioning data should declare `required_tags_any: ["hotspot_conditioning"]`, so it only runs on
inputs that actually have hotspots instead of erroring, or worse, quietly returning junk.

Design tags and refolding tags are **separate, unrelated namespaces**, with no hierarchy and no
implication between them. A metric gated on `required_tags_any: ["design"]` will not run just
because you passed `unconditional_refolding`, and the reverse holds too. To run both families, pass
both tags: `metrics_tags='[design, unconditional_refolding]'`.

```yaml
# configs/metrics/custom_metrics/all_custom_metrics.yaml
# @package metrics

# Design metrics. Omit this section if you have none.
inline_metrics:
  residue_count:
    _target_: rfproteina.metrics.custom_metrics.residue_count.ResidueCountMetric
    min_length: 50

# Refolding metrics. Omit this section if you have none.
refolding_metrics:
  refolding_residue_delta:
    _target_: rfproteina.metrics.custom_metrics.refolding_residue_count_delta.RefoldingResidueCountDelta
    required_tags_any: [unconditional_refolding]

# Pass criteria. Omit this section if you have none.
pass_criteria:
  my_custom_plddt:
    _target_: rfproteina.metrics.oracles.pass_criterion_metric.PassCriterionMetric
    metric_thresholds:
      oracle.overall_plddt:
        min: 75.0
    required_tags_any: [unconditional_refolding]
```

Hydra dict merges each section into the built in config, covering `inline_metrics`,
`refolding_metrics`, `pass_criteria`, `refolding_oracles` and `pre_oracle_filters`. New entries get
added, and existing built in entries survive.

### Splitting across multiple files

If `all_custom_metrics.yaml` grows unwieldy, add sibling YAML files in the same
`configs/metrics/custom_metrics/` directory, each starting with `# @package metrics`, and pull them
in through an ordinary Hydra `defaults:` list.

```yaml
# configs/metrics/custom_metrics/all_custom_metrics.yaml
# @package metrics
defaults:
  - my_extra_metrics   # loads configs/metrics/custom_metrics/my_extra_metrics.yaml

inline_metrics:
  residue_count:
    _target_: rfproteina.metrics.custom_metrics.residue_count.ResidueCountMetric
```

```yaml
# configs/metrics/custom_metrics/my_extra_metrics.yaml
# @package metrics
refolding_metrics:
  refolding_residue_delta:
    _target_: rfproteina.metrics.custom_metrics.refolding_residue_count_delta.RefoldingResidueCountDelta
```

Both files dict merge into the same `metrics` package. Only `all_custom_metrics.yaml` needs a
reference from `configs/metrics/all_metrics.yaml`, and everything it pulls in resolves transitively.

The same goes on the Python side. `src/rfproteina/metrics/custom_metrics/` is a package, so add as
many modules as you like and reference each one as
`rfproteina.metrics.custom_metrics.<module_name>.<ClassName>`.

### Pass criteria config

`PassCriterionMetric` evaluates one or more values from the merged oracle and refolding metrics
dict, through `metric_thresholds`, a dict of `metric_key: {min: ..., max: ...}`. The output key is
`{config_name}.pass`. You need no custom Python class.

```yaml
pass_criteria:
  backbone_rmsd_tight:
    _target_: rfproteina.metrics.oracles.pass_criterion_metric.PassCriterionMetric
    metric_thresholds:
      refolded_backbone_rmsd.rmsd:   # dot-notation into merged metrics
        max: 1.0
    required_tags_any: [unconditional_refolding]
```

#### Thresholding a reduction over several keys

Listing two keys as separate entries **ANDs** them, so both must pass. To threshold a *reduction*
over several keys instead, give one spec a `min_of`, `max_of` or `mean_of` list. The outer key then
becomes a label and nothing more.

Take a three chain complex that only needs to engage **one** of two targets. Threshold the best,
meaning lowest, of the two per chain pair interface ipAEs. `max_of` would demand both are good, and
`mean_of` would ask them to hold on average.

```yaml
pass_criteria:
  either_target_engaged:
    _target_: rfproteina.metrics.oracles.pass_criterion_metric.PassCriterionMetric
    metric_thresholds:
      best_target_ipae:          # a label only, the output column is `either_target_engaged.pass`
        min_of:
          - oracle_ppi.min_ipae_A_B
          - oracle_ppi.min_ipae_A_C
        max: 1.5
    required_tags_any: [ppi_refolding]
```

Oracles emit one `min_ipae_<chain_i>_<chain_j>` key per chain pair, prefixed by the oracle's config
entry name, which is what makes per interface thresholds like this possible. Every listed key must
be present. A design missing a chain gets reported as indeterminate, with `all_metrics_present = 0`,
rather than passing on the interfaces that happened to survive. One spec carries at most one
reducer.

## Step 3, run it

Custom metrics get picked up automatically, because `all_custom_metrics.yaml` is a default. Most
often you want them computed **inline during generation**, so pass whichever tags your metrics
declared in `required_tags_any` or `required_tags_all` as `metrics_tags`.

Here are the common ones across this repo's built in metrics.

| Tag | Covers |
| --- | --- |
| `design` | Raw structure design metrics, so clash, bond length, SSE distribution, and any custom design metric |
| `unconditional_refolding` | AF3 refolding plus pLDDT and scRMSD, for unconditional generation |
| `ppi_refolding`, `motif_scaffold_refolding`, `sm_binder_refolding`, `ame_refolding` | Task specific AF3 refolding variants. See [Evaluating results](running.md#evaluating-results) |
| `hotspot_conditioning` | Hotspot obedience metrics, only meaningful when your input specifies hotspots |
| `3d_coordinate_conditioning` | Motif coordinate RMSD metrics |
| `rasa_conditioning` | RASA buried and exposed adherence metrics |

Tags are free form strings matched by set intersection, with no fixed enum to register against. So
if you invent a new tag for a custom metric, remember to pass that exact string in `metrics_tags`.
Get it wrong and the metric is silently skipped, with no error.

```bash
python src/rfproteina/generate.py \
    inputs=/path/to/design.cif \
    num_replicates=50 \
    metrics_tags='[unconditional_refolding]'
```

Generation results land in `inference_metrics/per_example_metrics.csv` under the run directory,
with column names prefixed by the config entry key. Offline evaluation uses
`evaluation_metrics/per_example_metrics.csv` under its `out_dir`.

- Design metric: `residue_count.n_residues`, `residue_count.meets_min_length`
- Refolding metric: `refolding_residue_delta.residue_count_delta`
- Pass criterion: `my_custom_plddt.pass`

To score structures you **already generated**, run the same metrics offline with `evaluate.py`,
passing `inputs=<dir> out_dir=<dir> metrics_tags='[...]'`. See the [Developer
Guide](../development.md).

## Quick reference

| | Design metric | Refolding metric | Pass criterion |
|---|---|---|---|
| **Config section** | `inline_metrics` | `refolding_metrics` | `pass_criteria` |
| **Key inputs** | `predicted_atom_array`, a bare `AtomArray` | `ground_truth_atom_array` and `predicted_atom_array`, both bare `AtomArray` | `extra_info["metrics"]`, the merged oracle and refolding dict |
| **When** | Before refolding | After the oracle folds the structure | After the refolding metrics |
| **Slow?** | No | No, the oracle already ran | No |

## Worked example: an existing metric, your own config

Often you need no new Python at all. An existing metric just needs **your** parameters. Say you are
doing enzyme design and want to check catalytic triad geometry. `AtomPairDistanceMetric`, listed
under [Ready-made but opt-in](metrics.md#ready-made-but-opt-in), already evaluates atom
pair upper bound distance constraints. You only have to tell it *which* atoms and *what* distances.

**Step 1, write the config.** Custom metrics live in the git ignored
`configs/metrics/custom_metrics/` directory, which `setup.sh` scaffolds on first run.

```yaml
# configs/metrics/custom_metrics/all_custom_metrics.yaml
# @package metrics

inline_metrics:
  catalytic_geometry:
    _target_: rfproteina.metrics.atom_pair_distance_metric.AtomPairDistanceMetric
    # No required_tags_* set, so this always runs, on every generated design.
    constraints:
      - name: triad_his_asp
        max_distance_angstrom: 3.2
        atom_a: "res_id == 31 and res_name == 'HIS' and atom_name == 'NE2'"
        atom_b: "res_id == 14 and res_name == 'ASP' and atom_name == 'OD1'"
      - name: oxyanion_cys
        max_distance_angstrom: 3.0
        atom_a: "chain_id == 'B' and res_name == 'lpr' and atom_name == 'O29'"
        atom_b: "chain_id == 'A' and res_name == 'CYS' and atom_name == 'N'"
```

Each `atom_a` and `atom_b` is an AtomWorks query string that must resolve to exactly one atom. They
go through [`resolve_mask`](../../src/rfproteina/utils/mask_spec.py), the one shared string to mask
evaluator this codebase uses everywhere. Give a list of query strings instead of a single one and
the metric checks the minimum distance across several alternative atoms. Alternatives that resolve
to zero or several atoms get skipped individually, so one inapplicable option does not discard the
rest of the list. For the full query and path selection grammar, see AtomWorks' [atom selection
syntax](https://rosettacommons.github.io/atomworks/latest/io/utils/selection_syntax.html)
guide.

Leaving out `required_tags_any` and `required_tags_all` means the metric has no gate and runs on
every design. That fits here, because a design either has a catalytic triad or it does not, and the
metric reports `inf` and a violation when the named atoms are missing rather than erroring.

**Step 2, run it.** Hydra dict merges
`configs/metrics/custom_metrics/all_custom_metrics.yaml` into the built in metrics config
automatically, since it is already a default in
[`all_metrics.yaml`](../../configs/metrics/all_metrics.yaml), listed last so it merges on top. New keys
get **added**, and a key matching an existing built in name gets **deep merged over it**. Redeclare
`clash_metrics:` with just a new `threshold:` and you override that one field while inheriting
everything else. So the normal run command picks it up with no extra flags.

```bash
python src/rfproteina/generate.py \
    inputs=enzyme_design.cif num_replicates=50
```

Results land in `per_example_metrics.csv` as `catalytic_geometry.triad_his_asp_distance`,
`catalytic_geometry.triad_his_asp_satisfied`, `catalytic_geometry.n_satisfied`,
`catalytic_geometry.all_satisfied` and more. The [`AtomPairDistanceMetric`
docstring](../../src/rfproteina/metrics/atom_pair_distance_metric.py) has the full output set, covering
per constraint distance, violation, satisfied and soft score, plus the aggregates
`soft_min_score`, `total_violation` and `worst_violation`.

## Contributing back

```bash
# 1. Create a branch before making any changes
git switch -c feat/my-metric-name

# 2. Add your class to a tracked metrics file under src/rfproteina/metrics/

# 3. Add a YAML entry to an existing tracked config

# 4. Stage all your changes and commit
git add .
git commit -m "feat: describe your metric here"
git push -u origin HEAD

# 5. Open a pull request on the foundry-dev GitHub page
```

See [CLAUDE.md](https://github.com/RosettaCommons/RFD4-Proteina/blob/main/CLAUDE.md) for the
full commit message format and the PR checklist.
