# Annotated-CIF validation requirements

Source of truth: `skills/rfd4-proteina-inputs/scripts/check_annotated_cif.py`, the
RFD4-Proteína condition definitions, and pipeline invariant checks.

## Registry initialization

```python
import rfproteina.conditions  # noqa: F401 -- registers RFD4-Proteína annotations
```

Import `rfproteina.conditions` before parsing annotated CIFs. The reader resolves
standard-annotation columns from the runtime registry; without registration,
RFD4-Proteína-specific conditions are unavailable. The provided build and validation
helpers perform this import.

## Missing atoms

- Express partial-atom and tip-atom conditioning with masks on a complete
  residue, not by deleting unconditioned atoms. Missing polymer atoms are a
  warning: confirm whether the source is genuinely unresolved and inspect any
  custom-CCD registration before using it for generation.
- A known CCD code with a different heavy-atom set is also a warning. Confirm
  whether it is intended to represent that CCD component or a deliberately
  authored custom component.
- Non-CCD component codes are supported. Validate their explicit atom inventory,
  bond graph, bond orders, aromaticity, and formal charges rather than requiring
  a CCD template match.
- Count heavy atoms only when comparing against a CCD definition.

## Current input constraints

- Preserve requested indexing and H-bond roles, then run the checked-out
  validation pipeline. If it reports an unsupported combination, surface that
  exact design conflict; do not force indexing on or discard one H-bond role.
- `nucleic_ss`, `ori_token`, and `atomwise_hbond` are not supported JSON/YAML
  compatibility fields. Use supported annotated-CIF conditions where available.
- Multiple variable contig segments constrained by one total length are not
  representable as independent expandable segments.
- Residue-ID gaps are evaluated within each index group, whether its residues are
  globally indexed or unindexed. Renumber only after confirming that the gap is
  unintended. `--fix_res_id_gap` applies only to numbering produced by the RFD3
  compatibility adapter; validation is not a general repair operation.

Never alter bonds, charges, atom identities, or conditions merely to make the
validator pass. Resolve the scientific intent first, rebuild, and rerun the full
validation helper.
