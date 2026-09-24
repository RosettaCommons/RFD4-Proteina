# Legacy adapter edge cases

Source of truth: `skills/rfd4-proteina-inputs/scripts/build_from_rfd3.py`, the
current input-schema implementation, and legacy compatibility tests. Read this
only when the implicated fields or behaviors appear.

## Missing atoms

- Do not delete atoms to reproduce a `select_fixed_atoms` subset; express the
  selector as a condition mask on the complete built component.
- Missing polymer atoms and known CCD codes with a different heavy-atom set are
  review items. Count heavy atoms only, disclose preparation behavior, and do
  not automatically repair them.
- A deliberately authored non-CCD component is valid when its explicit atom
  inventory, bond graph, bond orders, aromaticity, and formal charges are
  complete and intentional.

## Known legacy incompatibilities

- **Unindexed atomized residues are invalid.** This includes noncanonical
  residues, covalently modified residues/PTMs, and anything explicitly atomized.
  The native invariant names the offending built-CIF residue. Surface the
  incompatibility; do not force it through.
- **H-bond `BOTH` was unreachable through the legacy adapter.** The old path was
  last-write-wins when one atom was selected as donor and acceptor. A faithful
  annotated CIF can encode `HBondEnum.BOTH`, but if the current validation/model
  path rejects it, ask which single role the author intends rather than silently
  dropping one.
- **`nucleic_ss` was parsed but not applied by the legacy path.** Applying it in
  the annotated CIF changes behavior: the migrated input finally does what the
  old tag claimed. Report that difference.
- **`ori_token` and `atomwise_hbond` are dead legacy knobs.** They are rejected
  as extras and do not provide the claimed behavior. Do not pretend to preserve
  them.
- **Legacy-oracle unindexed residue numbering can contain an artificial gap.**
  Use `--fix_res_id_gap` when the known double-offset behavior is the only source
  of the gap, and report the normalization.

## Oracle disagreements

The legacy oracle is evidence about prior behavior, not the specification of
correct RFD4 intent. When native conversion and oracle output differ:

1. Identify the exact field and structural consequence.
2. Determine whether the oracle exposed an intentional legacy behavior, a lossy
   behavior, or a bug.
3. Ask the author which behavior they intend in RFD4.
4. Never change the native build merely to make a byte/annotation comparison
   pass.
