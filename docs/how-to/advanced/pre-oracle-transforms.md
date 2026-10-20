# Pre-oracle transforms

`OffTargetSubstitutionTransform` swaps the designed chain's binding partner for each off-target you
list in `off_target_paths`, one refold per entry — a specificity assay wired up as `oracle_off_target`
in [`af3.yaml`](https://github.com/RosettaCommons/RFD4-Proteina/blob/main/configs/metrics/refolding_oracles/af3.yaml). Substitution is only the most
common use: a transform can rewrite the design however it likes before folding, so the fan out can ask
a design several questions in a single row of the metrics table. In enzyme design, for example, a
transform can swap the bound species for another state along the reaction coordinate and fold that
too.

Two things to get right when you use it:

- **Off-target files need matching annotations.** Concatenation only keeps what both sides share, so
  an off-target missing a non-default annotation value just reads as the default once substituted in —
  a registered lazy annotator or `StandardAnnotation` fills in the rest automatically. Put a
  *non-default* value in the off-target file itself if you need one. `OffTargetSubstitutionTransform`
  flags every substituted atom `is_loaded_target`, which is how you template the off-target through
  `template_query: "is_loaded_target & is_protein"` without handing it conditioning it never had on
  disk — a query selecting coordinate-conditioning masks, as `oracle_ppi` uses, will generally select
  nothing on its half of the complex.
- **Check any substitution that cuts into bonded chemistry.** It only safely joins protein chains that
  were never bonded to each other. Anything that breaks or forms a covalent bond, or touches a residue
  coordinating a metal, has to come out consistent in its bonds, atoms, charges, elements and
  annotations — a substituted array that wires a bond to the wrong atom, or silently drops one, still
  folds without complaint.

Refolds are cached by filename, as `{example_id}_{save_id}_{oracle_name}_*`, so re-running into the
same `output_dir` reuses whatever a previous run left under a given `save_id`. If you repoint
`off_target_paths` at a different structure, write to a fresh `output_dir` or delete the stale
refolds — otherwise the previous structure's numbers get reported under the new one's name.

To write your own transform, subclass `PreOracleTransform` and implement `forward`, which receives the
generated `AtomArray` and returns one `PreOracleOutput` per structure to fold. Both live in
[`pre_oracle_transform.py`](https://github.com/RosettaCommons/RFD4-Proteina/blob/main/src/rfproteina/metrics/oracles/pre_oracle_transform.py).
