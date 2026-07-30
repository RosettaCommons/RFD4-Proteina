# Skill Card

## Description

`rfd4-proteina-inputs` converts a design intent or a legacy RFD3-style JSON/YAML spec into a correct, validated **annotated CIF** for RFProteína, by interrogating the author's intent (length, coordinate vs sequence conditioning, indexing, ligand atoms, charges, bonds) and building with native biotite/atomworks primitives.

This skill is for research and development only.

## Owner

RFProteína maintainers — Baker Lab x NVIDIA GenAI collaboration.

## License/Terms of Use

See the RFProteína repository. Skill text follows the NVIDIA agent-skills documentation convention (CC-BY-4.0); helper scripts wrap `scripts/json2cif/*` under the repository's source license.

## Use Case

Anyone preparing an input for RFProteína — especially for non-canonicals, PTMs, small molecules, metals, or the newer conditioning types (RASA, H-bond, secondary structure, internal coordinates) that the legacy JSON path cannot express losslessly. Feeds `rfd4-proteina-generate` and `rfd4-proteina-search`.

## Deployment Geography for Use

Global, subject to the repository license.

### Requirements / Dependencies

Requires API Key or External Credential: No

Credential Type(s): None. `ALLOW_BIOTITE_CCD=True` is set by the wrapper scripts for CCD lookups (offline; no network credential).

Depends on: a completed `rfd4-proteina-setup` (venv + `atomworks` submodule). No GPU required to build/validate CIFs.

Do not include secrets in prompts/logs/output; use least-privilege credentials; rotate keys as appropriate.

## Known Risks and Mitigations

Risk: Silently transliterating a JSON produces an input that means something different from the author's intent (e.g. fixing a sidechain atom also fixes the residue's sequence; a range length is resampled per design).
Mitigation: The skill mandates interrogation ("this means XYZ, is that right?") before building, and documents the counter-intuitive defaults; `check_annotated_cif.sh` validates the result.

Risk: Parsing an annotated CIF without `import rfproteina.conditions` silently drops all conditions.
Mitigation: The provided scripts import it first; the skill documents the landmine for any custom code.

Risk: A truncated ligand or missing coordination bond misrepresents chemistry the model embeds.
Mitigation: The validator flags ligand heavy-atom truncation as an ERROR; metals are left unbonded and surfaced for the author to decide, never inferred.

## References

- `.claude/commands/design2cif.md`
- `docs/guides/annotated_cif_inputs.md`, `docs/guides/input.md`, `docs/guides/json_to_annotated_cif.md`
- `scripts/json2cif/build_cif.py`, `scripts/json2cif/check_annotated_cif.py`
- `docs/guides/tutorials/annotated_cif_tutorial.ipynb`

## Skill Output

Output type(s): a file (annotated `.cif`) plus a validation report.

Output format: mmCIF with AtomWorks condition columns/categories; console PASS/violation report.

Output parameters: one `.cif` per design problem, passing `check_annotated_cif.sh` with 0 errors.

Other properties: `build_cif.sh` writes only the `--output` file; `check_annotated_cif.sh` is read-only and never repairs.

## Skill Version

0.1.0 (initial authoring, 2026-07).

## Ethical Considerations

Defines the design problem (target, motif, ligand) for a de novo design model; the same dual-use considerations as generation apply. Do not encode inputs intended to design hazardous biomolecules. Report destructive preprocessing (dropped waters, altloc/MSE choices) so the author can confirm the input is what they intend.
