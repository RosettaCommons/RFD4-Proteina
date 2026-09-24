# Skill Card

## Description

`rfd4-proteina-inputs` adapts legacy RFD3-style JSON/YAML `InputSpecification` configs into validated RFD4-Proteína annotated CIFs. It interrogates ambiguous legacy semantics such as contigs, length ranges, fixed-atom selectors, sequence defaults, unindexing, ligand pose, charges, and bonds before conversion.

This skill is for research and development only.

## Owner

RFD4-Proteína maintainers — Baker Lab x NVIDIA GenAI collaboration.

## License/Terms of Use

See the RFD4-Proteína repository. Skill text follows the NVIDIA agent-skills documentation convention
(CC-BY-4.0); helper scripts are covered by the repository's source license.

## Use Case

Anyone migrating a legacy RFD3 JSON/YAML benchmark or reproducing/interpreting old `InputSpecification` behavior. General annotated-CIF creation and editing belongs to `rfd4-proteina-annotated-cif`. The converted output feeds `rfd4-proteina-generate` and `rfd4-proteina-search`.

## Deployment Geography for Use

Global, subject to the repository license.

### Requirements / Dependencies

Requires API Key or External Credential: No

Credential Type(s): None. `ALLOW_BIOTITE_CCD=True` is set by the wrapper scripts for CCD lookups (offline; no network credential).

Depends on: a completed `rfd4-proteina-setup` (venv + `atomworks` submodule)
and `rfd4-proteina-annotated-cif` for native CIF semantics and validation. No GPU
is required to build or validate CIFs.

Do not include secrets in prompts/logs/output; use least-privilege credentials; rotate keys as appropriate.

## Known Risks and Mitigations

Risk: Silently transliterating a JSON produces an input that means something different from the author's intent (e.g. fixing a sidechain atom also fixes the residue's sequence; a range length is resampled per design).
Mitigation: The skill mandates interrogation ("this means XYZ, is that right?") before building, and documents the counter-intuitive defaults; `check_annotated_cif.sh` validates the result.

Risk: The legacy oracle can encode lossy or buggy behavior and may disagree with the native RFD4 representation.
Mitigation: The skill treats oracle disagreement as an intent question, catalogs known incompatibilities, and does not rewrite native output merely to match the oracle.

Risk: An incomplete polymer, a component whose atoms differ from its CCD code, or a missing coordination bond can misrepresent chemistry the model embeds.
Mitigation: Partial conditioning uses masks without deleting atoms; the validator surfaces incomplete or mismatched components for intent review, supports explicit custom components, and never infers metal bonds.

## References

- `skills/rfd4-proteina-inputs/SKILL.md` (canonical instructions)
- `docs/workflow/inputs.md`, `docs/workflow/design2cif.md`
- `skills/rfd4-proteina-inputs/scripts/build_from_rfd3.py`
- `skills/rfd4-proteina-inputs/scripts/inspect_cif.py`
- `skills/rfd4-proteina-inputs/scripts/check_annotated_cif.py`
- Sibling skill: `rfd4-proteina-annotated-cif`

## Skill Output

Output type(s): a file (annotated `.cif`) plus a validation report.

Output format: mmCIF with AtomWorks condition columns/categories; console PASS/violation report.

Output parameters: one `.cif` per design problem, passing `check_annotated_cif.sh` with 0 errors.

Other properties: `inspect_cif.sh` and `check_annotated_cif.sh` are read-only;
`build_from_rfd3.sh` is used only when the user supplies an existing RFD3
JSON/YAML artifact and writes only the requested output.

## Skill Version

0.2.0 (narrowed to legacy RFD3 input adaptation, 2026-08).

## Ethical Considerations

Defines the design problem (target, motif, ligand) for a de novo design model; the same dual-use considerations as generation apply. Do not encode inputs intended to design hazardous biomolecules. Report destructive preprocessing (dropped waters, altloc/MSE choices) so the author can confirm the input is what they intend.
