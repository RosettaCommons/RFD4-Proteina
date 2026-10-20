# DISCO Figure 2C inputs

The Figure 2C preset selects 77 annotated CIFs: natural, non-natural, and multi-ligand targets
selected from Figure 2C of [*General Multimodal Protein Design Enables
DNA-Encoding of Chemistry*](https://arxiv.org/html/2604.05181v1).
The exact selection is `subset_to_keys` in
[`disco_figure_2c.yaml`](../../../../configs/datasets/val/small_molecule_binder/disco_figure_2c.yaml):
65 single-target entries and 12 multi-ligand entries, excluding the figure's
DNA/RNA panels. Both `phylloquinone` and `vitamin_K1` entries are retained;
this is a selection of benchmark entries, not deduplicated chemical identities.

Inputs come from our RFD4-Proteina-annotated `studio_179_annotated` dataset
(2026-09-18 snapshot), derived from the
[DISCO studio-179 inputs](https://github.com/DISCO-design/DISCO/tree/82b594f838eb61dd8c78ae3a403ccb8f00cf7abf/studio-179).
Zstandard level 3 compression is lossless: the 72 CIFs other than B12,
rocuronium bromide, and the three NAD-containing inputs decompress to the
original bytes. Rocuronium's isolated Br ion is a separate `BR` residue on
chain C with the standard atom name `BR`; the cation stays on chain A. The NAD
inputs replace only the 44 NAD heavy-atom coordinates with the RCSB CCD ideal
conformer. All atom names, conditions, and collapsed 100–155-residue protein
expansion ranges are preserved. These are not the
paper's fixed-length 150/200/250 sampling inputs. Chlorophyll a and staurosporine
are included here as lossless copies of the same source inputs for the existing
12-target `disco_subset`; they are not added to the Figure 2C preset's selection.

## Chemistry caveats

Comparisons use the 2026-01-06 CCD snapshot. Atom names below are the input names.
B12 has corrected chemistry, the NAD inputs use the CCD ideal conformer, and
rocuronium bromide has separate chains for its ions. The other findings are
deliberately retained.

| Input | Finding and disposition |
| --- | --- |
| Vitamin B12 | Corrected to CCD B12: 18 single bonds become double, 10 bond and 9 atom aromatic flags are restored, and Co formal charge becomes +2 in both charge tables. No coordinates, names, hydrogen fields, or other metadata change. |
| NAD, including FMN + NAD and heme b + NAD | The 44 NAD heavy-atom coordinates come from the [RCSB CCD ideal NAD conformer](https://files.rcsb.org/ligands/download/NAD.cif), SHA-256 `f202c206d06c8aafa01516df17e91d21a6c60ed1a02f815e60bca00d12a62439`, graph-mapped and rigidly aligned into the source frame. Source names, bonds, component tables, formal charge (-1), conditions, and all ten coordinate-inferred stereocenter assignments are preserved; the CCD charge (0) is not transferred, and no relaxation is applied. This removes the source's nonbonded 1.470 Å C5–O6 contact. |
| Uric acid (URC) | C1–N1 and C3–N2 are double instead of single; C1–C3 is single instead of double. One excess double bond remains; this is not an equivalent tautomer. |
| SAM, including SAM + Fe₂–S₂ | Methionine α-carbon C1 (CCD CA) is R rather than CCD SAM's S. This is also present in the upstream reference conformer. |
| Thyroxine | α-carbon C13 (CCD CA) is R rather than CCD T44's S (L-thyroxine), also in the upstream reference conformer. |
| 6PPD-q | Encodes oxygen-free 6PPD, not 6PPD-quinone; both quinone oxygens/carbonyls are absent in the upstream input too. |
| Rocuronium bromide | The 38-atom cation is on chain A and the isolated Br⁻ is a separate `BR` residue on chain C, giving them separate molecule/index groups. Chain B is the 100–155-residue protein segment. Charges and all 43 cation bonds are preserved; Br has no bonds. |
| Heme b, standalone and six combinations | Seven heme instances have 16 heavy-atom double bonds versus CCD HEM's 15, and lack its 10 aromatic bond flags. This is a metal-chemistry curation concern, not a recommendation to blindly copy CCD conventions. |
| Aromatic atom metadata (51 files) | 620 atom instances across 52 ligand components have atom-level aromatic flags set to N despite aromatic bond flags. RFD4-Proteina derives pairwise aromaticity from parsed bonds; these redundant atom flags do not by themselves indicate missing aromatic model features. |

B12 and the three NAD inputs intentionally differ from the published inputs.
B12's charge correction can affect DISCO's charge features. The existing
Figure 2C config does not coordinate-condition ligand poses, so the NAD coordinate
correction does not change what those prior unfixed runs supplied to the model;
it makes the canonical inputs suitable for consumers that do fix the ligand.
Absence from the table does not certify chemistry; some ligands lack an exact CCD
match, and protonation/metal conventions can vary.

All 77 compressed files parse and pass the current design-validation pipeline.
Passing these checks does not resolve the chemistry caveats above.

Upstream licensing and attribution are preserved in [LICENSE](LICENSE) and
[NOTICE](NOTICE). RFD4-Proteina annotations, the B12 chemistry correction, and
rocuronium's ion-chain assignments, and the NAD coordinate correction are local
modifications.
