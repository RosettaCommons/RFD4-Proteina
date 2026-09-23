# DISCO Figure 2C inputs

These 77 annotated CIFs are the natural, non-natural, and multi-ligand targets
selected from Figure 2C of [*General Multimodal Protein Design Enables
DNA-Encoding of Chemistry*](https://arxiv.org/html/2604.05181v1).
The exact selection is `subset_to_keys` in
[`disco_figure_2c.yaml`](../../../../configs/datasets/val/small_molecule_binder/disco_figure_2c.yaml):
65 single-target entries and 12 multi-ligand entries, excluding the figure's
DNA/RNA panels. Both `phylloquinone` and `vitamin_K1` entries are retained;
this is a selection of benchmark entries, not deduplicated chemical identities.

Inputs come from our RFD4-Proteína-annotated `studio_179_annotated` dataset
(2026-09-18 snapshot), derived from the
[DISCO studio-179 inputs](https://github.com/DISCO-design/DISCO/tree/82b594f838eb61dd8c78ae3a403ccb8f00cf7abf/studio-179).
Zstandard level 3 compression is lossless: the 75 CIFs other than B12 and
rocuronium bromide decompress to the original bytes. Rocuronium's isolated Br
ion is a separate `BR` residue on chain C with the standard atom name `BR`;
the cation stays on chain A. Coordinates, all other atom names, conditions,
and collapsed 100–155-residue protein expansion ranges
are preserved. These are not the
paper's fixed-length 150/200/250 sampling inputs; the corresponding fixed-length
validation configs still use their separate external datasets. The smaller
`disco_subset` config is also unchanged.

## Chemistry caveats

Comparisons use the 2026-01-06 CCD snapshot. Atom names below are the input names.
B12 has corrected chemistry; rocuronium bromide has separate chains for its ions.
The other findings are deliberately retained.

| Input | Finding and disposition |
| --- | --- |
| Vitamin B12 | Corrected to CCD B12: 18 single bonds become double, 10 bond and 9 atom aromatic flags are restored, and Co formal charge becomes +2 in both charge tables. No coordinates, names, hydrogen fields, or other metadata change. |
| Uric acid (URC) | C1–N1 and C3–N2 are double instead of single; C1–C3 is single instead of double. One excess double bond remains; this is not an equivalent tautomer. |
| SAM, including SAM + Fe₂–S₂ | Methionine α-carbon C1 (CCD CA) is R rather than CCD SAM's S. This is also present in the upstream reference conformer. |
| Thyroxine | α-carbon C13 (CCD CA) is R rather than CCD T44's S (L-thyroxine), also in the upstream reference conformer. |
| 6PPD-q | Encodes oxygen-free 6PPD, not 6PPD-quinone; both quinone oxygens/carbonyls are absent in the upstream input too. |
| Rocuronium bromide | The 38-atom cation is on chain A and the isolated Br⁻ is a separate `BR` residue on chain C, giving them separate molecule/index groups. Chain B is the 100–155-residue protein segment. Charges and all 43 cation bonds are preserved; Br has no bonds. |
| Heme b, standalone and six combinations | Seven heme instances have 16 heavy-atom double bonds versus CCD HEM's 15, and lack its 10 aromatic bond flags. This is a metal-chemistry curation concern, not a recommendation to blindly copy CCD conventions. |
| Aromatic atom metadata (51 files) | 620 atom instances across 52 ligand components have atom-level aromatic flags set to N despite aromatic bond flags. RFD4-Proteína derives pairwise aromaticity from parsed bonds; these redundant atom flags do not by themselves indicate missing aromatic model features. |

B12 intentionally differs from the published input. In particular, its charge
correction can affect DISCO's charge features, so this is not an entirely
unmodified DISCO benchmark. Absence from the table does not certify chemistry;
some ligands lack an exact CCD match, and protonation/metal conventions can vary.

All 77 compressed files parse and pass the current design-validation pipeline.
Passing these checks does not resolve the chemistry caveats above.

Upstream licensing and attribution are preserved in [LICENSE](LICENSE) and
[NOTICE](NOTICE). RFD4-Proteína annotations, the B12 chemistry correction, and
rocuronium's ion-chain assignments are local modifications.
