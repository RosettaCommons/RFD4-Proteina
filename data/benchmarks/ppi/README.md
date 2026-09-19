# PPI annotated inputs

Both sets contain 24 targets; `ppi_bb_only/` conditions N/CA/C/O only, excluding OXT and sidechains.
Neither set coordinate-conditions zero-occupancy atoms, including terminal OXT.
Coordinates, atom rows, sequence conditions, hotspots and lengths are preserved; missing coordinates are never fabricated.

`28_HER2_AAV` retains chain C residues 23–101 and 111–268 from [PDB 1N8Z](https://www.rcsb.org/structure/1N8Z).
In both input sets, target chain B uses local residue IDs 1–79 and 89–246:
the nine-residue gap is the unresolved sequence `NNTTPVTGA` (1N8Z C102–110).
The two flanking residues are not directly peptide-bonded; no missing loop atoms are supplied.

`rfd3_pdl1` uses experimentally observed coordinates from [PDB 5O45](https://www.rcsb.org/structure/5O45),
chain A residues 17–131, mapped to target chain B residues 1–115 in both input sets.
The 15 unobserved sidechain atoms retain their atom rows with NaN coordinates, zero occupancy,
and no coordinate conditioning. The documentation input `docs/examples/inputs/5o45_cropped.cif` uses the same geometry.

`rfd3_il7ra` uses observed sidechain coordinates from [PDB 3DI3](https://www.rcsb.org/structure/3DI3),
chain B residues 17–209, mapped to target B1–193. The 47 source-unobserved atoms remain present
with NaN coordinates, zero occupancy, and no coordinate conditioning in both input sets.

The following target residue ranges retain source sequence offsets in both input sets.
Gaps contain no supplied atoms and have no direct peptide bond across their endpoints.

| Target | Local target residue ranges | Source |
| --- | --- | --- |
| `24_SpCas9` | B1–79, B211–351 | [4ZT0](https://www.rcsb.org/structure/4ZT0) residues 98–176, 308–448; AF2 target coordinates are retained |
| `35_H1` | B1–45, B47–50, B76–80, B107–111, B259–322; C1–68, C80–170 | [5VLI](https://www.rcsb.org/structure/5VLI) chains A/B, using label sequence IDs |
| `37_IL17A` | B1–13, B25–115; C1–109 | [4HSA](https://www.rcsb.org/structure/4HSA) chains A/B; chain A residues 30–40 (`NRNTNTNPKRS`) are unresolved |

H1 label sequence IDs count the omitted author residue A49A (LYS); integer author numbering alone
would lose this one-residue gap. SpCas9's reference supports the sequence offsets, not replacement
of its AF2 coordinates with experimental coordinates. Atom inventory, conditioning and disulfide
annotations are independent of these residue-number gaps.
