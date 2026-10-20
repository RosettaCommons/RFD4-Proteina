# AE PDB temporal holdout

640 reconstruction examples: 128 each of protein–ligand, NCAA/PTM, glycan, DNA, and RNA. There is no
protein-only panel. Structures are annotated mmCIF files compressed with Zstandard level 3.

## Source and selection

Structures come from
`s3://YOUR_DATASET_STORE/train_datasets/streaming-latest/pdb/2026_10_04_tmol_plip_cath/`.
This processed release uses the same frozen 2026-01-06 PDB mirror as the previous holdout, with updated
parsing and chemistry. All examples have deposition dates on or after 2024-01-01.

The original temporal selection is retained wherever available. Four unavailable glycan examples are
replaced by 8z9i, 8zjf, 8zqp, and 8zqq; the unavailable 9bb0 ligand interface is replaced by 13rz.
The 8vnd DNA crop, whose initial DG residue has no resolved atoms in the new source, is replaced by 8rm6.
Latest-source residue identities are retained, including 9hh3 D65 CYS and 9kfx C16 C.

Interface selection uses inherited metadata: resolution below 9 Å, protein–ligand quality null or true
(never false), bond and ligand-identity filters true, at most 300 polymer PN units, a non-null interface
cluster, and an intermolecular interface. There is no ligand-count gate. Parent assemblies are capped at
20,000 atoms. Interface crops contain the two selected PN units.

NCAA/PTM and glycan selection additionally requires the respective holdout split, an in-vocabulary modified
residue or glycan, and passing structure, bond, and PoseBusters metadata flags. PTM crops retain their
selected residues; glycan crops contain the queried glycan and directly bonded attachment residues.
These quality flags describe the original selection metadata, not a fresh PoseBusters certification of
the updated conformers.

## Export and validation

Resolved latest-source atoms, coordinates, charges, bonds, and imputation labels are preserved; non-finite
coordinate placeholders are omitted. Source interaction-conditioning fields are excluded from the assets.
Sequence conditioning is supplied by the configured AE validation pipeline.

Every compressed file is reparsed and checked for matching atom identities, coordinates, charges,
elements, bonds, occupancy, B-factors, and retained extra annotations. All examples pass the complete AE
validation preprocessing pipeline with OXT both allowed and removed. Original residue numbering and
internal gaps are preserved.
