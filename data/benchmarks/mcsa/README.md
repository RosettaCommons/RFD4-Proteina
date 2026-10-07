# MCSA benchmark inputs

Each of `mcsa_37`, `mcsa_37_fixed_ligand`, and `mcsa_37_no_ligand` contains the
same 37 of the original 41 targets. The validation subsets contain 15 targets, including
`M0907_1rbl` (Rubisco).

The unindexed covalent-adduct targets `M0054_1qfe` (LYS–DHS), `M0711_2esd`
(CYS–G3H), and `M0731_1mt5` (SER–MAY) are excluded. They require unindexed
PTM support and inputs that preserve the residue–ligand bonds. MAY's absent
fluorine is the leaving group in the SER–MAY adduct.

`M0024_1nzy` is excluded because its BCA ligand has an experimentally unresolved
phosphate group (`P3D`, `O7A`, `O8A`, `O9A`), with no observed coordinates.
This is a missing-coordinate case, not a residue–ligand covalent-adduct clash.

`M0110_1c0p` is retained: its covalent connections are between small molecules,
not between a protein residue and a modifier. Both ligand-containing variants
encode the original benchmark's FAD:N5–PER:O2 and DAL:O–PER:O1 single bonds.
FAD, DAL, and PER share an index group so both bonds reach the model's pair
features. The no-ligand variant contains none of these components.

Coordinates, atom inventories, charges, scaffold length, and coordinate,
sequence, and index conditioning are preserved. These inputs encode the
benchmark's authored chemistry; pipeline validation does not establish that
chemistry as a unique interpretation of the experimental structure.
