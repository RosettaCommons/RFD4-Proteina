# Public Complexa eval jsons

The purpose of this directory is to contain json/yaml inference specifications for common benchmarks (PPI, AME)
where the public complexa samples utilized different chain naming conventions than used in this codebase.
In particular, public complexa always set the "binder" (designed protein chain) to be the last chain alphabetically.
To enable proper evaluation, we need json/yaml inputs whose contigs reflect this.

## PPI
For PPI, the only change is reordering the contigs such that the binder is last rather than first

## AME
For the AME benchmark, our codebase will preserve the chain_id of the ligand (chain A in the MCSA input pdb files),
matching the pattern of the public complexa samples. However, the contigs need to be adjusted such that the diffused
protein chain is chain B.