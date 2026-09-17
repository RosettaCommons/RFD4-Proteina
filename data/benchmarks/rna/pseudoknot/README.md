# Pseudoknot validation inputs

`W05` specifies a 90-residue RNA with 37 base pairs; `Q20` specifies a
240-residue RNA with 61 base pairs. Each input conditions every residue as
paired or explicitly unpaired and records the complete pairing topology,
including crossing pairs.

The conditions are serialized as `condition_na-ss-state_1_residue` and
`condition_basepair_2_residue`. The validation config enables
`na_ss_conditioning`; the resulting model features must contain both the
positive pairs and the explicit unpaired-state information. These conditions
specify secondary structure without fixing sequence or atomic coordinates.
