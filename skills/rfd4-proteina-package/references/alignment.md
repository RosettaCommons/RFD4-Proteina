# Refold alignment

The generated design is both the oracle input and the fixed alignment reference.
Do not create another oracle-input copy or a second aligned-design copy.

For each refold, select the recorded chain or comma-separated chains of interest
in both structures. Match polymer anchor atoms by chain, residue ID, and atom name;
use protein `CA` and nucleic-acid `C4'` anchors, with common polymer-backbone atoms
as a fallback. At least three non-collinear corresponding atoms are required.

Superimpose the refold anchors onto the design anchors and apply that transform to
the entire refolded complex. Write the transformed refold into the package and
leave the archived source prediction unchanged.

Stop on a missing chain, ambiguous correspondence, insufficient anchors, or
non-finite coordinates. Do not guess chain mappings from filename order. The
package README states the chain-of-interest selection; no separate alignment JSON
is written.
