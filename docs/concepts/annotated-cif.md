# Annotated CIF

An annotated CIF is an ordinary mmCIF with extra columns. The usual columns hold the structure, so
chains, residues, atoms and coordinates. The extra ones hold what you want the model to do with each
atom: Hold this one still. Keep that residue's identity. Treat this one as a hotspot.

Both live in the same file, so nothing gets converted and nothing gets rebuilt. What you see is what
the model gets.

<div class="cif-figure">
<svg viewBox="0 0 766 228" xmlns="http://www.w3.org/2000/svg" class="wf aa-fig" role="img" aria-labelledby="aa-t aa-d">
  <title id="aa-t">Where each condition attaches</title>
  <desc id="aa-d">An AtomArray drawn as a strip, one cell per atom, with bars above grouping the atoms into residues and the residues into chains. Below the strip, two rows show where conditions are set. Coordinate conditioning is per atom, so it marks single cells: the CA of the first glycine and the OG of the serine. Sequence conditioning is per residue, so it marks a whole residue at a time, the serine. Everything unset defaults to NaN coordinates and UNK identity, which is what the model designs.</desc>
  <text x="68" y="34" class="aa-rowlab">chain</text>
  <rect x="80" y="22" width="488" height="16" rx="3" class="aa-group"/>
  <text x="324.0" y="34" class="aa-grouplab">A</text>
  <rect x="600" y="22" width="128" height="16" rx="3" class="aa-group"/>
  <text x="664.0" y="34" class="aa-grouplab">B</text>
  <text x="68" y="56" class="aa-rowlab">residue</text>
  <rect x="80" y="44" width="128" height="16" rx="3" class="aa-group"/>
  <text x="144.0" y="56" class="aa-grouplab">GLY</text>
  <rect x="212" y="44" width="160" height="16" rx="3" class="aa-group"/>
  <text x="292.0" y="56" class="aa-grouplab">ALA</text>
  <rect x="376" y="44" width="192" height="16" rx="3" class="aa-group"/>
  <text x="472.0" y="56" class="aa-grouplab">SER</text>
  <rect x="600" y="44" width="128" height="16" rx="3" class="aa-group"/>
  <text x="664.0" y="56" class="aa-grouplab">GLY</text>
  <text x="68" y="84" class="aa-rowlab">atom</text>
  <rect x="80" y="66" width="32" height="26" rx="3" class="aa-atom"/>
  <text x="96.0" y="84" class="aa-atomlab">N</text>
  <rect x="112" y="66" width="32" height="26" rx="3" class="aa-atom"/>
  <text x="128.0" y="84" class="aa-atomlab">CA</text>
  <rect x="144" y="66" width="32" height="26" rx="3" class="aa-atom"/>
  <text x="160.0" y="84" class="aa-atomlab">C</text>
  <rect x="176" y="66" width="32" height="26" rx="3" class="aa-atom"/>
  <text x="192.0" y="84" class="aa-atomlab">O</text>
  <rect x="212" y="66" width="32" height="26" rx="3" class="aa-atom"/>
  <text x="228.0" y="84" class="aa-atomlab">N</text>
  <rect x="244" y="66" width="32" height="26" rx="3" class="aa-atom"/>
  <text x="260.0" y="84" class="aa-atomlab">CA</text>
  <rect x="276" y="66" width="32" height="26" rx="3" class="aa-atom"/>
  <text x="292.0" y="84" class="aa-atomlab">C</text>
  <rect x="308" y="66" width="32" height="26" rx="3" class="aa-atom"/>
  <text x="324.0" y="84" class="aa-atomlab">O</text>
  <rect x="340" y="66" width="32" height="26" rx="3" class="aa-atom"/>
  <text x="356.0" y="84" class="aa-atomlab">CB</text>
  <rect x="376" y="66" width="32" height="26" rx="3" class="aa-atom"/>
  <text x="392.0" y="84" class="aa-atomlab">N</text>
  <rect x="408" y="66" width="32" height="26" rx="3" class="aa-atom"/>
  <text x="424.0" y="84" class="aa-atomlab">CA</text>
  <rect x="440" y="66" width="32" height="26" rx="3" class="aa-atom"/>
  <text x="456.0" y="84" class="aa-atomlab">C</text>
  <rect x="472" y="66" width="32" height="26" rx="3" class="aa-atom"/>
  <text x="488.0" y="84" class="aa-atomlab">O</text>
  <rect x="504" y="66" width="32" height="26" rx="3" class="aa-atom"/>
  <text x="520.0" y="84" class="aa-atomlab">CB</text>
  <rect x="536" y="66" width="32" height="26" rx="3" class="aa-atom"/>
  <text x="552.0" y="84" class="aa-atomlab">OG</text>
  <rect x="600" y="66" width="32" height="26" rx="3" class="aa-atom"/>
  <text x="616.0" y="84" class="aa-atomlab">N</text>
  <rect x="632" y="66" width="32" height="26" rx="3" class="aa-atom"/>
  <text x="648.0" y="84" class="aa-atomlab">CA</text>
  <rect x="664" y="66" width="32" height="26" rx="3" class="aa-atom"/>
  <text x="680.0" y="84" class="aa-atomlab">C</text>
  <rect x="696" y="66" width="32" height="26" rx="3" class="aa-atom"/>
  <text x="712.0" y="84" class="aa-atomlab">O</text>
  <text x="584.0" y="84" class="aa-ell">…</text>
  <text x="744.0" y="84" class="aa-ell">…</text>
  <path d="M80 108 H728" class="aa-rule"/>
  <text x="68" y="129" class="aa-condlab">C_CRD</text>
  <text x="68" y="142" class="aa-condsub">per atom</text>
  <rect x="80" y="120" width="32" height="20" rx="3" class="aa-off"/>
  <rect x="112" y="120" width="32" height="20" rx="3" class="aa-on"/>
  <rect x="144" y="120" width="32" height="20" rx="3" class="aa-off"/>
  <rect x="176" y="120" width="32" height="20" rx="3" class="aa-off"/>
  <rect x="212" y="120" width="32" height="20" rx="3" class="aa-off"/>
  <rect x="244" y="120" width="32" height="20" rx="3" class="aa-off"/>
  <rect x="276" y="120" width="32" height="20" rx="3" class="aa-off"/>
  <rect x="308" y="120" width="32" height="20" rx="3" class="aa-off"/>
  <rect x="340" y="120" width="32" height="20" rx="3" class="aa-off"/>
  <rect x="376" y="120" width="32" height="20" rx="3" class="aa-off"/>
  <rect x="408" y="120" width="32" height="20" rx="3" class="aa-off"/>
  <rect x="440" y="120" width="32" height="20" rx="3" class="aa-off"/>
  <rect x="472" y="120" width="32" height="20" rx="3" class="aa-off"/>
  <rect x="504" y="120" width="32" height="20" rx="3" class="aa-off"/>
  <rect x="536" y="120" width="32" height="20" rx="3" class="aa-on"/>
  <rect x="600" y="120" width="32" height="20" rx="3" class="aa-off"/>
  <rect x="632" y="120" width="32" height="20" rx="3" class="aa-on"/>
  <rect x="664" y="120" width="32" height="20" rx="3" class="aa-off"/>
  <rect x="696" y="120" width="32" height="20" rx="3" class="aa-off"/>
  <text x="68" y="165" class="aa-condlab">C_SEQ</text>
  <text x="68" y="178" class="aa-condsub">per residue</text>
  <rect x="80" y="156" width="128" height="20" rx="3" class="aa-off"/>
  <rect x="212" y="156" width="160" height="20" rx="3" class="aa-off"/>
  <rect x="376" y="156" width="192" height="20" rx="3" class="aa-on"/>
  <text x="472.0" y="170" class="aa-onlab">SER</text>
  <rect x="600" y="156" width="128" height="20" rx="3" class="aa-off"/>
  <path d="M80 196 H728" class="aa-rule aa-rule-soft"/>
  <text x="80" y="214" class="aa-note"><tspan class="aa-swatch-on">■</tspan> set, so the model is given it &#160;·&#160; <tspan class="aa-swatch-off">■</tspan> unset, so the model designs it</text>
</svg>
</div>

This example fixes two things and designs everything else. `C_CRD` pins the backbone CA of the
first glycine and the OG tip atom of the serine, so those two atoms land where you put them.
`C_SEQ` reveals the serine, so the model knows that residue has to *be* a serine rather than
whatever fits. Everything else is open: the alanine, the second chain, and every atom without a
marked cell. That is a motif of one anchored residue and one catalytic tip, scaffolded by the
model.

Notice the two rows work at different granularities, and that is the whole point. Coordinates are
per atom, which is what lets you pin one tip atom and leave the rest of its residue free. Identity
is per residue, all or nothing. Anything unset falls back to `NaN` coordinates and `UNK` identity,
and that is exactly what the model designs.

**Prefer to learn by doing?** The [Python API tutorial](../examples/python_api_tutorial.ipynb)
builds a pocket-conditioned input step by step and ends with an interactive view of exactly what
each annotation fixed.

## Why annotated CIFs

An annotated CIF is an ordinary structure file that also carries the conditions. They sit next to the
atoms they apply to, so the model reads one file and there is no second config to keep in sync. Use an
annotated CIF when the design depends on explicit component chemistry or conditions that should be
inspected and edited alongside the structure.

Three details deserve particular attention for non-canonicals, PTMs and small molecules.

- **Bonds.** Confirm bond order, aromaticity, and explicit covalent or metal
  coordination bonds. The model uses all of them as features.
- **Charges.** Confirm formal charge rather than treating an unknown value as
  neutral. The model embeds formal charge per atom.
- **PTMs and non-canonicals.** Preserve the atoms, bonds, charges, and component
  identity that define the modification.

An annotated CIF preserves all three, and the model reads CIFs natively.

> The [`rfd4-proteina-annotated-cif` skill](../how-to/input_prep/agent.md) authors directly from
> prose, structures, or an existing annotated CIF.

## It's still a CIF

The annotations are extra columns and categories layered onto a standard mmCIF; the coordinates and
topology underneath are ordinary mmCIF. That means any tool that reads mmCIF, PyMOL included, opens
an annotated CIF and sees a normal structure. It just will not show you the conditioning, because
that lives in categories those tools don't know to look at. To inspect or edit the annotations
themselves, parse the file with `ANNOTATED_CIF` through the Python API, or use the
[`rfd4-proteina-annotated-cif` skill](../how-to/input_prep/agent.md).

## Where to go next

| Task | Page |
| --- | --- |
| Build one from scratch, or combine conditions into a motif problem | [Preparing inputs](../how-to/input_prep/index.md) |
| Look up every condition and its CIF column | [Annotations reference](../reference/annotations.md) |
| Understand why conditions behave the way they do (active masks, coupled conditions, charges and bonds) | [Conditioning](conditioning.md) |
| Describe a design problem as data instead of Python | [YAML condition spec](../how-to/input_prep/yaml-json.md) |
| Have an agent author or edit one from prose or an existing structure | [`design2cif`](../how-to/input_prep/agent.md) |
| Work through a full example end to end | [Python API tutorial](../examples/python_api_tutorial.ipynb) |
