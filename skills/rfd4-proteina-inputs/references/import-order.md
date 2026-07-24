# Import-order landmine + known-bad corpus

Source of truth: `.claude/commands/json2cif.md`, `scripts/json2cif/*`.

## The import-order landmine

```python
import rfproteina.conditions  # noqa: F401  -- REQUIRED, registry side-effect only
```

The StandardAnnotation registry is populated at **import time**; the CIF reader
only recognizes columns whose names are registered. Parse an annotated CIF without
this import and **every rfproteina-specific condition is silently dropped**
(index-group, hotspot, RASA, hbond, ChainCentroid, CoordinateGroup) — because
deserialization failures are *warnings, not errors*
(`atomworks/io/utils/standard_annotations/serialization.py`).

`rfproteina/__init__.py` registers `utils.standard_annotations` but **not**
`conditions` — so `SA_SRC` survives while the conditions vanish. That looks like a
partial success, which is more dangerous than a crash: the condition is written,
lost on reload, and the pipeline dies downstream at `UnindexFlaggedTokens` with a
misleading "missing annotation" error. Both `build_cif.py` and
`check_annotated_cif.py` already do this import; do it first in any custom script.

## Missing atoms: ligands matter, sidechains don't

- **Polymer residue missing sidechain atoms -> normal.** That is what tip-atom
  scaffolding *is*, and what any `select_fixed_atoms` subset produces. A note,
  never a gate. **Count heavy atoms only** — the completion probe otherwise
  "restores" CCD hydrogens and reports phantom missing atoms.
- **Ligand missing heavy atoms vs its CCD entry -> a real defect (ERROR).** The
  model embeds each ligand's bond graph, aromaticity, and formal charge; a
  truncated ligand misrepresents all three.

## Known-bad in the corpus — surface, don't fix

- **An unindexed residue that is also ATOMIZED breaks the pipeline.**
  `UnindexFlaggedTokens` assigns one `token_id` per *residue*, correct only while a
  residue is a single token; an atomized residue is N tokens, so N−1 are lost.
  Reached three ways: covalently modified residue, non-canonical residue (e.g.
  rubisco's `KCX`), and `RandomAtomizeResidues` in training. Now guarded with a
  loud assert naming the residue as it appears in the **built** CIF (`A/KCX3`). The
  legacy JSON path trips it too; `enz_design` was deleted as a val set over this.
- **H-bond `BOTH` — legacy path silently drops donor roles.** A hydroxyl legitimately
  donates and accepts, but the legacy adapter is last-write-wins so `HBondEnum.BOTH`
  is unreachable there. Encoding faithfully yields `BOTH`, which trips
  `check_hbond_no_both_violations` — that is the bug becoming visible; do not drop a
  side to dodge it. Ask the author which role they want if only one can survive.
- **`nucleic_ss` is parsed, resolved, then never applied by the legacy path** (zero
  secondary structure delivered). An annotated CIF carries the condition on the
  array, so converting these **changes behaviour** — the benchmark starts doing what
  its tag always claimed.
- **`ori_token` / `atomwise_hbond` are dead knobs** (`extra_forbidden`). DNA binders
  run without ori tokens. `paper_dna.json`'s `*_dimer_ori` keys (two variable
  segments under a joint `length`) would be genuinely inexpressible, but no live
  config uses them, so the inexpressible set is empty for every working config.
- **The oracle places unindexed residues at `A362-A365`** (double-applied `max_id+1`),
  leaving a res_id gap that trips `_check_res_id_gaps`. Use `--fix_res_id_gap` to
  number unindexed residues contiguously.

**The oracle is not automatically right.** Where your build and the oracle
disagree, that is a question for the user, not an automatic bug in your build.
