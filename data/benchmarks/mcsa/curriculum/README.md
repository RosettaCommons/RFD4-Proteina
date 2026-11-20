# M-CSA motif curriculum

`tasks.json` defines seven fragmented tasks and seven contiguous controls, with
8, 12, 16, 20, 24, 28, and 32 coordinate-conditioned protein residues. These are
additional reconstruction exercises, separate from the existing 37-target AME
benchmark. No success rates or empirical difficulty ordering are asserted.

The experimental reference is [PDB 2DHN](https://www.rcsb.org/structure/2DHN),
7,8-dihydroneopterin aldolase with product PH2. Its relationship to M-CSA entry
664 and catalytic anchors Glu22 and Lys100 follows the bundled
`mcsa_37/M0664_2dhn.cif.zst` input. The reference snapshot was retrieved from
`https://files.rcsb.org/download/2DHN.cif`; its compressed SHA-256 is pinned in
`tasks.json`. No coordinates were synthesized. Numbering is **label chain A /
label_seq_id**, which agrees with author numbering for the selected protein.

The curriculum uses observed backbone atoms plus observed heavy atoms of the
catalytic anchors included in each task. Supporting residues are geometric
context, not additional M-CSA-annotated catalytic residues. The source includes
the crystallographic context, but the built tasks retain only the specified
protein fragments: **no PH2, water, additional subunits, or native scaffold** is
provided to generation. This makes an indexed, apo reconstruction track; it
does not reproduce the biological assembly or demonstrate enzymatic function.

| Level | Residues | Segments, all on chain A |
| --- | --- | --- |
| 1 | 8 | 18–25 |
| 2 | 12 | 20–25; 98–103 |
| 3 | 16 | 20–25; 70–74; 98–102 |
| 4 | 20 | 20–24; 70–74; 98–102; 112–116 |
| 5 | 24 | 15–18; 20–23; 65–68; 71–74; 98–101; 112–115 |
| 6 | 28 | 15–18; 20–23; 65–68; 71–74; 98–101; 104–107; 112–115 |
| 7 | 32 | 15–18; 20–23; 27–30; 65–68; 71–74; 98–101; 104–107; 112–115 |

Contiguous controls begin at residue 18 and extend to the requested length.
They all contain Glu22; the fragmented levels 2–7 additionally contain Lys100.
Thus the series intentionally changes both residue count and spatial
fragmentation, and the controls do not constitute complete catalytic sites.

From the repository root, in the configured environment:

```bash
python -m script_utils.mcsa_curriculum build \
  --manifest data/benchmarks/mcsa/curriculum/tasks.json \
  --out-dir runs/mcsa-curriculum --replicates 100 --length 200 --seed 0
```

This writes `inputs.json`, `run_manifest.json`, and `generate.sh` into a **new**
directory. It does not launch inference. See the
[guide](../../../../docs/how-to/mcsa-task.md) for preflight, execution, refolding,
reporting, and limitations. The equivalent installed command is `rfd4-mcsa`.
