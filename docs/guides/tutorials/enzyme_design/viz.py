#!/usr/bin/env python3
"""Three-dimensional views of generated designs, switchable between samples.

Colour convention, held fixed across the notebook:

    light grey cartoon   the designed scaffold
    orange sticks        the substrate (ligand)
    green sticks+spheres coordinate-conditioned tip atoms
    grey thin sticks     the remaining atoms of each catalytic residue

The tip atoms are located by the ``mapped_to_coordinate_conditioned_region`` annotation the
generated structure carries; the full catalytic residues follow from the residues those atoms
belong to. Both are written into a single model so that bonds are perceived across the
tip / non-tip boundary and each side chain is drawn continuous.
"""

from __future__ import annotations

import io
import json
import uuid

import biotite.structure as struc
import biotite.structure.io.pdb as biopdb
import numpy as np
import pandas as pd
from atomworks.io import parse
from atomworks.io.utils.annotator import ensure_annotations
from IPython.display import HTML

from rfproteina.utils.parse_presets import ANNOTATED_CIF

VIEWER_JS = "https://3Dmol.org/build/3Dmol-min.js"


def _to_pdb(sub: struc.AtomArray) -> str:
    f = biopdb.PDBFile()
    f.set_structure(sub)
    buf = io.StringIO()
    f.write(buf)
    return buf.getvalue()


def motif_model(cif_path: str) -> dict:
    """Write the catalytic residues, whole, as one PDB model.

    Returns:
        ``{"pdb": str, "tips": [[resi, atom_name], ...], "n_rest": int}``. Keeping the residues
        in one model lets bonds be perceived across the tip / non-tip boundary, so each side
        chain is drawn continuous. The tip atoms are identified to the viewer by residue number
        and atom name, which are carried in the PDB itself and need no assumption about how
        serial numbers are assigned.
    """
    aa = parse(str(cif_path), config=ANNOTATED_CIF)["assemblies"]["1"][0]
    ensure_annotations(aa, "is_ligand")
    tip = np.asarray(aa.get_annotation("mapped_to_coordinate_conditioned_region")) & ~aa.is_ligand

    chain = np.asarray(aa.chain_iid).astype(str)
    keys = set(zip(chain[tip].tolist(), aa.res_id[tip].astype(int).tolist(), strict=False))
    same_res = np.array([(c, int(r)) in keys for c, r in zip(chain, aa.res_id, strict=False)], dtype=bool)
    motif = same_res & ~aa.is_ligand

    tips = [[int(r), str(n)] for r, n in zip(aa.res_id[tip], aa.atom_name[tip], strict=False)]
    return {"pdb": _to_pdb(aa[motif]), "tips": tips, "n_rest": int((motif & ~tip).sum())}


def switchable_view(samples: list[dict], width: int = 820, height: int = 560) -> HTML:
    """A viewer with one button per sample and a toggle for all side chains.

    Args:
        samples: entries of ``{"label", "cif", "motif", "n_tip", "note"}``, where ``motif`` and
            ``n_tip`` come from :func:`motif_model`.

    Returns:
        ``IPython.display.HTML`` holding the viewer and its controls; the result survives being
        saved into the notebook.
    """
    uid = "v" + uuid.uuid4().hex[:10]

    payload = ",\n".join(
        "{{label:{l},cif:{c},motif:{m},tips:{t},note:{n}}}".format(
            l=json.dumps(s["label"]),
            c=json.dumps(s["cif"]),
            m=json.dumps(s["motif"]),
            t=json.dumps(s["tips"]),
            n=json.dumps(s.get("note", "")),
        )
        for s in samples
    )

    buttons = "".join(
        f'<button id="{uid}_b{i}" onclick="{uid}_show({i})" '
        f'style="margin:2px 4px 2px 0;padding:4px 10px;border:1px solid #bbb;'
        f'border-radius:4px;background:#f4f4f4;cursor:pointer;font:12px monospace">'
        f'{s["label"]}</button>'
        for i, s in enumerate(samples)
    )

    return HTML(f"""
<div style="font:13px/1.5 system-ui,sans-serif">
  <div style="margin-bottom:4px">{buttons}</div>
  <div style="margin-bottom:6px">
    <button id="{uid}_sc" onclick="{uid}_toggle()"
      style="padding:4px 10px;border:1px solid #bbb;border-radius:4px;background:#f4f4f4;
             cursor:pointer;font:12px monospace">show all side chains</button>
  </div>
  <div id="{uid}_note" style="margin:4px 0 8px;color:#333;font:12px monospace"></div>
  <div id="{uid}" style="width:{width}px;height:{height}px;position:relative"></div>
  <div style="margin-top:6px;color:#555;font-size:12px">
    light grey cartoon = scaffold &nbsp;|&nbsp; orange = substrate &nbsp;|&nbsp;
    green = conditioned tip atoms &nbsp;|&nbsp; grey sticks = rest of the catalytic residues
  </div>
</div>
<script src="{VIEWER_JS}"></script>
<script>
(function() {{
  var data = [
{payload}
  ];
  var cur = 0, showAll = false, viewer = null;

  function draw() {{
    var s = data[cur];
    viewer.clear();
    viewer.addModel(s.cif, "cif");                       // model 0: the whole design
    viewer.setStyle({{}}, {{cartoon: {{color: "lightgrey"}}}});
    if (showAll) {{                        // every side chain, thin sticks, elements coloured
      viewer.addStyle({{model: 0, hetflag: false}},
                      {{stick: {{colorscheme: "grayCarbon", radius: 0.08}}}});
    }}
    viewer.setStyle({{hetflag: true}}, {{stick: {{colorscheme: "orangeCarbon", radius: 0.30}}}});

    viewer.addModel(s.motif, "pdb");                     // model 1: catalytic residues, whole
    viewer.setStyle({{model: 1}}, {{stick: {{colorscheme: "grayCarbon", radius: 0.12}}}});
    // Tip atoms are addressed by residue number and atom name rather than by serial, which the
    // viewer assigns independently of the order they were written in.
    for (var t = 0; t < s.tips.length; t++) {{
      var sel = {{model: 1, resi: s.tips[t][0], atom: s.tips[t][1]}};
      viewer.addStyle(sel, {{stick: {{colorscheme: "greenCarbon", radius: 0.24}}}});
      viewer.addStyle(sel, {{sphere: {{scale: 0.32, colorscheme: "greenCarbon"}}}});
    }}

    viewer.zoomTo({{hetflag: true}}); viewer.zoom(0.45); viewer.render();
    document.getElementById("{uid}_note").textContent = s.note;
    for (var i = 0; i < data.length; i++) {{
      var b = document.getElementById("{uid}_b" + i);
      if (b) {{ b.style.background = (i === cur) ? "#cfe3cf" : "#f4f4f4";
                b.style.fontWeight = (i === cur) ? "bold" : "normal"; }}
    }}
    var sc = document.getElementById("{uid}_sc");
    sc.style.background = showAll ? "#cfe3cf" : "#f4f4f4";
    sc.textContent = showAll ? "hide other side chains" : "show all side chains";
  }}

  window["{uid}_show"] = function(i) {{ cur = i; draw(); }};
  window["{uid}_toggle"] = function() {{ showAll = !showAll; draw(); }};

  function build() {{
    var el = document.getElementById("{uid}");
    if (!el || typeof $3Dmol === "undefined") {{ setTimeout(build, 120); return; }}
    viewer = $3Dmol.createViewer(el, {{backgroundColor: "white"}});
    draw();
  }}
  build();
}})();
</script>
""")


def samples_from_metrics(
    df: pd.DataFrame, n: int = 6, sort_by: str | None = None, note_cols: tuple[str, ...] = ()
) -> list[dict]:
    """Take the first ``n`` designs from a metrics table and prepare them for the viewer."""
    d = df.sort_values(sort_by) if sort_by and sort_by in df else df
    out = []
    for i, (_, row) in enumerate(d.head(n).iterrows()):
        path = row["designed_structure_path"]
        mm = motif_model(path)
        with open(path) as f:
            cif = f.read()
        bits = [
            f"{c.split('.')[0]}={row[c]:.3f}"
            for c in note_cols
            if c in row and isinstance(row[c], int | float) and row[c] == row[c]
        ]
        label = str(row.get("example_id", i)).split("__")[-1].replace("_job0", "")
        out.append({"label": label, "cif": cif, "motif": mm["pdb"], "tips": mm["tips"], "note": "  ".join(bits)})
    return out
