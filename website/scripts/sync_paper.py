"""Sync the website's byline and figure exports from reviewed manuscript sources.

Requires Pillow, pdftoppm and Inkscape. Run from the repository:
  python website/scripts/sync_paper.py --paper-root /path/to/RFD4-Proteina-paper \
      --work-dir /path/to/scratch/paper-site-sync

Captions and surrounding prose must be reviewed when source captions change.
"""

import argparse
import gzip
import hashlib
import html
import json
import re
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

from PIL import Image

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--paper-root", required=True, type=Path)
parser.add_argument("--work-dir", required=True, type=Path)
args = parser.parse_args()
paper = args.paper_root.resolve()
work = args.work_dir.resolve()
site = Path(__file__).resolve().parents[1]
work.mkdir(parents=True, exist_ok=True)
dpi = 300
sources = {
    "1": ("figure_1_concept", "figure_1_concept"),
    "2": ("figure_2_in_silico", "figure_2_in_silico"),
    "4": ("figure_4_specificity", "figure_4_wetlab_overview"),
    "5": ("figure_5_ptm_glycosylation", "figure_5_ptm_glycosylation"),
    "6": ("figure_6_enzymes", "figure_6_enzymes"),
}
# Points in the reviewed PDFs, top-left origin. Crops retain the paper artwork.
# None means the whole page. Bounds are re-reviewed when layouts change.
exports = {
    "figure1a-latent-callouts.png": ("1", None),
    "figure1d-trajectory.png": ("1", [16, 334, 494, 164]),
    "figure1f-search.png": ("1", [85, 593.4, 335, 61.6]),
    "fig2-in-silico.webp": ("2", None),
    "fig4-specificity-full.webp": ("4", None),
    "fig4a-calmodulin.webp": ("4", [0, 0, 216, 249]),
    "fig4b-pmhc.webp": ("4", [216, 0, 302.740157, 249]),
    "fig4cd-idr.webp": ("4", [0, 249, 518.740157, 211]),
    "fig4ef-trka.webp": ("4", [0, 460, 518.740157, 240.157480]),
    "fig5-ptm-glycosylation-full.webp": ("5", None),
    "fig5abc-glycan.webp": ("5", [0, 0, 518.740157, 165]),
    "fig5def-phospho.webp": ("5", [0, 165, 518.740157, 185.078740]),
    "fig6-enzymes-full.webp": ("6", None),
    "fig6a-d-pte.webp": ("6", [0, 0, 518.740173, 300]),
    "fig6h-n-ligase.webp": ("6", [0, 444, 518.740173, 426]),
}
structures = {
    "cam_resolved.cif.gz": "drafting/figure4_cam/second_hit.cif",
    "cam_selected.cif.gz": "drafting/figure4_cam/updated_20261016/selected_complex.cif",
    "h3pt11_hit09.cif.gz": "drafting/figure5_hbonds/updated_20261020/approved-style/H3T11_hit09.cif",
}
paths = ["authors.tex", "figures_raw/figure_1_concept.svg", *structures.values()]
for pdf, caption in sources.values():
    paths.extend([f"figures_raw/{pdf}.pdf", f"figures_tex/{caption}.tex"])
revision = subprocess.check_output(["git", "-C", str(paper), "rev-parse", "origin/main"], text=True).strip()
subprocess.run(["git", "-C", str(paper), "diff", "--quiet", revision, "--", *paths], check=True)
hashes = {p: hashlib.sha256((paper / p).read_bytes()).hexdigest() for p in paths}
tex = (paper / "authors.tex").read_text()
matches = list(re.finditer(r"\\author(\*)?\[([\d,]+)\]\{\\fnm\{([^}]+)\}\s*\\sur\{([^}]+)\}\}", tex))
authors = []
for i, match in enumerate(matches):
    tail = tex[match.end() : matches[i + 1].start() if i + 1 < len(matches) else tex.index(r"\affil")]
    authors.append(
        {
            "name": match[3] + " " + match[4],
            "affiliations": [int(n) for n in match[2].split(",")],
            "equal_first": r"\equalcont" in tail,
            "equal_second": r"\secondcont" in tail,
            "corresponding": bool(match[1]),
            "email": (re.search(r"\\email\{([^}]+)\}", tail)[1] if match[1] else None),
        }
    )
affiliations = [{"id": int(m[1]), "name": m[2]} for m in re.finditer(r"\\affil\[(\d+)\]\{\\orgname\{([^}]+)\}\}", tex)]
assert len(authors) == len({a["name"] for a in authors})
assert [a["id"] for a in affiliations] == list(range(1, len(affiliations) + 1))
assert sum(a["equal_first"] for a in authors) == 4
assert all(n in {v["id"] for v in affiliations} for a in authors for n in a["affiliations"])
rows = []
for first in [True, False]:
    names = []
    for author in authors:
        if author["equal_first"] != first:
            continue
        marks = ""
        if author["equal_first"]:
            marks += '<sup class="mark" title="Equal contribution">*</sup>'
        if author["equal_second"]:
            marks += '<sup class="mark" title="Equal contribution as co-second author">&#8225;</sup>'
        if author["corresponding"]:
            marks += '<sup class="mark" title="Corresponding author">&#9993;</sup>'
        aff = ",".join(map(str, author["affiliations"]))
        names.append(
            f'<span class="author-name">{html.escape(author["name"])}<sup class="aff">{aff}</sup>{marks}</span>'
        )
    if first:
        rows.append(
            '<div class="author-row-new author-row-first">\n'
            '    <span id="co-first-authors">\n    ' + ", ".join(names) + "\n    </span>\n"
            '    <button id="shuffle-authors" type="button" aria-controls="co-first-authors" hidden>'
            "Shuffle again</button>\n"
            "</div>"
        )
    else:
        rows.append('<div class="author-row-new author-row-rest">\n    ' + ", ".join(names) + ".\n</div>")
byline = "<!-- authors:start (generated from authors.tex by website/scripts/sync_paper.py) -->\n"
byline += "\n".join(rows)
byline += (
    '\n<details class="affiliations">\n'
    f"    <summary>Show {len(affiliations)} affiliations</summary>\n"
    '    <ol class="affiliation-list">\n'
)
byline += "\n".join(f'        <li value="{a["id"]}">{html.escape(a["name"])}</li>' for a in affiliations)
byline += '\n    </ol>\n</details>\n<p class="author-notes">\n'
byline += "    <span>* Equal contribution as co-first authors; order randomized on every refresh</span>\n"
byline += "    <span>&#8225; Equal contribution as co-second authors; alphabetical by surname</span>\n"
byline += (
    "    <span>&#9993; Correspondence: "
    + " and ".join(f'{html.escape(a["name"])} ({html.escape(a["email"])})' for a in authors if a["corresponding"])
    + "</span>\n</p>\n<!-- authors:end -->"
)
index = site / "index.html"
markup, count = re.subn(r"<!-- authors:start.*?<!-- authors:end -->", lambda _: byline, index.read_text(), flags=re.S)
assert count == 1
for filename in exports:
    path = f"assets/paper/{filename}"
    markup = re.sub(re.escape(path) + r'(?:\?v=[a-f0-9]+)?(?=")', path + f"?v={revision[:7]}", markup)
index.write_text(markup)
out = site / "assets/paper"
out.mkdir(exist_ok=True)
pages = {}
for key, (pdf, _) in sources.items():
    target = work / pdf
    subprocess.run(
        [
            "pdftoppm",
            "-f",
            "1",
            "-l",
            "1",
            "-r",
            str(dpi),
            "-png",
            "-singlefile",
            str(paper / f"figures_raw/{pdf}.pdf"),
            str(target),
        ],
        check=True,
        stderr=subprocess.PIPE,
    )
    pages[key] = Image.open(target.with_suffix(".png")).convert("RGB")
records = []
for filename, (key, box) in exports.items():
    image = pages[key]
    if filename == "figure1a-latent-callouts.png":
        # Preserve native callouts and projection, without adjacent artwork.
        svg = ET.parse(paper / "figures_raw/figure_1_concept.svg").getroot()
        children = list(svg)
        start = next(i for i, child in enumerate(children) if child.get("id") == "g35")
        end = next(i for i, child in enumerate(children) if child.get("id") == "panelA_GLC-selected-point-1")
        selected = children[start : end + 1]
        for child in children:
            if child not in selected and child.tag.rsplit("}", 1)[-1] not in ("defs", "style"):
                svg.remove(child)
        svg.set("viewBox", "124 16 342 211")
        svg.set("width", "342")
        svg.set("height", "211")
        native = work / "latent-callouts.svg"
        ET.ElementTree(svg).write(native, encoding="utf-8", xml_declaration=True)
        target = work / "latent-callouts.png"
        subprocess.run(
            [
                "inkscape",
                str(native),
                "--export-area-page",
                "--export-width=1600",
                "--export-background=white",
                "--export-background-opacity=1",
                f"--export-filename={target}",
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        image = Image.open(target).convert("RGB")
    if box:
        x, y, width, height = box
        bounds = tuple(round(n * dpi / 72) for n in [x, y, x + width, y + height])
        assert bounds[2] <= image.width and bounds[3] <= image.height, filename
        image = image.crop(bounds)
    target = out / filename
    if target.suffix == ".webp":
        image.save(target, lossless=True, method=6)
    else:
        image.save(target)
    records.append(
        {
            "path": f"assets/paper/{filename}",
            "source": "figures_raw/figure_1_concept.svg"
            if filename == "figure1a-latent-callouts.png"
            else f"figures_raw/{sources[key][0]}.pdf",
            "caption": f"figures_tex/{sources[key][1]}.tex",
            "crop_pdf_points": box,
            **(
                {"selected_svg_span": ["g35", "panelA_GLC-selected-point-1"], "svg_viewbox": [124, 16, 342, 211]}
                if filename == "figure1a-latent-callouts.png"
                else {}
            ),
            "pixels": list(image.size),
            "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
        }
    )
    print(filename, image.size, flush=True)
docs_asset = site.parent / "docs/_static/benchmark-figure.webp"
docs_asset.write_bytes((out / "fig2-in-silico.webp").read_bytes())
records.append(
    {
        "path": "../docs/_static/benchmark-figure.webp",
        "source": "figures_raw/figure_2_in_silico.pdf",
        "caption": "figures_tex/figure_2_in_silico.tex",
        "crop_pdf_points": None,
        "pixels": list(pages["2"].size),
        "sha256": hashlib.sha256(docs_asset.read_bytes()).hexdigest(),
    }
)
structure_records = []
for filename, source in structures.items():
    data = (paper / source).read_bytes()
    target = site / "assets/structures" / filename
    target.write_bytes(gzip.compress(data, mtime=0))
    assert gzip.decompress(target.read_bytes()) == data
    structure_records.append(
        {
            "path": f"assets/structures/{filename}",
            "source": source,
            "source_sha256": hashes[source],
            "export": "Unmodified source CIF, gzip compressed",
        }
    )
assert all(
    hashlib.sha256((paper / p).read_bytes()).hexdigest() == sha for p, sha in hashes.items()
), "Manuscript sources changed during export"
manifest = {
    "source_repository": "kierandidi/RFD4-Proteina-paper",
    "source_commit": revision,
    "source_hashes": hashes,
    "dpi": dpi,
    "authors": authors,
    "affiliations": affiliations,
    "figures": records,
    "structures": structure_records,
    "notes": [
        (
            "Reviewed manuscript PDFs are rendered and cropped; "
            "latent callouts use the corresponding native SVG elements. Scientific content is not redrawn."
        ),
        "Surrounding captions and numerical statements were reviewed separately against the recorded caption sources.",
    ],
}
(out / "manuscript-sync.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
print(f"Synced {len(authors)} authors, {len(affiliations)} affiliations and {len(records)} figure exports.", flush=True)
