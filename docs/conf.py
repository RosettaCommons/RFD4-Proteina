import os
import sys
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as package_version
from pathlib import Path
from typing import Any

import nbformat
from docutils import nodes
from sphinx.application import Sphinx

DOCS_DIR = Path(__file__).resolve().parent
REPO_ROOT = DOCS_DIR.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

# Autodoc uses local settings without a private runtime profile.
os.environ["CLUSTER"] = "local"
os.environ.setdefault("DATA_PATH", str(Path(__file__).resolve().parent / "_build" / "data"))
os.environ.pop("CCD_MIRROR_PATH", None)

# nbsphinx invokes pandoc as an executable. The docs extra supplies a bundled
# binary so local and CI builds do not depend on a system package.
try:
    import pypandoc

    pandoc_dir = str(Path(pypandoc.get_pandoc_path()).parent)
    os.environ["PATH"] = f"{pandoc_dir}{os.pathsep}{os.environ['PATH']}"
except ImportError:
    pass

project = "RFD4-Proteína"
author = "Baker Lab and NVIDIA GenAI"
copyright = "2026, Baker Lab and NVIDIA GenAI"

try:
    release = package_version("rfproteina")
except PackageNotFoundError:
    release = "0.0.2"
version = release

extensions = [
    "myst_parser",
    "nbsphinx",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx_togglebutton",
    "sphinx_reredirects",
]

# The 2026-08 rework reorganised the docs around the pipeline: one page per stage under
# `workflow/`, lookup tables under `reference/`. Everything below is an address the tutorial
# decks, Mattermost threads and people's bookmarks still point at, so it keeps resolving.
redirects = {
    # 2026-08: "reference" was a leftover bucket, not a category. Its pages now sit under the
    # workflow stage they belong to, and the two that duplicated a workflow page were merged in.
    "reference": "workflow.html",
    "reference/metrics": "../workflow/metrics.html",
    "reference/parameters": "../workflow/running.html#every-parameter",
    "reference/custom-metrics": "../workflow/custom-metrics.html",
    "reference/condition-spec": "../workflow/condition-spec.html",
    "reference/small-molecule-featurization": "../workflow/inputs.html",
    "workflow/conditions": "inputs.html",
    # guides -> workflow
    "guides/annotated_cif_inputs": "../workflow/inputs.html",
    "guides/inference": "../workflow/running.html",
    "guides/metrics": "../workflow/metrics.html",
    "guides/search": "../workflow/search.html",
    "guides/enzyme_design": "../examples/enzyme_design_tutorial.html",
    "guides/json_to_annotated_cif": "../workflow/design2cif.html",
    # guides -> reference
    "guides/parameters": "../workflow/running.html#every-parameter",
    "guides/small_molecule_featurization": "../workflow/inputs.html",
    # the small-molecule featurization page was retired; inputs covers what the model reads
    "workflow/small-molecule-featurization": "inputs.html",
    "condition_spec_yaml": "workflow/condition-spec.html",
    # /design2cif outgrew being a JSON converter, so the page moved off that name
    "workflow/from-json": "design2cif.html",
    # the enzyme design page was a condensed copy of its own tutorial; the tutorial is the page now
    "workflow/enzyme-design": "../examples/enzyme_design_tutorial.html",
    # guides -> top level
    "guides/custom_metrics": "../workflow/custom-metrics.html",
    # custom metrics moved off the top-level nav and under the metrics reference
    "custom-metrics": "workflow/custom-metrics.html",
    "guides/development": "../development.html",
    "guides": "workflow.html",
    # preserve the former input-guide URL
    "guides/input": "../workflow/inputs.html",
    # design-task stubs never had content — send readers to the runnable examples
    "guides/unconditional_generation": "../examples.html",
    "guides/motif_scaffolding": "../examples.html",
    "guides/protein_binder_design": "../examples.html",
    "guides/small_molecule_binder_design": "../examples.html",
    "guides/dna_binder_design": "../examples.html",
    # quickstart moved out of its single-page wrapper directory
    "tutorial/index": "../quickstart.html",
    "tutorial/quickstart": "../quickstart.html",
}

# Put a "show code" button on a notebook's incidental code -- setup, plotting, Slurm plumbing --
# so a tutorial reads as prose and results, while the cells that show how the model is actually
# driven stay open. Two ways to opt a cell in: everything under a `Setup` heading, or any cell
# tagged `collapse-code` (see `_mark_tagged_cells` below). `.nbinput` is the input half of an
# nbsphinx cell, so outputs -- tables, charts, viewers -- always stay visible.
togglebutton_selector = "#Setup .nbinput, .nbinput.collapse-code"
togglebutton_hint = "show code"
togglebutton_hint_hide = "hide code"


source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}
myst_enable_extensions = ["colon_fence", "deflist", "linkify"]
myst_heading_anchors = 6

exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    # Superpowers plans and specs are process docs checked in for traceability, not site pages.
    "superpowers/plans/**",
    "superpowers/specs/**",
]

autodoc_member_order = "bysource"
autodoc_typehints = "signature"
napoleon_google_docstring = True
napoleon_numpy_docstring = True

# Keep optional runtime backends from preventing API documentation builds.
autodoc_mock_imports = [
    "alphafold3",
    "cuequivariance",
    "cuequivariance_torch",
    "mace",
    "tmol",
]

html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_title = "RFD4-Proteína documentation"

html_theme_options = {
    "show_nav_level": 2,
    "collapse_navigation": False,
    "navigation_depth": -1,
    "globaltoc_collapse": False,
    "globaltoc_includehidden": True,
    "globaltoc_maxdepth": -1,
    "header_links_before_dropdown": 8,
    "navbar_start": ["navbar-logo"],
    "navbar_end": ["theme-switcher", "navbar-icon-links"],
    # Drop the theme's default "Show Source" link. Readers want the rendered page, and the source
    # is a click away on GitHub for anyone who wants it.
    "secondary_sidebar_items": ["page-toc"],
    "logo": {
        "image_light": "_static/wordmark.svg",
        "image_dark": "_static/wordmark.svg",
        "alt_text": "RFD4-Proteína",
    },
    "icon_links": [
        {
            "name": "Home",
            "url": "https://rosettacommons.github.io/RFD4-Proteina/",
            "icon": "fa-solid fa-house",
        },
        {
            "name": "Leaderboard",
            "url": "https://rosettacommons.github.io/RFD4-Proteina/leaderboard.html",
            "icon": "fa-solid fa-ranking-star",
        },
        {
            "name": "GitHub",
            "url": "https://github.com/RosettaCommons/RFD4-Proteina",
            "icon": "fa-brands fa-github",
        },
    ],
}


nbsphinx_execute = "never"

# Thumbnail for the `.. nbgallery::` tutorial card (notebooks emit interactive 3Dmol, not a
# static image, so point nbsphinx at one explicitly).
nbsphinx_thumbnails = {
    "examples/enzyme_design_tutorial": "_static/enzyme_tutorial_thumb.png",
    "examples/annotated_cif_tutorial": "_static/annotated_cif_thumb.png",
    "examples/ppi_specificity_tutorial": "_static/ppi_tutorial_thumb.png",
}

# Offer the runnable notebook for download at the bottom of every notebook page.
nbsphinx_epilog = r"""
----

:download:`Download this notebook <{{ env.docname.split("/")|last }}.ipynb>`
"""


def _convert_markdown_docstring_fences(
    _app: Sphinx,
    _what: str,
    _name: str,
    _obj: Any,
    _options: Any,
    lines: list[str],
) -> None:
    """Convert fenced code blocks used by package docstrings into reST."""
    converted = []
    in_fence = False
    for line in lines:
        if line.strip().startswith("```"):
            if in_fence:
                converted.append("")
                in_fence = False
            else:
                language = line.strip()[3:] or "text"
                converted.extend(["", f".. code-block:: {language}", ""])
                in_fence = True
        elif in_fence:
            converted.append(f"   {line}")
        else:
            converted.append(line)
    lines[:] = converted


def _mark_tagged_cells(app: Sphinx, doctree: nodes.document, docname: str) -> None:
    """Copy the ``collapse-code`` cell tag onto the rendered cell, as a CSS class.

    nbsphinx does not carry Jupyter cell tags through to the HTML, so a notebook has no way to
    ask for one particular code cell to be collapsed. Pair the doctree's ``nbinput`` containers
    back to the notebook's code cells by their source text and stamp the class
    :confval:`togglebutton_selector` looks for.
    """
    source = Path(str(app.env.doc2path(docname)))
    if source.suffix != ".ipynb":
        return
    tagged = {
        cell.source.strip()
        for cell in nbformat.read(source, as_version=4).cells
        if cell.cell_type == "code" and "collapse-code" in cell.metadata.get("tags", [])
    }
    for node in doctree.findall(nodes.container):
        if "nbinput" not in node["classes"]:
            continue
        # An `nbinput` holds two literal blocks: the `[n]:` prompt, then the cell source.
        *_, code = node.findall(nodes.literal_block)
        if code.astext().strip() in tagged:
            node["classes"].append("collapse-code")


def setup(app: Sphinx) -> None:
    app.connect("autodoc-process-docstring", _convert_markdown_docstring_fences)
    app.connect("doctree-resolved", _mark_tagged_cells)
