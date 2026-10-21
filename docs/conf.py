import os
import sys
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as package_version
from pathlib import Path
from typing import Any

import nbformat
import tomllib
from docutils import nodes
from sphinx.application import Sphinx
from sphinx.builders.html._assets import _CascadingStyleSheet

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

project = "RFD4-Proteina"
author = "Baker Lab and NVIDIA GenAIR"
copyright = "2026, Baker Lab and NVIDIA GenAIR"

try:
    release = package_version("rfd4-proteina")
except PackageNotFoundError:
    release = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())["project"]["version"]
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
    # The metrics catalog returned to "reference/metrics" later, so that one redirect is gone.
    "reference": "concepts/workflow.html",
    "reference/parameters": "../how-to/run-the-model.html#every-parameter",
    "reference/custom-metrics": "../how-to/advanced/custom-metrics.html",
    "reference/condition-spec": "../how-to/input_prep/yaml-json.html",
    "reference/small-molecule-featurization": "../how-to/input_prep/index.html",
    "workflow/conditions": "../how-to/input_prep/index.html",
    "workflow/metrics": "../how-to/evaluate-with-metrics.html",
    # guides -> workflow
    "guides/annotated_cif_inputs": "../how-to/input_prep/index.html",
    "guides/inference": "../how-to/run-the-model.html",
    "guides/metrics": "../how-to/evaluate-with-metrics.html",
    "guides/search": "../concepts/search.html",
    "guides/enzyme_design": "../examples/enzyme_design_tutorial.html",
    "guides/json_to_annotated_cif": "../how-to/input_prep/agent.html",
    # guides -> reference
    "guides/parameters": "../how-to/run-the-model.html#every-parameter",
    "guides/small_molecule_featurization": "../how-to/input_prep/index.html",
    # the small-molecule featurization page was retired; inputs covers what the model reads
    "workflow/small-molecule-featurization": "../how-to/input_prep/index.html",
    "condition_spec_yaml": "how-to/input_prep/yaml-json.html",
    # condition-spec.md's content was split into the how-to guide, the FAQ, and the field reference
    "workflow/condition-spec": "../how-to/input_prep/yaml-json.html",
    # /design2cif outgrew being a JSON converter, so the page moved off that name
    "workflow/from-json": "../how-to/input_prep/agent.html",
    # design2cif moved under input_prep alongside the other input-authoring routes
    "workflow/design2cif": "../how-to/input_prep/agent.html",
    # the enzyme design page was a condensed copy of its own tutorial; the tutorial is the page now
    "workflow/enzyme-design": "../examples/enzyme_design_tutorial.html",
    # guides -> top level
    "guides/custom_metrics": "../how-to/advanced/custom-metrics.html",
    # custom metrics moved off the top-level nav and under the metrics reference
    "custom-metrics": "how-to/advanced/custom-metrics.html",
    "guides/development": "../contributing/developer-guide.html",
    "development": "contributing/developer-guide.html",
    "guides": "concepts/workflow.html",
    # preserve the former input-guide URL
    "guides/input": "../how-to/input_prep/index.html",
    # inputs moved out of the workflow stage into its own input-preparation section
    "workflow/inputs": "../how-to/input_prep/index.html",
    # inputs became the index page of its own input-preparation section
    "how-to/inputs": "input_prep/index.html",
    # metric guidance moved into the how-to advanced subsection
    "how-to/metric-guidance": "advanced/metric-guidance.html",
    # custom metrics moved under the how-to advanced subsection alongside metric guidance
    "workflow/custom-metrics": "../how-to/advanced/custom-metrics.html",
    # design-task stubs never had content — send readers to the runnable examples
    "guides/unconditional_generation": "../examples/index.html",
    "guides/motif_scaffolding": "../examples/index.html",
    "guides/protein_binder_design": "../examples/index.html",
    "guides/small_molecule_binder_design": "../examples/index.html",
    "guides/dna_binder_design": "../examples/index.html",
    # quickstart moved out of its single-page wrapper directory
    "tutorial/index": "../quickstart.html",
    "tutorial/quickstart": "../quickstart.html",
    # the workflow overview moved under concepts/ alongside the rest of the conceptual docs
    "workflow": "concepts/workflow.html",
    # metric guidance, partial diffusion, and search moved under concepts/ with the workflow overview
    "workflow/metric-guidance": "../concepts/metric-guidance.html",
    "workflow/partial-diffusion": "../concepts/partial-diffusion.html",
    "workflow/search": "../concepts/search.html",
    # pages moved by the restructuring, including those added to the old layout since
    "workflow/running": "../how-to/run-the-model.html",
    "workflow/backends": "../how-to/advanced/backends.html",
    "workflow/outputs": "../reference/output-format.html",
    "contributor_guide": "contributing/index.html",
    "glossary": "reference/glossary.html",
    "examples/annotated_cif_tutorial": "python_api_tutorial.html",
    "api_reference": "reference/index.html",
    "api/conditions": "../reference/api/conditions.html",
    "api/inference_utils": "../reference/api/inference_utils.html",
    "api/metrics": "../reference/api/metrics.html",
    "api/rewards": "../reference/api/rewards.html",
    "api/search": "../reference/api/search.html",
    "api/transforms": "../reference/api/transforms.html",
    # the examples landing page moved alongside its tutorial notebooks
    "examples": "examples/index.html",
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
    # Release/packaging steps are for maintainers, not RFD4-Proteina users; keep it out of the built site.
    "contributing/packaging.md",
]

autodoc_member_order = "bysource"
viewcode_follow_imported_members = False
autodoc_typehints = "signature"
napoleon_google_docstring = True
napoleon_numpy_docstring = True

# Keep optional runtime backends from preventing API documentation builds.
autodoc_mock_imports = [
    "alphafold3",
    "cuequivariance",
    "cuequivariance_torch",
    "tmol",
]

# Set when the docs are published beside the project page (docs.yaml), so the header can link to it.
_SITE_ROOT = os.environ.get("RFD4_SITE_ROOT", "")

html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]
# _templates/layout.html marks the <body> of the latent explorer, which scopes its stylesheets.
templates_path = ["_templates"]
html_css_files = ["custom.css", "architecture/explorer.css"]
# Sizes and themes the neural circuit diagrams framed in the architecture explorer.
html_js_files = [
    ("architecture/ncd-frames.js", {"defer": "defer"}),
    ("https://scripts.simpleanalyticscdn.com/latest.js", {"async": "async", "defer": "defer"}),
]
# The architecture explorer and the latent explorer give the full width to their diagrams and figures, so
# neither shows the section navigation.
LATENT_EXPLORER_PAGES = ("architecture/expanded-chemistry",)
# The stylesheets of the latent explorer, which _architecture_page_assets adds to that page alone; arch.css (the
# story pages' fonts, type and shared components) comes first.
LATENT_EXPLORER_CSS_FILES = ("arch.css", "viz/viz.css")
html_sidebars = {
    "architecture/index": [],
    **{page: [] for page in LATENT_EXPLORER_PAGES},
}
# nbsphinx adds require.js from a CDN to every page; the explorer uses none of it and loads nothing remote.
NO_REMOTE_SCRIPT_PAGES = ("architecture/index",)
html_context = {"site_root": _SITE_ROOT}
if _SITE_ROOT:
    html_css_files = [*globals().get("html_css_files", []), "site-nav.css"]
    html_js_files = [*html_js_files, ("site-nav.js", {"defer": "defer"})]
html_favicon = "_static/favicon-4p.svg"
html_title = "RFD4-Proteina documentation"
html_baseurl = "https://rfd4-proteina.org/latest/"

html_theme_options = {
    "show_nav_level": 2,
    "collapse_navigation": False,
    "navigation_depth": -1,
    "globaltoc_collapse": False,
    "globaltoc_includehidden": True,
    "globaltoc_maxdepth": -1,
    "header_links_before_dropdown": 8,
    # Beside the project page the shared tab line (_templates/site-nav.html) carries the wordmark and the
    # color mode; the documentation header keeps its section links, search and icons.
    "navbar_start": [] if _SITE_ROOT else ["navbar-logo"],
    "navbar_end": ["navbar-icon-links"] if _SITE_ROOT else ["theme-switcher", "navbar-icon-links"],
    # Drop the theme's default "Show Source" link. Readers want the rendered page, and the source
    # is a click away on GitHub for anyone who wants it.
    "secondary_sidebar_items": ["page-toc"],
    "logo": {
        "image_light": "_static/wordmark.webp",
        "image_dark": "_static/wordmark-dark.webp",
        "alt_text": "RFD4-Proteina",
    },
    "icon_links": [
        *(
            []
            if _SITE_ROOT
            else [
                {
                    "name": "Home",
                    "url": "https://rfd4-proteina.org/",
                    "icon": "fa-solid fa-house",
                }
            ]
        ),
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
    "examples/python_api_tutorial": "_static/annotated_cif_thumb.png",
    "examples/ppi_specificity_tutorial": "_static/ppi_tutorial_thumb.png",
    "examples/antibody_design_tutorial": "_static/antibody_tutorial_thumb.png",
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


def _architecture_page_assets(
    app: Sphinx, pagename: str, _templatename: str, context: dict[str, Any], _doctree: nodes.document | None
) -> None:
    """Give the latent explorer its stylesheets, and drop remote scripts.

    A page's stylesheets load after html_css_files and on no other page.
    nbsphinx adds require.js from a CDN to every page; nothing on these pages uses it, so it is dropped.
    """
    if pagename in LATENT_EXPLORER_PAGES:
        stylesheets = LATENT_EXPLORER_CSS_FILES
    elif pagename in NO_REMOTE_SCRIPT_PAGES:
        stylesheets = ()
    else:
        return
    # Appended to this page's own context so no other page sees the stylesheets.
    context["css_files"] = [
        *context.get("css_files", []),
        *(_CascadingStyleSheet(f"_static/{name}", priority=800) for name in stylesheets),
    ]
    remote = ("http://", "https://", "//")
    scripts = context.get("script_files", [])
    context["script_files"] = [s for s in scripts if not str(getattr(s, "filename", s)).startswith(remote)]


def setup(app: Sphinx) -> None:
    app.connect("autodoc-process-docstring", _convert_markdown_docstring_fences)
    app.connect("doctree-resolved", _mark_tagged_cells)
    app.connect("html-page-context", _architecture_page_assets)
