import os
import sys
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as package_version
from pathlib import Path
from typing import Any

from sphinx.application import Sphinx

DOCS_DIR = Path(__file__).resolve().parent
REPO_ROOT = DOCS_DIR.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

# Package imports load a cluster-specific environment. Use a minimal docs
# profile so autodoc never depends on cluster-only data mirrors.
os.environ["CLUSTER"] = "docs"
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
    release = "0.0.1"
version = release

extensions = [
    "myst_parser",
    "nbsphinx",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx_togglebutton",
]

# Collapse the tutorial's setup helper cells by default. sphinx-togglebutton works off a
# CSS selector, not Jupyter cell tags (those do not survive nbsphinx -> HTML). The selector
# targets the code cells under the notebook's Setup section (#Pass-criteria is nested inside
# #Setup); later sections stay expanded.
togglebutton_selector = "#Setup .nbinput"
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


def setup(app: Sphinx) -> None:
    app.connect("autodoc-process-docstring", _convert_markdown_docstring_fences)
