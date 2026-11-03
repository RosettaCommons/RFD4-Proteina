"""Format embedded architecture data for review, including serialized subdiagrams.

Run from any directory; use --check to verify formatting without changing files.
The Sphinx build restores compact serialized payloads only in its output directory.
"""

import argparse
import re
import sys
from pathlib import Path

DOCS = Path(__file__).resolve().parents[1] / "docs"
sys.path.insert(0, str(DOCS / "_ext"))

from architecture_assets import MESSAGE_ID, expand_payloads  # noqa: E402
from embedded_json import rewrite_embedded_json  # noqa: E402

DIAGRAMS = DOCS / "_static/architecture/ncd"


def format_page(source: str) -> str:
    """Indent the embedded message and separate adjacent HTML tags."""
    source = source.replace(
        "<!-- Generated diagram page; do not edit. -->",
        "<!-- Generated diagram; format embedded data with scripts/format_architecture_html.py. -->",
    )
    source = rewrite_embedded_json(source, MESSAGE_ID, expand_payloads, pretty=True)
    return re.sub(r">(?=<)", ">\n", source)


def main() -> int:
    """Format every standalone diagram, or report files needing formatting."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Exit nonzero if any page needs formatting")
    args = parser.parse_args()
    changed = []
    for path in sorted(DIAGRAMS.glob("*.html")):
        source = path.read_text()
        formatted = format_page(source)
        if formatted != source:
            changed.append(path.name)
            if not args.check:
                path.write_text(formatted)
    print(f"{'Needs formatting' if args.check else 'Formatted'}: {', '.join(changed) or 'none'}")
    return int(args.check and bool(changed))


if __name__ == "__main__":
    raise SystemExit(main())
