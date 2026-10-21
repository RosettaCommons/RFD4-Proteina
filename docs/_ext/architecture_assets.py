"""Keep diagram source readable and serialize only the built HTML assets."""

import json
from pathlib import Path
from typing import TYPE_CHECKING

from embedded_json import rewrite_embedded_json

if TYPE_CHECKING:
    from sphinx.application import Sphinx

MESSAGE_ID = "tsncd-embedded-message"


def expand_payloads(value: object) -> object:
    """Expand JSON-string diagrams and subdiagrams without changing ordinary text."""
    if isinstance(value, list):
        return [expand_payloads(item) for item in value]
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            if key in {"data", "expansion"} and isinstance(item, str):
                item = json.loads(item)
            result[key] = expand_payloads(item)
        return result
    return value


def serialize_payloads(value: object) -> object:
    """Restore the renderer's string fields, retaining internal data objects."""
    if isinstance(value, list):
        return [serialize_payloads(item) for item in value]
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            item = serialize_payloads(item)
            serialized_field = key == "expansion" or (key == "data" and value.get("msgType") == "dataUpdate")
            if serialized_field and isinstance(item, (dict, list)):
                item = json.dumps(item, ensure_ascii=False, separators=(",", ":"))
            result[key] = item
        return result
    return value


def compact_architecture_assets(app: "Sphinx", exception: Exception | None) -> None:
    """Compact the copied assets after Sphinx finishes an HTML build."""
    if exception is not None or app.builder.format != "html":
        return
    directory = Path(app.outdir) / "_static/architecture/ncd"
    for path in sorted(directory.glob("*.html")):
        source = path.read_text()
        compact = rewrite_embedded_json(source, MESSAGE_ID, serialize_payloads, pretty=False)
        if compact != source:
            path.write_text(compact)


def setup(app: "Sphinx") -> dict[str, bool]:
    """Register output-only compaction for local and published documentation."""
    app.connect("build-finished", compact_architecture_assets)
    return {"parallel_read_safe": True, "parallel_write_safe": True}
