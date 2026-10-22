"""Shared helpers for readable source and compact embedded JSON in HTML assets."""

import json
import re
from collections.abc import Callable


def rewrite_embedded_json(
    source: str,
    script_id: str,
    transform: Callable[[object], object],
    *,
    pretty: bool,
) -> str:
    """Rewrite one application/json payload, preserving the surrounding HTML."""
    pattern = re.compile(rf'(<script type="application/json" id="{re.escape(script_id)}">)(.*?)(</script>)', re.DOTALL)
    match = pattern.search(source)
    if match is None:
        raise ValueError(f"Missing {script_id} script")
    payload = transform(json.loads(match[2]))
    options = {"indent": 2} if pretty else {"separators": (",", ":")}
    formatted = json.dumps(payload, ensure_ascii=False, **options).replace("<", r"\u003c")
    if pretty:
        formatted = "\n" + formatted + "\n"
    return source[: match.start(2)] + formatted + source[match.end(2) :]
