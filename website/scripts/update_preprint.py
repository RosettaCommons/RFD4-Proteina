#!/usr/bin/env python3
"""Archive preprint revisions, keep the existing latest-PDF URL, and render their history."""

import argparse
import hashlib
import html
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

SITE = Path(__file__).resolve().parents[1]
LATEST_PDF = "assets/paper/RFD4-Proteina.pdf"
HISTORY = "data/preprint-versions.json"


def upload_time(value: str) -> datetime:
    timestamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if timestamp.tzinfo is None:
        raise ValueError("Upload timestamps must include a timezone.")
    return timestamp.astimezone(timezone.utc)


def load_history(site: Path) -> dict:
    history = json.loads((site / HISTORY).read_text())
    if history["schema_version"] != 1 or history["latest_pdf"] != LATEST_PDF or not history["versions"]:
        raise ValueError("Invalid preprint history or changed latest-PDF URL.")
    previous = None
    seen = set()
    for number, version in enumerate(history["versions"], 1):
        timestamp = upload_time(version["uploaded_at"])
        if version["number"] != number or (previous is not None and timestamp < previous):
            raise ValueError("Version numbers and upload timestamps must remain chronological.")
        previous = timestamp
        path = Path(version["pdf"])
        if path.parent != Path("assets/paper/versions") or path.suffix != ".pdf":
            raise ValueError("Version PDFs must live in assets/paper/versions/.")
        archive = (site / path).resolve()
        if not archive.is_relative_to(site.resolve()):
            raise ValueError("Archive path escapes the website.")
        digest = hashlib.sha256(archive.read_bytes()).hexdigest()
        if digest != version["sha256"] or digest in seen:
            raise ValueError(f"Changed or duplicate archive: {path}")
        seen.add(digest)
        if not version["changes"] or any(not isinstance(note, str) or not note.strip() for note in version["changes"]):
            raise ValueError("Every version needs concise, nonempty revision notes.")
    if hashlib.sha256((site / LATEST_PDF).read_bytes()).hexdigest() != history["versions"][-1]["sha256"]:
        raise ValueError("The latest PDF does not match its recorded version; use the update helper.")
    return history


def render_history(site: Path, history: dict) -> str:
    homepage = (site / "index.html").read_text()
    match = re.search(r'<header class="site-nav".*?</header>', homepage, re.S)
    if match is None:
        raise ValueError("Cannot find the shared website header.")
    navigation = match[0].replace(' aria-current="page"', "")
    entries = []
    for version in reversed(history["versions"]):
        number = version["number"]
        current = number == history["versions"][-1]["number"]
        badge = '<span class="version-current">Current</span>' if current else ""
        date = upload_time(version["uploaded_at"]).astimezone(ZoneInfo("America/Los_Angeles"))
        label = date.strftime("%B %d, %Y at %I:%M %p %Z").replace(" 0", " ")
        notes = "\n".join(f"          <li>{html.escape(note)}</li>" for note in version["changes"])
        entries.append(f"""    <article class="preprint-version" id="v{number}" aria-labelledby="version-{number}">
      <div class="version-heading">
        <h2 id="version-{number}">Version {number} {badge}</h2>
        <a class="version-pdf" href="{html.escape(version["pdf"], quote=True)}"
           target="_blank" rel="noopener noreferrer" aria-label="Read version {number} PDF">PDF &#8599;</a>
      </div>
      <p class="version-date">Uploaded
        <time datetime="{html.escape(version["uploaded_at"], quote=True)}">{label}</time></p>
      <ul>
{notes}
      </ul>
    </article>""")
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Preprint versions · RFD4-Proteina</title>
  <meta name="description" content="RFD4-Proteina preprint PDFs and concise revision history.">
  <link rel="icon" href="assets/favicon.svg?v=northern-nights" type="image/svg+xml">
  <link rel="stylesheet" href="assets/site-colors.css">
  <link rel="stylesheet" href="css/site-nav.css">
  <link rel="stylesheet" href="css/preprint-versions.css">
  <script src="assets/theme.js"></script>
  <script src="site-nav.js" defer></script>
</head>
<body class="preprint-history-page">
{navigation}
<main class="preprint-history">
  <h1>Preprint versions</h1>
  <p class="preprint-current-link"><a href="{LATEST_PDF}" target="_blank" rel="noopener noreferrer">
    Read the current preprint &#8599;</a></p>
  <div class="preprint-version-list">
{chr(10).join(entries)}
  </div>
</main>
</body>
</html>
"""


def add_version(
    site: Path, pdf: Path, changes: list[str], uploaded_at: str, manuscript_commit: str | None = None
) -> dict:
    history = load_history(site)
    changes = [note.strip() for note in changes]
    if not changes or any(not note for note in changes):
        raise ValueError("Supply at least one nonempty changelog item.")
    data = pdf.read_bytes()
    if not data.startswith(b"%PDF-") or b"%%EOF" not in data[-2048:]:
        raise ValueError("The input must be a complete PDF.")
    digest = hashlib.sha256(data).hexdigest()
    if any(version["sha256"] == digest for version in history["versions"]):
        raise ValueError("This PDF is already in the revision history.")
    timestamp = upload_time(uploaded_at)
    if timestamp < upload_time(history["versions"][-1]["uploaded_at"]):
        raise ValueError("A new upload cannot predate the latest recorded upload.")
    if manuscript_commit and not re.fullmatch(r"[0-9a-f]{40}", manuscript_commit):
        raise ValueError("Use the full manuscript Git commit SHA.")
    number = history["versions"][-1]["number"] + 1
    filename = f"assets/paper/versions/RFD4-Proteina-v{number}-{timestamp.strftime('%Y%m%dT%H%M%SZ')}.pdf"
    archive = site / filename
    if archive.exists():
        raise ValueError(f"Refusing to overwrite an archived version: {filename}")
    version = {
        "number": number,
        "uploaded_at": timestamp.isoformat(timespec="seconds").replace("+00:00", "Z"),
        "pdf": filename,
        "sha256": digest,
        "changes": changes,
    }
    if manuscript_commit:
        version["manuscript_commit"] = manuscript_commit
    history["versions"].append(version)
    page = render_history(site, history)
    with archive.open("xb") as handle:
        handle.write(data)
    (site / LATEST_PDF).write_bytes(data)
    (site / HISTORY).write_text(json.dumps(history, indent=2, ensure_ascii=False) + "\n")
    (site / "preprint-versions.html").write_text(page)
    return version


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    add = commands.add_parser("add", help="Archive a new PDF and update the existing latest link.")
    add.add_argument("pdf", type=Path)
    add.add_argument("--change", action="append", required=True, help="One concise changelog item; repeat as needed.")
    add.add_argument(
        "--uploaded-at", default=None, help="ISO timestamp with timezone; defaults to the upload time now."
    )
    add.add_argument("--manuscript-commit", help="Full manuscript source commit, when available.")
    commands.add_parser("render", help="Regenerate the history page after shared-header changes.")
    commands.add_parser("check", help="Check archived bytes, latest PDF, and the rendered history page.")
    args = parser.parse_args()
    try:
        if args.command == "add":
            timestamp = args.uploaded_at or datetime.now(timezone.utc).isoformat(timespec="seconds")
            version = add_version(SITE, args.pdf, args.change, timestamp, args.manuscript_commit)
            print(f"Added version {version['number']}: {version['pdf']}")
        else:
            history = load_history(SITE)
            expected = render_history(SITE, history)
            page = SITE / "preprint-versions.html"
            if args.command == "render":
                page.write_text(expected)
                print("Rendered preprint history.")
            elif page.read_text() != expected:
                raise ValueError("The history page is stale. Run this script with 'render'.")
            else:
                print(f"Verified {len(history['versions'])} immutable versions and the latest PDF.")
    except (ValueError, OSError, KeyError) as exc:
        parser.exit(1, f"{exc}\n")


if __name__ == "__main__":
    main()
