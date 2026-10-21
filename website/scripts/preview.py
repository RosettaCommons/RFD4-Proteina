#!/usr/bin/env python3
"""Serve the project page and built Sphinx docs together, without copying build outputs."""

import argparse
from contextlib import suppress
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

SITE_ROOT = Path(__file__).resolve().parents[1]


class PreviewHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, docs_directory: str | Path, **kwargs):
        self.docs_directory = str(docs_directory)
        super().__init__(*args, **kwargs)

    def translate_path(self, path: str) -> str:
        route = urlsplit(path).path
        if route == "/latest" or route.startswith("/latest/"):
            directory = self.directory
            self.directory = self.docs_directory
            try:
                return super().translate_path(path[len("/latest") :])
            finally:
                self.directory = directory
        return super().translate_path(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=4187)
    parser.add_argument("--bind", default="127.0.0.1")
    parser.add_argument("--docs-dir", type=Path, default=SITE_ROOT.parent / "docs/_build/html")
    args = parser.parse_args()
    if not (args.docs_dir / "index.html").is_file():
        parser.error("Build the docs first with RFD4_SITE_ROOT=../ python -m sphinx -b html docs docs/_build/html")
    handler = partial(PreviewHandler, directory=str(SITE_ROOT), docs_directory=args.docs_dir.resolve())
    with ThreadingHTTPServer((args.bind, args.port), handler) as server:
        print(f"Website: http://{args.bind}:{args.port}/", flush=True)
        print(f"Docs: http://{args.bind}:{args.port}/latest/", flush=True)
        with suppress(KeyboardInterrupt):
            server.serve_forever()


if __name__ == "__main__":
    main()
