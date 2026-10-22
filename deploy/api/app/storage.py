"""Persistent local artifacts; one replica owns the mounted disk."""

import os
from pathlib import Path


class Blob:
    def __init__(self, root: Path, name: str):
        self.name = name
        self.path = root / name
        if not self.path.resolve().is_relative_to(root.resolve()):
            raise ValueError("Artifact path must stay within the store")

    def exists(self) -> bool:
        return self.path.is_file()

    def upload_from_string(self, content: str | bytes, content_type: str | None = None) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temp = self.path.with_suffix(self.path.suffix + ".tmp")
        temp.write_bytes(content.encode() if isinstance(content, str) else content)
        temp.replace(self.path)

    def upload_from_filename(self, source: str) -> None:
        self.upload_from_string(Path(source).read_bytes())

    def download_as_text(self) -> str:
        return self.path.read_text()

    def download_as_bytes(self) -> bytes:
        return self.path.read_bytes()

    def download_to_filename(self, target: str) -> None:
        Path(target).write_bytes(self.path.read_bytes())


class Store:
    def __init__(self):
        self.root = Path(os.environ.get("STORE_PATH", "/workspace/store"))
        self.root.mkdir(parents=True, exist_ok=True)

    def blob(self, name: str) -> Blob:
        return Blob(self.root, name)

    def list_blobs(self, prefix: str) -> list[Blob]:
        root = self.root / prefix
        return [self.blob(p.relative_to(self.root).as_posix()) for p in root.rglob("*") if p.is_file()]
