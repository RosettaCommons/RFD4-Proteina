"""Regression checks for immutable PDF history and the stable latest URL."""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from update_preprint import HISTORY, LATEST_PDF, add_version, load_history, render_history


def pdf_bytes(label: str) -> bytes:
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 100 100] >>",
    ]
    data = b"%PDF-1.4\n%" + label.encode() + b"\n"
    offsets = []
    for number, content in enumerate(objects, 1):
        offsets.append(len(data))
        data += str(number).encode() + b" 0 obj\n" + content + b"\nendobj\n"
    start = len(data)
    data += b"xref\n0 4\n0000000000 65535 f \n"
    data += b"".join(f"{offset:010d} 00000 n \n".encode() for offset in offsets)
    data += b"trailer\n<< /Size 4 /Root 1 0 R >>\nstartxref\n" + str(start).encode() + b"\n%%EOF\n"
    return data


class PreprintVersionsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.site = Path(self.temp.name) / "website"
        (self.site / "assets/paper/versions").mkdir(parents=True)
        (self.site / "data").mkdir()
        (self.site / "index.html").write_text(
            '<header class="site-nav"><a href="index.html" aria-current="page">Home</a></header>'
        )
        self.original = pdf_bytes("original")
        self.archive = self.site / "assets/paper/versions/original.pdf"
        self.archive.write_bytes(self.original)
        (self.site / LATEST_PDF).write_bytes(self.original)
        self.history = {
            "schema_version": 1,
            "latest_pdf": LATEST_PDF,
            "versions": [
                {
                    "number": 1,
                    "uploaded_at": "2026-10-21T12:07:43Z",
                    "pdf": "assets/paper/versions/original.pdf",
                    "sha256": hashlib.sha256(self.original).hexdigest(),
                    "changes": ["Initial preprint."],
                }
            ],
        }
        (self.site / HISTORY).write_text(json.dumps(self.history))

    def test_successive_updates_preserve_old_bytes_and_latest_url(self) -> None:
        for number in [2, 3]:
            pdf = Path(self.temp.name) / f"revision-{number}.pdf"
            data = pdf_bytes(f"revision {number}")
            pdf.write_bytes(data)
            version = add_version(self.site, pdf, [f"Correction {number}."], f"2026-10-{20 + number}T12:00:00Z")
            self.assertEqual(version["number"], number)
            self.assertEqual((self.site / LATEST_PDF).read_bytes(), data)
            self.assertEqual((self.site / version["pdf"]).read_bytes(), data)
        history = load_history(self.site)
        self.assertEqual(len(history["versions"]), 3)
        self.assertEqual(self.archive.read_bytes(), self.original)
        self.assertEqual(history["latest_pdf"], LATEST_PDF)
        second = history["versions"][1]
        self.assertEqual((self.site / second["pdf"]).read_bytes(), pdf_bytes("revision 2"))

    def test_duplicate_pdf_does_not_mutate_history(self) -> None:
        before = (self.site / HISTORY).read_bytes()
        with self.assertRaisesRegex(ValueError, "already"):
            add_version(self.site, self.archive, ["No change."], "2026-10-22T12:00:00Z")
        self.assertEqual((self.site / HISTORY).read_bytes(), before)
        self.assertEqual((self.site / LATEST_PDF).read_bytes(), self.original)

    def test_changed_archive_or_alias_is_detected(self) -> None:
        self.archive.write_bytes(pdf_bytes("tampered"))
        with self.assertRaisesRegex(ValueError, "archive"):
            load_history(self.site)
        self.archive.write_bytes(self.original)
        (self.site / LATEST_PDF).write_bytes(pdf_bytes("unrecorded replacement"))
        with self.assertRaisesRegex(ValueError, "latest PDF"):
            load_history(self.site)

    def test_backdated_upload_and_invalid_pdf_are_rejected(self) -> None:
        pdf = Path(self.temp.name) / "new.pdf"
        pdf.write_bytes(pdf_bytes("new"))
        with self.assertRaisesRegex(ValueError, "predate"):
            add_version(self.site, pdf, ["Correction."], "2026-10-20T12:00:00Z")
        pdf.write_text("<html>This is not a PDF.</html>")
        with self.assertRaisesRegex(ValueError, "complete PDF"):
            add_version(self.site, pdf, ["Correction."], "2026-10-22T12:00:00Z")
        self.assertEqual((self.site / LATEST_PDF).read_bytes(), self.original)

    def test_existing_archive_is_never_overwritten(self) -> None:
        pdf = Path(self.temp.name) / "new.pdf"
        pdf.write_bytes(pdf_bytes("new"))
        collision = self.site / "assets/paper/versions/RFD4-Proteina-v2-20261022T120000Z.pdf"
        collision.write_bytes(b"Keep this file.")
        with self.assertRaisesRegex(ValueError, "overwrite"):
            add_version(self.site, pdf, ["Correction."], "2026-10-22T12:00:00Z")
        self.assertEqual(collision.read_bytes(), b"Keep this file.")

    def test_changelog_is_escaped_and_versions_are_newest_first(self) -> None:
        pdf = Path(self.temp.name) / "new.pdf"
        pdf.write_bytes(pdf_bytes("new"))
        add_version(self.site, pdf, ["Corrected <script>alert(1)</script> & labels."], "2026-10-22T12:00:00Z")
        page = render_history(self.site, load_history(self.site))
        self.assertIn("&lt;script&gt;alert(1)&lt;/script&gt; &amp; labels.", page)
        self.assertNotIn("<script>alert(1)</script>", page)
        self.assertLess(page.index('id="v2"'), page.index('id="v1"'))
        self.assertIn('href="assets/paper/RFD4-Proteina.pdf"', page)
        self.assertNotIn('aria-current="page"', page)
        self.assertIn("PDT", page)


if __name__ == "__main__":
    unittest.main()
