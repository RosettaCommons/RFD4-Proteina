"""Exercise notebook setup without installing packages or requiring a GPU.

Run independently of the model test fixtures:
    python tests/test_colab_setup.py
"""

import ast
import io
import json
import subprocess
import sys
import tempfile
import unittest
import urllib.request
from contextlib import ExitStack, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

ConcretePath = type(Path())
ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = sorted((ROOT / "docs/examples").glob("*.ipynb"))


def setup_source(path):
    notebook = json.loads(path.read_text())
    return next(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code" and "IN_COLAB =" in "".join(cell["source"])
    )


class ColabSetupTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.cuda = 12
        self.pid = 100
        self.commands = []
        self.fail_install = False
        self.torch = SimpleNamespace(
            __version__="test",
            version=SimpleNamespace(cuda="12.6"),
            cuda=SimpleNamespace(is_available=lambda: True, get_device_name=lambda _: "A100"),
            ones=Mock(return_value=Mock()),
        )
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        for context in [
            patch.dict(
                sys.modules, {"google.colab": Mock(), "torch": self.torch, "rfproteina": Mock(), "py3Dmol": Mock()}
            ),
            patch.object(sys, "version_info", (3, 12, 13)),
            patch("pathlib.Path", side_effect=self.map_path),
            patch("shutil.which", return_value="/usr/bin/nvidia-smi"),
            patch("subprocess.run", side_effect=self.run_command),
            patch("os.getpid", side_effect=lambda: self.pid),
            patch("os.chdir"),
            patch.dict("os.environ", {}, clear=True),
            patch.object(urllib.request, "urlretrieve", side_effect=self.download),
        ]:
            self.stack.enter_context(context)

    def map_path(self, value):
        path = ConcretePath(value)
        return self.root / path.relative_to("/content") if str(path).startswith("/content/") else path

    def run_command(self, command, **kwargs):
        self.commands.append(command)
        if command[0] == "nvidia-smi":
            output = (
                "NVIDIA A100, 8.0\n" if "--query-gpu=name,compute_cap" in command else f"CUDA Version: {self.cuda}.0"
            )
            return SimpleNamespace(stdout=output, returncode=0)
        if self.fail_install:
            raise subprocess.CalledProcessError(1, command)
        return SimpleNamespace(returncode=0)

    def download(self, url, dest):
        self.assertIn("/v0.0.3/", url)
        dest.write_bytes(b"fixture")

    def execute(self, source):
        output = io.StringIO()
        with redirect_stdout(output):
            exec(compile(source, "<colab setup>", "exec"), {})
        return output.getvalue()

    def test_install_restart_and_rerun_all_notebooks(self):
        for notebook in NOTEBOOKS:
            for cuda, backend in [(12, "cu126"), (13, "cu130")]:
                with self.subTest(notebook=notebook.name, cuda=cuda):
                    marker = self.root / ".rfd4-colab-setup.json"
                    marker.unlink(missing_ok=True)
                    self.cuda, self.pid = cuda, 100
                    self.torch.version.cuda = str(cuda)
                    source = setup_source(notebook)
                    self.commands.clear()
                    with self.assertRaisesRegex(SystemExit, "Restart session"):
                        self.execute(source)
                    install = self.commands[-1]
                    self.assertEqual(install[:4], [sys.executable, "-m", "uv", "pip"])
                    self.assertIn(f"torch==2.14.1+{backend}", install)
                    self.assertIn(f"rfd4-proteina[all-cuda{cuda}]==0.0.3", install)
                    self.assertEqual(install[install.index("--python") + 1], sys.executable)
                    self.assertEqual(install[install.index("--torch-backend") + 1], backend)
                    self.commands.clear()
                    with self.assertRaisesRegex(SystemExit, "Restart session"):
                        self.execute(source)
                    self.pid = 101
                    self.assertIn("Ready:", self.execute(source))
                    self.assertIn("Ready:", self.execute(source))
                    self.assertTrue(all(command[0] == "nvidia-smi" for command in self.commands))
                    self.torch.ones.assert_called_with(1, device="cuda")

    def test_wrong_python_and_missing_gpu_stop_before_install(self):
        source = setup_source(NOTEBOOKS[0])
        with patch.object(sys, "version_info", (3, 13, 0)), self.assertRaisesRegex(SystemExit, "2026.07"):
            self.execute(source)
        with patch("shutil.which", return_value=None), self.assertRaisesRegex(SystemExit, "Select a GPU"):
            self.execute(source)
        self.assertEqual(self.commands, [])

    def test_old_driver_and_failed_install_do_not_mark_ready(self):
        source = setup_source(NOTEBOOKS[0])
        self.cuda = 11
        with self.assertRaisesRegex(SystemExit, "CUDA 12"):
            self.execute(source)
        self.cuda, self.fail_install = 12, True
        with self.assertRaises(subprocess.CalledProcessError):
            self.execute(source)
        self.assertFalse((self.root / ".rfd4-colab-setup.json").exists())

    def test_changed_cuda_profile_requires_clean_runtime(self):
        source = setup_source(NOTEBOOKS[0])
        with self.assertRaises(SystemExit):
            self.execute(source)
        self.cuda, self.pid = 13, 101
        with self.assertRaisesRegex(SystemExit, "Disconnect and delete runtime"):
            self.execute(source)

    def test_local_setup_does_not_install_or_download(self):
        sys.modules.pop("google.colab")
        for notebook in NOTEBOOKS:
            self.assertIn("existing environment", self.execute(setup_source(notebook)))
        self.assertEqual(self.commands, [])
        self.assertEqual(list(self.root.iterdir()), [])

    def test_notebook_code_and_badges(self):
        for path in NOTEBOOKS:
            notebook = json.loads(path.read_text())
            self.assertEqual(notebook["cells"][1]["cell_type"], "markdown")
            for cell in notebook["cells"]:
                if cell["cell_type"] == "code":
                    ast.parse("".join(cell["source"]))
                    self.assertFalse(cell.get("outputs"))
                    self.assertIsNone(cell.get("execution_count"))


if __name__ == "__main__":
    unittest.main()
