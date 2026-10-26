"""Exercise notebook setup without installing packages or requiring a GPU.

Run independently of the model test fixtures:
    python tests/test_colab_setup.py
"""

import ast
import io
import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
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
                    self.assertIn(f"torchvision==0.29.1+{backend}", install)
                    self.assertIn(
                        [sys.executable, "-m", "uv", "pip", "uninstall", "--python", sys.executable, "torchaudio"],
                        self.commands,
                    )
                    self.assertIn(f"rfd4-proteina[all-cuda{cuda}]==0.0.3", install)
                    self.assertEqual(install[install.index("--python") + 1], sys.executable)
                    self.assertEqual(install[install.index("--torch-backend") + 1], backend)
                    self.commands.clear()
                    with self.assertRaisesRegex(SystemExit, "Restart session"):
                        self.execute(source)
                    self.pid = 101
                    self.assertIn("Ready:", self.execute(source))
                    self.assertIn("Ready:", self.execute(source))
                    self.assertTrue(
                        all(
                            command[0] == "nvidia-smi" or command[:3] == [sys.executable, "-m", "rfproteina.generate"]
                            for command in self.commands
                        )
                    )
                    self.assertIn([sys.executable, "-m", "rfproteina.generate", "--help"], self.commands)
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

    def test_updated_requirements_reinstall_without_deleting_assets(self):
        source = setup_source(NOTEBOOKS[0])
        with self.assertRaises(SystemExit):
            self.execute(source)
        marker = self.root / ".rfd4-colab-setup.json"
        state = json.loads(marker.read_text())
        state["requirements"].append("torchaudio")
        marker.write_text(json.dumps(state))
        asset = self.root / "keep-results.cif"
        asset.write_text("existing design")
        self.pid = 101
        with self.assertRaisesRegex(SystemExit, "Restart session"):
            self.execute(source)
        self.assertEqual(asset.read_text(), "existing design")
        self.assertNotIn("torchaudio", json.loads(marker.read_text())["requirements"])

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
            self.assertEqual("", self.execute(setup_source(notebook)))
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


class EnzymeLaunchTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        notebook = json.loads((ROOT / "docs/examples/enzyme_design_tutorial.ipynb").read_text())
        source = next("".join(c["source"]) for c in notebook["cells"] if "def launch(" in "".join(c["source"]))
        tree = ast.parse(source)
        functions = ast.Module(
            body=[n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in {"run", "launch"}],
            type_ignores=[],
        )
        self.scope = dict(
            os=os,
            sys=SimpleNamespace(executable="/usr/bin/python3"),
            shlex=shlex,
            shutil=shutil,
            subprocess=subprocess,
            Path=Path,
            time=time,
            TUTORIAL_DIR=root,
            REPO_ROOT=root,
            HAS_SLURM=False,
            OVERWRITE_SAMPLING=False,
            METRICS_TAGS="[design]",
            _OF3_OVERRIDE="",
            CKPT_PATH="/model",
            CKPT_NAME="release.ckpt",
            STAGE_TIMES=[],
        )
        exec(compile(functions, "<enzyme helpers>", "exec"), self.scope)

    def test_local_launch_uses_kernel_python_not_sibling_executable(self):
        failure = subprocess.CalledProcessError(7, "generation")
        with patch.dict(os.environ, {"CLUSTER": "local"}), patch("subprocess.run", side_effect=failure) as run:
            with self.assertRaises(subprocess.CalledProcessError):
                self.scope["launch"](str(Path(self.temp.name) / "run"), "smoke", "input.cif", 1)
        command = shlex.split(run.call_args.args[0])
        self.assertEqual(command[:3], ["/usr/bin/python3", "-m", "rfproteina.generate"])
        self.assertIn("metrics_tags=[design]", command)

    def test_command_failure_is_reported_before_missing_metrics(self):
        failed = subprocess.CompletedProcess("generation", 127, "", "command not found")
        with patch("subprocess.run", return_value=failed), redirect_stdout(io.StringIO()):
            with self.assertRaises(subprocess.CalledProcessError) as error:
                self.scope["run"]("generation")
        self.assertEqual(error.exception.returncode, 127)


if __name__ == "__main__":
    unittest.main()
