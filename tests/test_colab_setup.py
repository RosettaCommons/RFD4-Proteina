"""Focused notebook regressions; run without model fixtures: python tests/test_colab_setup.py."""

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
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import MagicMock, patch

NOTEBOOKS = Path(__file__).resolve().parents[1] / "docs/examples"


def cell_source(path, marker):
    return next(
        "".join(cell["source"])
        for cell in json.loads(path.read_text())["cells"]
        if cell["cell_type"] == "code" and marker in "".join(cell["source"])
    )


class ColabTests(unittest.TestCase):
    def test_install_restart_and_rerun(self):
        for notebook in sorted(NOTEBOOKS.glob("*.ipynb")):
            for cuda, backend in [(12, "cu126"), (13, "cu130")]:
                with self.subTest(notebook=notebook.name, cuda=cuda):
                    # Mock filesystem and GPU/package operations; execute the actual setup cell.
                    path = MagicMock()
                    path.exists.return_value = False
                    torch = MagicMock(__version__="test")
                    torch.version.cuda = str(cuda)
                    modules = {
                        "google.colab": MagicMock(),
                        "torch": torch,
                        "rfproteina": MagicMock(),
                        "py3Dmol": MagicMock(),
                    }
                    with (
                        patch.dict(sys.modules, modules),
                        patch.object(sys, "version_info", (3, 12)),
                        patch("pathlib.Path", return_value=path),
                        patch("shutil.which", return_value="nvidia-smi"),
                        patch("subprocess.run", return_value=MagicMock(stdout=f"CUDA Version: {cuda}.0, 8.0")) as run,
                        patch("urllib.request.urlretrieve"),
                        patch("os.chdir"),
                        patch.dict(os.environ),
                        patch("os.getpid", return_value=100) as pid,
                        redirect_stdout(io.StringIO()) as output,
                    ):
                        source = cell_source(notebook, "IN_COLAB =")
                        with self.assertRaisesRegex(SystemExit, "Restart session"):
                            exec(source, {})
                        install = run.call_args.args[0]
                        for requirement in (
                            f"rfd4-proteina[all-cuda{cuda}]==0.0.3",
                            f"torch==2.14.1+{backend}",
                            f"torchvision==0.29.1+{backend}",
                        ):
                            self.assertIn(requirement, install)
                        self.assertEqual(install[install.index("--python") + 1], sys.executable)
                        self.assertEqual(install[install.index("--torch-backend") + 1], backend)
                        run.assert_any_call(
                            [sys.executable, "-m", "uv", "pip", "uninstall", "--python", sys.executable, "torchaudio"],
                            check=True,
                        )
                        path.exists.return_value = True
                        path.read_text.return_value = path.write_text.call_args.args[0]
                        run.reset_mock()
                        with self.assertRaisesRegex(SystemExit, "Restart session"):
                            exec(source, {})
                        pid.return_value = 101
                        exec(source, {})
                        exec(source, {})
                        self.assertEqual(output.getvalue().count("Ready:"), 2)
                        commands = [call.args[0] for call in run.call_args_list]
                        help_command = [sys.executable, "-m", "rfproteina.generate", "--help"]
                        self.assertTrue(all(cmd[0] == "nvidia-smi" or cmd == help_command for cmd in commands))
                        self.assertIn(help_command, commands)
                        torch.ones.assert_called_with(1, device="cuda")

    def test_invalid_runtime_stops_before_install(self):
        source = cell_source(NOTEBOOKS / "enzyme_design_tutorial.ipynb", "IN_COLAB =")
        for version, gpu, message in [((3, 13), "nvidia-smi", "2026.07"), ((3, 12), None, "Select a GPU")]:
            with (
                self.subTest(version=version, gpu=gpu),
                patch.dict(sys.modules, {"google.colab": MagicMock()}),
                patch.object(sys, "version_info", version),
                patch("shutil.which", return_value=gpu),
                patch("subprocess.run") as run,
            ):
                with self.assertRaisesRegex(SystemExit, message):
                    exec(source, {})
                run.assert_not_called()

    def test_enzyme_launch_uses_kernel_python_and_reports_failure(self):
        tree = ast.parse(cell_source(NOTEBOOKS / "enzyme_design_tutorial.ipynb", "def launch("))
        helpers = ast.Module(
            [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in {"run", "launch"}], []
        )
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            scope = {
                "os": os,
                "sys": sys,
                "shlex": shlex,
                "shutil": shutil,
                "subprocess": subprocess,
                "time": time,
                "TUTORIAL_DIR": Path(directory),
                "REPO_ROOT": Path(directory),
                "HAS_SLURM": False,
                "OVERWRITE_SAMPLING": False,
                "METRICS_TAGS": "[design]",
                "_OF3_OVERRIDE": "",
                "CKPT_PATH": "/model",
                "CKPT_NAME": "release.ckpt",
            }
            exec(compile(helpers, "<enzyme helpers>", "exec"), scope)
            with (
                patch.dict(os.environ, {"CLUSTER": "local"}),
                patch(
                    "subprocess.run", return_value=subprocess.CompletedProcess("generation", 127, "", "failed")
                ) as run,
                self.assertRaises(subprocess.CalledProcessError) as error,
            ):
                scope["launch"](str(Path(directory) / "run"), "smoke", "input.cif", 1)
            self.assertEqual(error.exception.returncode, 127)
            command = shlex.split(run.call_args.args[0])
            self.assertEqual(command[:3], [sys.executable, "-m", "rfproteina.generate"])
            self.assertIn("metrics_tags=[design]", command)


if __name__ == "__main__":
    unittest.main()
