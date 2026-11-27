"""Standalone tests for the HealthOmics example; no model, GPU or AWS credentials needed."""

import contextlib
import csv
import importlib.util
import io
import json
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

HELPER = Path(__file__).resolve().parents[1] / "examples" / "healthomics" / "bin" / "workflow.py"
SPEC = importlib.util.spec_from_file_location("healthomics_workflow", HELPER)
workflow = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(workflow)


class HealthOmicsWorkflowTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.previous = Path.cwd()
        os.chdir(self.root)
        self.addCleanup(os.chdir, self.previous)
        (self.root / "model").mkdir()
        (self.root / "model" / "config.json").write_text("{}")
        (self.root / "model" / "model.safetensors").write_bytes(b"test fixture, not model weights")
        self.input = self.root / "input ; echo unexpected.cif"
        self.input.write_text("data_fixture\n")
        self.settings = {"seed": 42, "num_replicates": 5, "batch_size": 1, "overrides": ["compile_lengths=null"]}
        self.settings_path = self.root / "settings.json"
        self.request = {
            "input_cif": str(self.input),
            "model_bundle": str(self.root / "model"),
            "settings": str(self.settings_path),
            "shard_id": 0,
            "num_shards": 2,
            "container_image": "test-image@sha256:fixture",
        }
        self.save_settings()

    def save_settings(self):
        self.settings_path.write_text(json.dumps(self.settings))

    def read_manifest(self, shard=0):
        return json.loads((self.root / f"shard-{shard}" / "provenance.json").read_text())

    def successful_cli(self, command, **kwargs):
        self.assertIsInstance(command, list)
        self.assertNotIn("shell", kwargs)
        if command[0] != "rfd4-generate":
            return subprocess.CompletedProcess(command, 0, stdout='{"gpu_name": "test fixture"}')
        if "--cfg" in command:
            kwargs["stdout"].write("seed: 42\n")
        else:
            output = Path(
                json.loads(next(value.removeprefix("out_dir=") for value in command if value.startswith("out_dir=")))
            )
            shard = int(next(value.removeprefix("job_id=") for value in command if value.startswith("job_id=")))
            seed = int(next(value.removeprefix("seed=") for value in command if value.startswith("seed="))) + shard
            count = 3 if shard == 0 else 2
            rows = []
            for index in range(count):
                path = output / "inference_samples" / str(index) / f"{index}_generated.cif"
                path.parent.mkdir(parents=True)
                path.write_text("data_fixture\n")
                rows.append({"seed": f"{seed}.0", "designed_structure_path": str(path)})
            inventory = output / "inference_metrics" / f"per_example_metrics_job{shard}.csv"
            inventory.parent.mkdir(parents=True)
            with inventory.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=["seed", "designed_structure_path"])
                writer.writeheader()
                writer.writerows(rows)
        return subprocess.CompletedProcess(command, 0)

    def test_uneven_shards_keep_cli_seed_and_replicate_semantics(self):
        with patch.object(workflow.subprocess, "run", side_effect=self.successful_cli) as cli:
            for shard in range(2):
                workflow.generate({**self.request, "shard_id": shard})
        first, second = self.read_manifest(0), self.read_manifest(1)
        self.assertEqual([first["generated_designs"], second["generated_designs"]], [3, 2])
        self.assertEqual([first["effective_seed"], second["effective_seed"]], [42, 43])
        self.assertIn("seed=42", second["command"])
        self.assertIn(f"inputs={json.dumps(str(self.input.resolve()))}", cli.call_args_list[2].args[0])
        self.assertEqual(first["runtime"]["gpu_name"], "test fixture")
        self.assertEqual([first["native_seed"], second["native_seed"]], [42, 43])
        self.assertEqual(len(first["input_sha256"]), 64)
        workflow.summarize(self.root, 2)
        summary = json.loads((self.root / "summary" / "summary.json").read_text())
        self.assertEqual(summary["generated_designs"], 5)
        self.assertEqual(summary["status"], "COMPLETED")

    def test_successful_exit_with_missing_outputs_fails(self):
        with (
            patch.object(workflow.subprocess, "run", return_value=subprocess.CompletedProcess([], 0, stdout="{}")),
            self.assertRaisesRegex(RuntimeError, "Expected 3"),
        ):
            workflow.generate(self.request)
        self.assertEqual(self.read_manifest()["status"], "FAILED")

    def test_cli_failure_retains_provenance(self):
        stderr = io.StringIO()
        with (
            patch.object(workflow.subprocess, "run", side_effect=subprocess.CalledProcessError(2, ["rfd4-generate"])),
            contextlib.redirect_stderr(stderr),
            self.assertRaises(subprocess.CalledProcessError),
        ):
            workflow.generate(self.request)
        self.assertEqual(self.read_manifest()["status"], "FAILED")
        self.assertIn("error", self.read_manifest())
        line = next(line for line in stderr.getvalue().splitlines() if line.startswith("RFD4_SHARD_FAILURE "))
        self.assertEqual(json.loads(line.split(" ", 1)[1]), self.read_manifest())

    def test_failed_cli_keeps_partial_inventory_in_failure_log(self):
        def partial_failure(command, **kwargs):
            result = self.successful_cli(command, **kwargs)
            if command[0] == "rfd4-generate" and "--cfg" not in command:
                (self.root / "shard-0/inference_samples/2/2_generated.cif").unlink()
                raise subprocess.CalledProcessError(1, command)
            return result

        stderr = io.StringIO()
        with (
            patch.object(workflow.subprocess, "run", side_effect=partial_failure),
            contextlib.redirect_stderr(stderr),
            self.assertRaises(subprocess.CalledProcessError),
        ):
            workflow.generate(self.request)
        record = self.read_manifest()
        self.assertEqual(record["status"], "FAILED")
        self.assertEqual(record["generated_designs"], 2)
        line = next(line for line in stderr.getvalue().splitlines() if line.startswith("RFD4_SHARD_FAILURE "))
        self.assertEqual(json.loads(line.split(" ", 1)[1])["generated_files"], record["generated_files"])

    def test_wrong_native_seed_prevents_success(self):
        def wrong_seed(command, **kwargs):
            result = self.successful_cli(command, **kwargs)
            if command[0] == "rfd4-generate" and "--cfg" not in command:
                path = self.root / "shard-0/inference_metrics/per_example_metrics_job0.csv"
                path.write_text(path.read_text().replace("42.0", "99.0"))
            return result

        with (
            patch.object(workflow.subprocess, "run", side_effect=wrong_seed),
            contextlib.redirect_stderr(io.StringIO()),
            self.assertRaisesRegex(ValueError, "Native inventory seed"),
        ):
            workflow.generate(self.request)
        self.assertEqual(self.read_manifest()["status"], "FAILED")

    def test_summary_rejects_missing_changed_and_unlisted_output_files(self):
        for alteration in ["missing", "changed", "unlisted"]:
            with self.subTest(alteration=alteration):
                case = self.root / alteration
                case.mkdir()
                previous = Path.cwd()
                os.chdir(case)
                try:
                    with patch.object(workflow.subprocess, "run", side_effect=self.successful_cli):
                        for shard in range(2):
                            workflow.generate({**self.request, "shard_id": shard})
                    path = case / "shard-0/inference_samples/0/0_generated.cif"
                    if alteration == "missing":
                        path.unlink()
                    elif alteration == "changed":
                        # Keep the byte count so the digest, not a size check, detects corruption.
                        path.write_text("data_changed\n")
                    else:
                        path.with_name("extra_generated.cif").write_text("data_fixture\n")
                    with self.assertRaises(ValueError):
                        workflow.summarize(Path("."), 2)
                    self.assertFalse((case / "summary").exists())
                finally:
                    os.chdir(previous)

    def test_summary_rejects_paths_and_symlinks_outside_the_shard(self):
        for alteration in ["parent-path", "symlink"]:
            with self.subTest(alteration=alteration):
                case = self.root / alteration
                case.mkdir()
                previous = Path.cwd()
                os.chdir(case)
                try:
                    with patch.object(workflow.subprocess, "run", side_effect=self.successful_cli):
                        for shard in range(2):
                            workflow.generate({**self.request, "shard_id": shard})
                    outside = case / "outside" / "0_generated.cif"
                    outside.parent.mkdir()
                    outside.write_text("data_fixture\n")
                    manifest = case / "shard-0/provenance.json"
                    record = json.loads(manifest.read_text())
                    if alteration == "parent-path":
                        record["generated_files"][0]["path"] = "../outside/0_generated.cif"
                        manifest.write_text(json.dumps(record))
                    else:
                        path = case / "shard-0/inference_samples/0/0_generated.cif"
                        path.unlink()
                        path.symlink_to(outside)
                    with self.assertRaisesRegex(ValueError, "escapes its shard directory"):
                        workflow.summarize(Path("."), 2)
                    self.assertFalse((case / "summary").exists())
                finally:
                    os.chdir(previous)

    def test_failure_provenance_does_not_dump_environment_credentials(self):
        marker = "unit-test-value-must-not-appear-in-provenance"
        stderr = io.StringIO()
        with (
            patch.dict(os.environ, {"AWS_SECRET_ACCESS_KEY": marker, "AWS_SESSION_TOKEN": marker}),
            patch.object(workflow.subprocess, "run", side_effect=subprocess.CalledProcessError(1, ["probe"])),
            contextlib.redirect_stderr(stderr),
            self.assertRaises(subprocess.CalledProcessError),
        ):
            workflow.generate(self.request)
        self.assertIn("RFD4_SHARD_FAILURE ", stderr.getvalue())
        self.assertNotIn(marker, stderr.getvalue())
        self.assertNotIn(marker, (self.root / "shard-0/provenance.json").read_text())

    def test_failed_gpu_preflight_stops_before_model_generation(self):
        error = subprocess.CalledProcessError(1, ["python", "-c", "probe"], stderr="CUDA GPU required")
        with (
            patch.object(workflow.subprocess, "run", side_effect=error) as cli,
            self.assertRaises(subprocess.CalledProcessError),
        ):
            workflow.generate(self.request)
        self.assertEqual(cli.call_count, 1)
        self.assertIn("CUDA GPU required", self.read_manifest()["error"])

    def test_stubs_are_never_reported_as_generated_structures(self):
        with patch.object(workflow.subprocess, "run") as cli:
            for shard in range(2):
                workflow.generate({**self.request, "shard_id": shard}, plan_only=True)
        cli.assert_not_called()
        workflow.summarize(self.root, 2)
        summary = json.loads((self.root / "summary" / "summary.json").read_text())
        self.assertEqual(summary["status"], "STUB")
        self.assertEqual(summary["generated_designs"], 0)

    def test_existing_outputs_are_preserved(self):
        output = self.root / "shard-0"
        output.mkdir()
        (output / "keep").write_text("existing")
        with self.assertRaises(FileExistsError):
            workflow.generate(self.request, plan_only=True)
        self.assertEqual((output / "keep").read_text(), "existing")

    def test_missing_shard_prevents_successful_summary(self):
        workflow.generate(self.request, plan_only=True)
        with self.assertRaisesRegex(ValueError, "Missing"):
            workflow.summarize(self.root, 2)

    def test_conflicting_shard_settings_prevent_summary(self):
        workflow.generate(self.request, plan_only=True)
        self.settings["seed"] = 100
        self.save_settings()
        workflow.generate({**self.request, "shard_id": 1}, plan_only=True)
        with self.assertRaisesRegex(ValueError, "settings"):
            workflow.summarize(self.root, 2)

    def test_invalid_shards_seeds_and_overrides_fail_before_execution(self):
        bad_cases = [
            ("seed", True),
            ("seed", 2**32 - 1),
            ("num_replicates", 0),
            ("batch_size", 1.5),
            ("overrides", ["seed=9"]),
            ("overrides", ["hydra.run.dir=/elsewhere"]),
            ("overrides", ["metrics_tags=[ppi_refolding]"]),
            ("overrides", ["search.beam_width=4"]),
            ("overrides", "compile_lengths=null"),
        ]
        for key, value in bad_cases:
            with self.subTest(key=key, value=value):
                original = self.settings[key]
                self.settings[key] = value
                self.save_settings()
                with self.assertRaises(ValueError):
                    workflow.prepare(self.request)
                self.settings[key] = original
        self.save_settings()
        with self.assertRaises(ValueError):
            workflow.prepare({**self.request, "num_shards": 6})

    @unittest.skipUnless(shutil.which("nextflow"), "Nextflow 25.10.0 is needed for the workflow stub test")
    def test_nextflow_shards_and_summary(self):
        config = self.root / "local.config"
        config.write_text(
            "process { cpus = 1; memory = '1 GB'; " "withLabel:generation { cpus = 1; memory = '1 GB' } }\n"
        )
        result = subprocess.run(
            [
                "nextflow",
                "run",
                str(HELPER.parents[1] / "main.nf"),
                "-stub-run",
                "-c",
                str(config),
                "--input_cif",
                str(self.input),
                "--model_bundle",
                str(self.root / "model"),
                "--settings",
                str(self.settings_path),
                "--container_image",
                "stub-only",
                "--num_shards",
                "2",
                "--outdir",
                str(self.root / "published"),
            ],
            cwd=self.root,
            capture_output=True,
            text=True,
            timeout=90,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        summary = json.loads((self.root / "published" / "summary" / "summary.json").read_text())
        self.assertEqual(summary["status"], "STUB")
        self.assertEqual(summary["generated_designs"], 0)
        self.assertEqual(sorted(record["expected_designs"] for record in summary["shards"]), [2, 3])

    @unittest.skipUnless(shutil.which("nextflow"), "Nextflow 25.10.0 is needed for the workflow validation test")
    def test_nextflow_rejects_invalid_resources_before_submitting_tasks(self):
        for option, value in [("generation_cpus", "0"), ("generation_accelerator", "cpu")]:
            with self.subTest(option=option):
                result = subprocess.run(
                    [
                        "nextflow",
                        "run",
                        str(HELPER.parents[1] / "main.nf"),
                        "-stub-run",
                        "--input_cif",
                        str(self.input),
                        "--model_bundle",
                        str(self.root / "model"),
                        "--settings",
                        str(self.settings_path),
                        "--container_image",
                        "stub-only",
                        f"--{option}",
                        value,
                    ],
                    cwd=self.root,
                    capture_output=True,
                    text=True,
                    timeout=90,
                )
                output = result.stdout + result.stderr
                self.assertNotEqual(result.returncode, 0, output)
                self.assertIn(f"--{option} must be", output)
                self.assertNotIn("Submitted process", output)


if __name__ == "__main__":
    unittest.main()
