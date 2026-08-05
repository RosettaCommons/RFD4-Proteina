#!/usr/bin/env python3
"""Download the public OpenFold3 checkpoint and convert it to an AF3 parameter directory.

Conversion and the reload check run inside the pinned ``af3-open`` image, which must carry
CPU torch and ``alphafold3.model.of3_weight_converter`` -- neither is in upstream AF3. To
build one, see "Rebuilding the AF3/OpenFold3 runtime" in
``skills/rfd4-proteina-search/references/af3-oracle.md``.

The output is an ordinary AF3 model directory for ``AF3_MODEL_DIR``; the provenance file
written beside the parameters is how the runtime knows to enable the OF3 code paths.
One-shot per cluster: ~2.3 GB of download and a few minutes of CPU.

Usage::

    python scripts/setup_of3_weights.py --image "$AF3_IMAGE" --output_dir <dir>
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import urllib.request
from pathlib import Path
from typing import Any

import fire

OF3_CHECKPOINT_URL = "https://openfold.s3.amazonaws.com/staging/of3-p2-155k.pt"
OF3_CHECKPOINT_SHA256 = "3ae79a701f55dbc184002d010c92d347a35faabb981314c2fd1325211af40b77"
CONVERTED_FILENAME = "of3_ported_weights.bin.zst"
# Read by src/rfproteina/af3/run_alphafold_batch.py; the two names must match.
PROVENANCE_FILENAME = "of3_ported_weights.provenance.json"
OF3_EXPECTED_PARAMETER_ARRAYS = 407
OF3_EXPECTED_PARAMETER_ELEMENTS = 368_388_058


def sha256(path: Path) -> str:
    """Return the hex SHA-256 of a file."""
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def check_integrity(path: Path, expected_sha256: str) -> str:
    """Return the digest, refusing to proceed unless it is the pinned one.

    The conversion unpickles this file, so the gate runs before anything reads it.
    """
    digest = sha256(path)
    if digest.lower() != expected_sha256.lower():
        raise RuntimeError(
            f"Checkpoint SHA-256 is {digest}; expected {expected_sha256}. "
            "Refusing to deserialize an untrusted PyTorch checkpoint."
        )
    return digest


def download(url: str, destination: Path) -> None:
    """Download ``url`` to ``destination``, appearing at that path only once complete."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(f"{destination.suffix}.partial")
    with urllib.request.urlopen(url) as response, partial.open("wb") as handle:
        shutil.copyfileobj(response, handle, length=8 * 1024 * 1024)
    os.replace(partial, destination)


def image_labels(image: Path) -> dict[str, str]:
    """Return the image's Apptainer labels, which record what it was built from."""
    result = subprocess.run(
        ["apptainer", "inspect", "--json", str(image)],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)
    labels = (payload.get("data") or payload).get("attributes", {}).get("labels") or {}
    if not labels.get("AlphaFold3Commit"):
        raise RuntimeError(
            f"{image} has no AlphaFold3Commit label, so the conversion cannot be attributed "
            "to a fork revision. Rebuild it from the af3-open definition."
        )
    return labels


def run_in_image(image: Path, paths: list[Path], code: str) -> str:
    """Run ``code`` with the image's Python, binding the parents of ``paths``."""
    binds = sorted({str(path.resolve().parent) for path in paths})
    command = [
        "apptainer",
        "exec",
        *(argument for bind in binds for argument in ("--bind", f"{bind}:{bind}")),
        str(image.resolve()),
        # Spelled out, not bare "python": exec does not run the image's runscript.
        "/alphafold3_venv/bin/python",
        "-c",
        code,
    ]
    result = subprocess.run(command, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"In-image command failed with exit code {result.returncode}.\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    return result.stdout


def convert(image: Path, checkpoint: Path, output_dir: Path) -> None:
    """Convert the OF3 checkpoint into ``output_dir`` as AF3 ``.bin.zst`` params.

    Stored exactly as the converter emits them: the fork re-indexes ``ref_element`` at
    inference under the same ``of3_weights`` flag, so shifting rows here double-shifts.
    """
    code = (
        "from pathlib import Path\n"
        "import inspect\n"
        "from alphafold3.model import of3_weight_converter as _conv\n"
        # An image built before kierandidi/alphafold3#1 converts silently wrong weights: RNA
        # residues embed as their neighbour, and the template aatype term comes out transposed.
        "assert hasattr(_conv, '_reorder_msa_weights') and \"(2, 'aatype_linear_2')\" in "
        "inspect.getsource(_conv.map_template_embedder), (\n"
        "    'This image predates kierandidi/alphafold3#1 and would produce wrong OF3 weights. '\n"
        "    'Rebuild it from a recipe pinning cf0fade or later.')\n"
        "from alphafold3.model.of3_weight_converter import load_of3_checkpoint, map_of3_to_af3, save_af3_params\n"
        f"state = load_of3_checkpoint(Path({str(checkpoint.resolve())!r}), use_ema=True)\n"
        "converted = map_of3_to_af3(state, n_pairformer_blocks=48, n_msa_blocks=4, "
        "n_diff_blocks=24, n_super_blocks=6, n_atom_blocks=3)\n"
        f"save_af3_params(converted, Path({str(output_dir.resolve())!r}))\n"
    )
    run_in_image(image, [checkpoint, output_dir / CONVERTED_FILENAME], code)


def verify(image: Path, model_dir: Path) -> dict[str, Any]:
    """Load ``model_dir`` with AF3's own loader and check the parameter inventory.

    Call this on the final directory, provenance file included: AF3 picks its parameter
    file by scanning every file there, so anything written alongside the weights has to
    be proven harmless here rather than on a compute node an hour into a batch.
    """
    code = (
        "import json\n"
        "import jax\n"
        "import torch\n"
        "from pathlib import Path\n"
        "from alphafold3.model import params\n"
        f"p = params.get_model_haiku_params(Path({str(model_dir.resolve())!r}))\n"
        "n_arrays = sum(len(scope) for scope in p.values())\n"
        "n_elements = sum(value.size for scope in p.values() for value in scope.values())\n"
        f"assert n_arrays == {OF3_EXPECTED_PARAMETER_ARRAYS}, n_arrays\n"
        f"assert n_elements == {OF3_EXPECTED_PARAMETER_ELEMENTS}, n_elements\n"
        # The diffusion head reads its Fourier constants from the params file for
        # ported weights; AF3's own constants are hardcoded and differ.
        "assert 'fourier_embedding_weight' in p['diffuser/~/diffusion_head']\n"
        "assert 'fourier_embedding_bias' in p['diffuser/~/diffusion_head']\n"
        # ModelRunner.run_inference dereferences this on every fold.
        "assert '__identifier__' in p['__meta__']\n"
        "print(json.dumps({'parameter_arrays': n_arrays, 'parameter_elements': n_elements, "
        "'torch_version': torch.__version__, 'jax_version': jax.__version__}))\n"
    )
    return json.loads(run_in_image(image, [model_dir / CONVERTED_FILENAME], code).strip().splitlines()[-1])


def main(
    image: str,
    output_dir: str = "/path/to/data",
    checkpoint: str = "/path/to/data",
    url: str = OF3_CHECKPOINT_URL,
    checkpoint_sha256: str = OF3_CHECKPOINT_SHA256,
) -> None:
    """Download, convert and verify the OpenFold3 weights.

    The download is cached; the conversion is always redone, so re-running after an
    image or converter bump replaces the parameters rather than leaving stale ones.

    Args:
      image: ``af3-open`` Apptainer image carrying CPU torch and the OF3 converter.
        See ``skills/rfd4-proteina-search/references/af3-oracle.md`` to build one.
      output_dir: Destination AF3 model directory for the converted parameters.
      checkpoint: Download/cache location of the source PyTorch checkpoint.
      url: Source of the checkpoint. Override it only together with its digest.
      checkpoint_sha256: Digest the downloaded checkpoint must have.
    """
    image_path = Path(image)
    checkpoint_path = Path(checkpoint)
    model_dir = Path(output_dir)

    if not checkpoint_path.is_file():
        print(f"Downloading {url} to {checkpoint_path}", flush=True)
        download(url, checkpoint_path)
    checkpoint_digest = check_integrity(checkpoint_path, checkpoint_sha256)

    # Before converting: a missing label should not cost several minutes of CPU.
    labels = image_labels(image_path)

    model_dir.mkdir(parents=True, exist_ok=True)
    converted = model_dir / CONVERTED_FILENAME
    provenance_path = model_dir / PROVENANCE_FILENAME
    print(f"Converting {checkpoint_path} in {image_path} (several minutes)", flush=True)
    # Stage inside model_dir: os.replace cannot cross filesystems.
    with tempfile.TemporaryDirectory(prefix=".of3_conversion.", dir=model_dir) as staging:
        convert(image_path, checkpoint_path, Path(staging))
        os.replace(Path(staging) / CONVERTED_FILENAME, converted)

    # Leave nothing behind on failure: parameters without provenance load as AF3.
    try:
        provenance = {
            "schema_version": 1,
            "weight_family": "openfold3",
            "license": "Apache-2.0",
            "source_checkpoint": {
                "url": url,
                "filename": checkpoint_path.name,
                "bytes": checkpoint_path.stat().st_size,
                "sha256": checkpoint_digest,
            },
            "conversion": {
                "af3_source_repository": labels.get("AlphaFold3Repository"),
                "af3_source_commit": labels.get("AlphaFold3Commit"),
                "of3_port_commit": labels.get("OpenFold3PortCommit"),
                "use_ema": True,
                "image": str(image_path.resolve()),
            },
            "converted_checkpoint": {
                "filename": CONVERTED_FILENAME,
                "bytes": converted.stat().st_size,
                "sha256": sha256(converted),
            },
        }
        # Written atomically so a reader never sees a half-written file.
        staged_provenance = provenance_path.with_suffix(".tmp")
        staged_provenance.write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")
        os.replace(staged_provenance, provenance_path)
        inventory = verify(image_path, model_dir)
    except BaseException:
        provenance_path.unlink(missing_ok=True)
        converted.unlink(missing_ok=True)
        raise
    print(f"Verified {converted} ({inventory['parameter_arrays']} arrays, {inventory['parameter_elements']} elements)")
    print(f"Provenance: {provenance_path}")


if __name__ == "__main__":
    fire.Fire(main)
