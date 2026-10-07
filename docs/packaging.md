# Releasing rfd4-proteina

The distribution is `rfd4-proteina`; the public import is:

```python
import rfd4_proteina as rfd4
```

`rfd4_proteina` exposes the implementation currently in `rfproteina`.
Packaging does not rename modules, checkpoint references or Hydra targets;
an internal rename is a separate change.
The package supports Python 3.12 and requires AtomWorks 3.0.0 and TMol 0.1.59.
Publish those dependencies first, with no Git requirements in their PyPI metadata.
TMol's native binaries must match the runtime Torch/CUDA versions; see the
[installation guide](installation.rst) for source-build settings.

## Validate

```bash
python -m pip install build twine packaging
python -m build
python -m twine check --strict dist/*
python scripts/check_distribution.py dist/*
```

The wheel is built from the sdist. Both contain model code, CLI entry points,
Hydra configs and explicitly selected reference inputs. Private files, custom
metrics, bulk experiment datasets and the repository scripts tree are excluded.
The optional PyRosetta helper stays beside its runtime metric.

The commands above check archive contents, metadata and dependency URLs.
Before release, also install wheel and source distributions
with dependencies in a clean Linux/CUDA environment, run `pip check`, compose all
four CLI configs, generate a design and verify a short training/resume run.
Use the actual PyPI dependencies for this final qualification, check the public
import from outside the checkout, and verify that the project metadata URLs
resolve to the published repository and documentation.

## Publish

No package publishing workflow is included. Building distributions locally does
not upload them to PyPI. Publication requires a separately reviewed release change
and explicit authorization; include the HF API and PEFT changes before
advertising those features.
