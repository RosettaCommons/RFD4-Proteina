#!/usr/bin/env python3
"""Convert a condition-spec YAML/JSON file into annotated CIF(s), one per example.

Run inside the project venv (``source ../../.venv/bin/activate``). ``generate.py`` consumes a
condition-spec file directly, so this is only for materializing CIFs to inspect or archive.

Usage::

    scripts/write_condition_spec_cifs.py docs/condition_spec_ppi.yaml --out-dir ./cifs
"""

import argparse
from pathlib import Path

from rfproteina.inference_utils.condition_spec import resolve_condition_spec_inputs
from rfproteina.utils.verbosity import configure_verbosity


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("spec", type=Path, help="Condition-spec YAML/JSON file (docs/condition_spec_yaml.md).")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Directory to write the CIFs into (default: <spec_stem>_cifs/ beside the spec).",
    )
    parser.add_argument(
        "--no-prevalidate",
        action="store_true",
        help="Skip the (default-on) validation-pipeline check on each built structure.",
    )
    parser.add_argument("--verbose", action="store_true", help="Show routine warnings and diagnostic output.")
    args = parser.parse_args()
    configure_verbosity(args.verbose)

    for path in resolve_condition_spec_inputs(args.spec, out_dir=args.out_dir, validate=not args.no_prevalidate):
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
