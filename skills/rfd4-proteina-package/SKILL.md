---
name: rfd4-proteina-package
description: >
  Package selected RFProteína designs into a simple shareable TAR organized by
  annotated-CIF input, with README and exact filter waterfall, one actual-values
  metrics CSV, each generated design, all associated on/off-target oracle refolds
  aligned on the recorded chain of interest, and available confidence JSON. Reach
  for this skill for "package passers", "make a design TAR", "share campaign
  results", or "bundle designs with their refolds". It may read a campaign ledger,
  but recipients need only ordinary files and no SQLite installation.
license: See the RFProteína repository (Baker Lab x NVIDIA collaboration)
---

# Package RFProteína designs

Create a recipient-facing archive, not a second provenance system. The campaign
ledger remains authoritative; the TAR contains only the files and explanations a
recipient needs to inspect selected designs and refolds.

## Required output

```text
package/
├── README.md
├── WATERFALL.md
├── metrics.csv
├── inputs/
│   └── <input-name>.cif
└── outputs/
    └── <input-name>/
        └── <design-id>/
            ├── <design-id>__design.cif
            ├── <design-id>__<role>__<backend>__<prediction-id>__refold.cif
            └── <design-id>__<role>__<backend>__<prediction-id>__confidence.json
```

The generated design is the oracle input and alignment reference. Include it once.
Transform each copied refold into the design's frame by superimposing the recorded
chain or chains of interest, then write only that aligned refold to the package.
Include confidence JSON when it exists. Do not add a database, manifest, checksum
file, duplicate raw/aligned structures, provenance JSON, filter-results table, or
alignment metadata file.

## Workflow

1. Confirm the immutable criteria-set version and whether the selection is all
   passers or an explicitly defined unique-passer subset.
2. Reconcile the campaign ledger. Stop on missing selected designs, missing input
   mappings, missing criteria evidence, or ambiguous chain-of-interest alignment.
3. Group selected designs by their exact annotated-CIF input and assign unique,
   filesystem-safe names.
4. Copy each annotated CIF and design. Copy and align every linked oracle prediction,
   preserving role (`on-target`, named off-target, checkpoint), backend, and
   prediction identity in its filename. Copy available confidence JSON beside it.
5. Write `metrics.csv` with one row per design and the actual numeric values used
   by every filter. Do not replace those values with Boolean pass/fail columns.
6. Write `WATERFALL.md` with the exact metric, operator, threshold, missing policy,
   and cumulative counts overall and per input.
7. Write a short `README.md`, create the TAR, reopen it, and validate its expected
   paths and one-to-one design/input grouping.

Read [references/package-format.md](references/package-format.md) when building or
reviewing the archive. Read [references/alignment.md](references/alignment.md)
before aligning refolds.

## Builder

```bash
python skills/rfd4-proteina-package/scripts/package_campaign.py --help
```

The builder reads the campaign database only while constructing the archive. The
archive itself is a pure directory tree of Markdown, CSV, CIF, and optional JSON
files.
Compressed source structures are decoded into ordinary CIFs in the archive; never
copy gzip or Zstandard bytes under a plain `.cif` filename.
