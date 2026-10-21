# RFD4-Proteina project page

Buildless static site, published to the root of GitHub Pages. The Sphinx docs are built with
`RFD4_SITE_ROOT=../` and published next to the site under `latest/`. The combined local preview below
serves the same paths. The tab line at the top is the same markup on
the project page, the Benchmarks page and the docs; the manuscript repository's `website/scripts/sync_nav.py` rewrites it in `index.html` and
`leaderboard.html`, and `docs/_templates/site-nav.html` is the docs copy.

Preview the website and documentation together from the repository root, using an
environment with the documentation dependencies installed:

```bash
RFD4_SITE_ROOT=../ python -m sphinx -b html docs docs/_build/html
python website/scripts/preview.py
```

Open <http://127.0.0.1:4187/>; documentation links resolve under `/latest/`.
Use `--docs-dir /path/to/built/html` for a different build location. The server reads
the website sources directly, so refresh the browser after edits; rebuild after docs edits.

- `index.html`: the project page (Home), laid out after the Proteina-Complexa page with a light and dark theme shared with the benchmark explorer through `localStorage`. Styles in `css/`
  and `widgets.css`; scripts `nav.js`, `authors.js`, `app.js`, `benchmarks.js`, `recycling.js` and
  `experiments.js` (structure viewers), `hero.js` and `structure-core.js` (3Dmol.js in `vendor/`). Structures are gzipped and decompressed in the browser; the four hero structures show a pre-rendered poster from `assets/hero/` until the live view is ready.
- `data/`: benchmark results and recycling schedules shown on the page.
- `assets/structures/`: design models shown in the viewers; `assets/paper/`: panels cropped from the
  paper figures.
- `leaderboard.html`, `benchmarks/`: the Benchmarks page (benchmark explorer) and its results snapshot
  (`benchmarks/data.json`). Home shows the paper's Figure 2 and links to it. `docs.html` redirects to `latest/index.html`.

The manuscript repository owns the byline and reviewed figure artwork. With Pillow,
`pdftoppm` and Inkscape installed, refresh those assets with:

    python website/scripts/sync_paper.py --paper-root /path/to/RFD4-Proteina-paper --work-dir /path/to/scratch/paper-site-sync

Fetch the manuscript repository first. The script verifies that selected sources match its
origin/main, regenerates the author list and paper panels, and copies selected CaM and H3pT11
CIFs without changing their coordinates. The manuscript-sync.json in assets/paper records the
manuscript commit, source/caption hashes, crop bounds and output hashes. Review every crop and
update surrounding captions, numerical statements and viewer selections when content changes.
The latent-space crop retains native SVG callouts; other panels use reviewed PDF artwork.
Keep website styling and interactions in this repository.

Refresh benchmark results from the same reviewed manuscript commit:

    python website/scripts/sync_benchmarks.py --paper-root /path/to/RFD4-Proteina-paper --source-commit COMMIT --snapshot-date YYYY-MM-DD

This imports the manuscript's per-target CSV and Figure 2 plotting tables, checks
counts, clustered rates, target coverage, and throughput against all plotted arms,
and writes the explorer JSON and downloadable CSVs under `benchmarks/`.
The default explorer view matches Figure 2; other conditioning arms and search
workflows remain available through the conditioning selector. Review headline
numbers in `index.html` whenever these source tables change.

The calmodulin viewer uses `drafting/figure4_cam/second_hit.cif`, sequence-matched to
`c09_31860910_1_bsfinal_11`: 2.48 nM in the final 2026-10-20 assay, with both
replicate fits rated Good and off-rates above the fitting floor. The manuscript's
Figure 4a retains its separate 69 pM example and the website footnote records its
fit limitation. Selection and source hashes are in `assets/paper/manuscript-sync.json`.

## Preprint revisions

The homepage Paper button and existing bookmarks use
`assets/paper/RFD4-Proteina.pdf`, which always contains the current version.
The smaller **Prior versions** link opens `preprint-versions.html`, with dated
PDF snapshots and concise changelogs, newest first.

Add a revision from the repository root:

```bash
python website/scripts/update_preprint.py add /path/to/revised.pdf \
    --change "Corrected Figure 4a." \
    --change "Clarified the calmodulin results." \
    --manuscript-commit FULL_MANUSCRIPT_COMMIT
```

The helper records the upload time in UTC, saves an immutable dated PDF under
`assets/paper/versions/`, updates the existing latest-PDF URL, and regenerates
the revision-history page. Dates on the page are displayed in Pacific time.
`--manuscript-commit` is optional. Use `--uploaded-at` only to backfill a known
upload timestamp, rather than using a PDF's build or file-modification date.
The initial upload date comes from public commit `f2c2ce74f`, when that PDF was added.

Run `python website/scripts/update_preprint.py check` before committing.
CI checks every archived checksum, the current PDF, and the generated page.
After changing the shared navigation header, regenerate the history page with
`python website/scripts/update_preprint.py render`.
