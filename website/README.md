# RFD4-Proteina project page

Buildless static site, published to the root of GitHub Pages. The Sphinx docs are built with
`RFD4_SITE_ROOT=../` and published next to the site under `latest/`, so the links to `latest/...` only
resolve on the deployed site. The tab line at the top is the same markup on
the project page, the Benchmarks page and the docs; the manuscript repository's `website/scripts/sync_nav.py` rewrites it in `index.html` and
`leaderboard.html`, and `docs/_templates/site-nav.html` is the docs copy.

Preview locally: `python -m http.server 4187 --bind 127.0.0.1 --directory website`, then open
<http://127.0.0.1:4187/> (docs links 404 locally).

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
