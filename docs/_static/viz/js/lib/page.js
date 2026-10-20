/* Mounting a visualisation page (docs/architecture/viz/<slug>.md) and the helpers every page shares.
 *
 * PAGE MARKUP (raw HTML in the page; tests/test_viz_docs.py checks it)
 *
 *   <div class="viz-stage" data-viz-page="SLUG" aria-label="...">
 *     <img class="viz-poster only-light" src="../../_static/viz/posters/SLUG-light.png" width=".." height=".."
 *          loading="lazy" decoding="async" alt="...">
 *     <img class="viz-poster only-dark" src="../../_static/viz/posters/SLUG-dark.png" ... same alt>
 *   </div>
 *   <script type="module" src="../../_static/viz/js/page-SLUG.js"></script>
 *
 * A stage marked `data-viz-pending` is still being built: it shows a placeholder, is never
 * mounted, and its posters carry `data-viz-src` instead of `src` until they exist.
 *
 * PAGE MODULE: js/page-SLUG.js calls definePage once at the top level:
 *
 *   import { definePage } from './lib/page.js';
 *   export default definePage({
 *     slug: 'atlas',
 *     styles: ['atlas.css'],                         // under css/, loaded before mount
 *     data: { params: 'data/atlas-params.json' },    // under _static/viz/, fetched in parallel
 *     async mount(stage, ctx) { ... },               // build the app inside the stage
 *   });
 *
 * FIGURE LAYOUT (arch.css, viz.css): a figure has no frame. Top to bottom: the title
 * (<p class="arch-title"><span class="arch-title-name">name</span> claim</p>), the drawing (a
 * `.viz-view` where it is a canvas or a scene), and, for the page's one instrument only, the one-line
 * readout (controls.js readout()) and the control row (controls.js controlRow(): play and one slider).
 * The caption follows. A stage is 88rem wide; add `viz-stage-wide` for 67rem. The instrument's stage
 * also takes `arch-band`, the flat band one tone off the page. Every other figure is a plain
 * <figure class="viz-figure">. Nothing moves without the reader's input.
 *
 * The stage mounts once it comes within half a screen of the viewport, after the story pages' fonts
 * (a canvas draws with the face that is ready). It goes live (class
 * viz-live: posters removed) when mount resolves. If a stylesheet, a data file
 * or mount fails, the stage gets its posters back, class viz-failed and a one-line note (the
 * error's `userMessage`, e.g. scene3d's "needs WebGL2", or a generic sentence); the error goes to
 * the console. `data-viz-state` on the stage is waiting | loading | live | failed, and
 * `document.documentElement.dataset.vizReady` holds the number of stages, for the poster tool.
 *
 * ctx (the second argument of mount):
 *   slug, stage                     the page and its .viz-stage element
 *   data                            {key: parsed JSON} for every entry of `data`
 *   load(path)                      fetch another JSON under _static/viz/ (e.g. an example picked later)
 *   asset(path)                     a URL under _static/viz/
 *   theme(), onTheme(fn)            'light' | 'dark'; fn(theme) after every theme change (returns unsubscribe)
 *   reducedMotion(), onReducedMotion(fn)
 *   unpack(field), take(array, ...i)  typed-array decoding (the explainer's lib/data.js)
 *   fact(key, dotted.path)          a value from ctx.data (key = data key or file stem)
 *   factSpan(key, path, options)    a <span data-viz-fact> showing that value (see formatNumber)
 *   urlState(defaults)              {get(), set(partial)}: page state kept in the query string, so a
 *                                   view can be linked (numbers and strings only)
 *   poster                          true when the page was opened by the poster tool (?viz-poster):
 *                                   show the canonical state, no hover, no transient hints
 *
 * Also exported for figures a page mounts outside its stage (they need no WebGL and mount on their
 * own): asset, fetchJson(url), urlState(defaults), loadStylesheet(name), and the formatting and fact
 * helpers below.
 *
 * FACT SPANS in page prose: <span data-viz-fact="STEM:dotted.path">1,234</span>, where STEM is a
 * data file under _static/viz/data/ without .json; add data-viz-source="explainer" for a file of
 * the explainer article (_static/explainer/data/), data-viz-scale="100" to show a fraction as a
 * percentage, and data-viz-match="prefix" for a commit hash. A number matches at the precision
 * shown: "170.9" with scale 1e-6 matches 170935601; thousands separators and a real minus sign are
 * allowed. The test checks every span; checkFacts() below repeats the check in the browser.
 */

import { fontsReady, onReducedMotion, onTheme, reducedMotion, theme } from './theme.js';
import { paintRange } from './controls.js';
import { take, unpack } from '../../../explainer/js/lib/data.js';

const hasDom = typeof document !== 'undefined';
const base = hasDom ? new URL('..', import.meta.url) : null;
const staticRoot = hasDom ? new URL('../', base) : null;
const MINUS = '−';

/** A URL under _static/viz/. */
export const asset = (path) => new URL(path, staticRoot).href;

/**
 * A number as text at a fixed precision, with thousands separators and a real minus sign.
 * @param {number} value
 * @param {{digits?: number, scale?: number, unit?: string, grouping?: boolean}} [options]
 *   the text shows value * scale to `digits` decimals, then a thin space and the unit
 */
export function formatNumber(value, { digits = 0, scale = 1, unit = '', grouping = true } = {}) {
  if (!Number.isFinite(value)) return '–';
  const v = value * scale;
  const text = Math.abs(v).toLocaleString('en-US', { minimumFractionDigits: digits, maximumFractionDigits: digits, useGrouping: grouping });
  const signed = v < 0 && Number(text.replace(/,/g, '')) !== 0 ? MINUS + text : text;
  return unit ? `${signed} ${unit}` : signed;
}

const SI = [
  [1e15, 'P'],
  [1e12, 'T'],
  [1e9, 'G'],
  [1e6, 'M'],
  [1e3, 'k'],
];
/** A count with an SI prefix and three significant digits: formatSI(3.2e9, 'FLOP') is '3.20 GFLOP'. */
export function formatSI(value, unit = '', significant = 3) {
  if (!Number.isFinite(value)) return '–';
  const [div, prefix] = SI.find(([d]) => Math.abs(value) >= d) || [1, ''];
  const scaled = value / div;
  const digits = Math.max(0, significant - 1 - Math.floor(Math.log10(Math.max(1e-12, Math.abs(scaled)))));
  return `${formatNumber(scaled, { digits, grouping: false })} ${prefix}${unit}`.trim();
}

/** Bytes in binary units (KiB, MiB, GiB, TiB), three significant digits. */
export function formatBytes(bytes) {
  const units = ['B', 'KiB', 'MiB', 'GiB', 'TiB', 'PiB'];
  let k = 0;
  let v = bytes;
  while (Math.abs(v) >= 1024 && k < units.length - 1) {
    v /= 1024;
    k++;
  }
  const digits = k === 0 ? 0 : Math.max(0, 2 - Math.floor(Math.log10(Math.max(1e-12, Math.abs(v)))));
  return `${formatNumber(v, { digits, grouping: false })} ${units[k]}`;
}

/** Resolve a dotted path ('a.b.3.c') in a document; list segments are indices. */
export function resolvePath(doc, dotted) {
  let value = doc;
  for (const segment of dotted.split('.')) {
    if (Array.isArray(value)) {
      const i = Number(segment);
      if (!Number.isInteger(i) || i < -value.length || i >= value.length) throw new Error(`${dotted}: no index ${segment}`);
      value = value[i < 0 ? value.length + i : i];
    } else if (value && typeof value === 'object' && segment in value) value = value[segment];
    else throw new Error(`${dotted}: no key ${segment}`);
  }
  return value;
}

/** Whether the text of a fact span shows `value` (the rule tests/test_viz_docs.py applies). */
export function factMatches(shown, value, { scale = 1, match = 'exact' } = {}) {
  if (typeof value !== 'number') {
    const text = String(value);
    return match === 'prefix' ? shown.length >= 7 && text.startsWith(shown) : shown === text;
  }
  const number = shown.match(/[-−]?\d[\d,  ]*(?:\.\d+)?/);
  if (!number) return false;
  const digits = number[0].replace('−', '-').replace(/[,  ]/g, '');
  const decimals = digits.includes('.') ? digits.split('.')[1].length : 0;
  return Math.abs(value * scale - Number(digits)) <= 0.5 * 10 ** -decimals * (1 + 1e-9) + 1e-12;
}

/**
 * Compare every fact span under `root` whose data file is loaded with its value; warn on the console.
 * @param {ParentNode} root
 * @param {Record<string, object>} docs  data documents by file stem
 * @returns {string[]}  the mismatches
 */
export function checkFacts(root, docs) {
  const problems = [];
  for (const span of root.querySelectorAll('[data-viz-fact]')) {
    const [stem, dotted] = span.dataset.vizFact.split(':');
    if (!docs[stem] || span.dataset.vizSource) continue;
    try {
      const value = resolvePath(docs[stem], dotted);
      const ok = factMatches(span.textContent.trim().replace(/\s+/g, ' '), value, { scale: Number(span.dataset.vizScale || 1), match: span.dataset.vizMatch || 'exact' });
      if (!ok) problems.push(`${span.dataset.vizFact}: shows ${span.textContent.trim()}, data ${value}`);
    } catch (error) {
      problems.push(`${span.dataset.vizFact}: ${error.message}`);
    }
  }
  if (problems.length) console.warn('viz facts that differ from their data:', problems);
  return problems;
}

/** Add css/NAME under _static/viz/ to the document once; resolves when it has loaded. */
export const loadStylesheet = (name) =>
  new Promise((resolve, reject) => {
    const href = new URL(`css/${name}`, staticRoot).href;
    if (document.querySelector(`link[rel="stylesheet"][href="${href}"]`)) return resolve();
    const link = document.createElement('link');
    link.rel = 'stylesheet';
    link.href = href;
    link.addEventListener('load', resolve, { once: true });
    link.addEventListener('error', () => reject(new Error(`stylesheet ${href} did not load`)), { once: true });
    document.head.append(link);
  });

/** Fetch and parse a JSON document; rejects on an HTTP error. */
export const fetchJson = async (url) => {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`${response.status} for ${response.url}`);
  return response.json();
};

/**
 * Page state in the query string: numbers stay numbers; defaults are left out of the URL. The same
 * store as ctx.urlState, for figures a page mounts outside its stage.
 * @param {Record<string, number | string>} defaults
 * @returns {{get: () => object, set: (partial: object) => void}}
 */
export function urlState(defaults) {
  const read = () => {
    const params = new URLSearchParams(location.search);
    const out = { ...defaults };
    for (const [key, fallback] of Object.entries(defaults)) {
      if (!params.has(key)) continue;
      const raw = params.get(key);
      out[key] = typeof fallback === 'number' ? (Number.isFinite(Number(raw)) ? Number(raw) : fallback) : raw;
    }
    return out;
  };
  return {
    get: read,
    set(partial) {
      const params = new URLSearchParams(location.search);
      for (const [key, value] of Object.entries(partial)) {
        if (value === defaults[key] || value === null || value === undefined) params.delete(key);
        else params.set(key, String(value));
      }
      const query = params.toString();
      try {
        history.replaceState(history.state, '', `${location.pathname}${query ? `?${query}` : ''}${location.hash}`);
      } catch {
        /* WebKit throws after 100 replaceState calls in 10 s (a fast drag): the link waits for the next change. */
      }
    },
  };
}

function note(stage, text) {
  let element = stage.querySelector(':scope > .viz-fallback-note');
  if (!element) {
    element = document.createElement('p');
    element.className = 'viz-fallback-note';
    stage.append(element);
  }
  element.textContent = text;
}

async function mountStage(stage, spec) {
  const original = [...stage.childNodes];
  stage.dataset.vizState = 'loading';
  try {
    const entries = Object.entries(spec.data || {});
    const [docs] = await Promise.all([
      Promise.all(entries.map(([, path]) => fetchJson(asset(path)))),
      Promise.all((spec.styles || []).map(loadStylesheet)),
      fontsReady(),
    ]);
    const data = Object.fromEntries(entries.map(([key], i) => [key, docs[i]]));
    const byStem = Object.fromEntries(entries.map(([, path], i) => [path.split('/').pop().replace(/\.json$/, ''), docs[i]]));
    const lookup = (key) => data[key] || byStem[key] || (() => { throw new Error(`no data "${key}"`); })();
    const fact = (key, dotted) => resolvePath(lookup(key), dotted);
    const ctx = {
      slug: spec.slug,
      stage,
      data,
      load: (path) => fetchJson(asset(path)),
      asset,
      theme,
      onTheme,
      reducedMotion,
      onReducedMotion,
      unpack,
      take,
      fact,
      factSpan(key, dotted, options = {}) {
        const span = document.createElement('span');
        const stem = Object.keys(byStem).find((s) => byStem[s] === lookup(key));
        span.dataset.vizFact = `${stem}:${dotted}`;
        if (options.scale) span.dataset.vizScale = String(options.scale);
        const value = fact(key, dotted);
        span.textContent = typeof value === 'number' ? formatNumber(value, options) : String(value);
        return span;
      },
      urlState,
      poster: new URLSearchParams(location.search).has('viz-poster'),
    };
    await spec.mount(stage, ctx);
    stage.querySelectorAll(':scope > img.viz-poster').forEach((poster) => poster.remove());
    stage.querySelectorAll('input[type="range"]').forEach(paintRange);
    stage.classList.add('viz-live');
    stage.dataset.vizState = 'live';
    checkFacts(document, byStem);
  } catch (error) {
    stage.replaceChildren(...original);
    stage.classList.add('viz-failed');
    stage.dataset.vizState = 'failed';
    note(stage, `${error.userMessage || 'The interactive version could not start.'} This is its static image.`);
    console.error(`visualisation "${spec.slug}" kept its static version:`, error);
  }
}

/**
 * Register a page module: mount every `.viz-stage[data-viz-page=slug]` of the document as it nears
 * the viewport. Without a document (node importing the module) it only returns the spec.
 * @param {{slug: string, styles?: string[], data?: Record<string, string>, mount: Function}} spec
 */
export function definePage(spec) {
  if (!hasDom) return spec;
  if (!document.documentElement.dataset.vizRanges) {
    document.documentElement.dataset.vizRanges = '1';
    document.addEventListener('input', (event) => event.target?.type === 'range' && paintRange(event.target));
  }
  const stages = [...document.querySelectorAll(`.viz-stage[data-viz-page="${spec.slug}"]`)].filter((s) => !s.hasAttribute('data-viz-pending'));
  const all = document.querySelectorAll('.viz-stage[data-viz-page]:not([data-viz-pending])').length;
  document.documentElement.dataset.vizReady = String(all);
  const nearby = new IntersectionObserver(
    (entries) => {
      for (const entry of entries) {
        if (!entry.isIntersecting) continue;
        nearby.unobserve(entry.target);
        mountStage(entry.target, spec);
      }
    },
    { rootMargin: '50% 0px' },
  );
  for (const stage of stages) {
    stage.dataset.vizState = 'waiting';
    nearby.observe(stage);
  }
  return spec;
}
