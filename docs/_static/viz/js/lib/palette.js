/* The shared palette of the visualisation pages, and colour helpers for canvases and WebGL.
 *
 * Every colour is a CSS custom property that viz.css defines for both docs themes; PALETTE below
 * holds the same values (tests/test_viz_docs.py checks that they agree), so a script can resolve a
 * token before the stylesheet has loaded. Surfaces, ink and lines are the docs' --wf-* values, and the
 * article and the Illustrated page use the same data colours, so one quantity has one colour everywhere.
 *
 * Colour names the tensor; ink measures the amount. Four hues are data identities:
 *
 *   pair        the pair track z (petrol): the darkest data colour in light, the heaviest block anywhere
 *   single      the single track s (saffron); light-mode marks under 6 px get a 1 px `single-ink` outline
 *   ca          Cα positions, velocities and contact ticks (cobalt); `designed` tokens share it
 *   lat         latents and changed residues (coral)
 *
 * Everything else keeps out of their way:
 *
 *   cond        the conditioning vector c (cool grey, always drawn dashed)
 *   recycle     recycling loops (deep moss): lines and loop arrows only, never an area
 *   given       tokens the designer gives (the docs' magenta; nothing else is magenta but the focus ring)
 *   context, weight, input   cool greys of distinct lightness: context tokens, weights, inputs
 *   ligand, nucleic          atomized ligand atoms (brown) and nucleotide tokens (violet)
 *   el-n, el-o, cat-0..3     stick-model N and O; secondary-structure and residue classes (ink, brown, violet, grey)
 *   attn        the attention mark: the attention ramp's darkest stop
 *   accent, accent-bg, select, hover   the interface is ink: pressed controls, sliders, selection rings
 *
 * Tones of a role: the base for lines, dots, chips and bars; `-fill` for treemap tiles and a bar's
 * second series (at full opacity); `-face` for 3D box faces (scene3d.js resolves a role to it);
 * `-tint` for washes; `-soft` for glyph cells; `-ink` for text on a tint; `-on-fill` for text on a fill.
 *
 * RAMPS are named, and pages ask for one by name: `sequential` (neutral ink) for magnitudes that belong
 * to no tensor; `attention` (neutral, flat below uniform attention, then three even steps) for every
 * probability; `diverging` (violet, neutral, green) for every signed value; `pair`, `single`, `ca`,
 * `lat` and `cond` for magnitudes of that quantity; `balance` (s, neutral, z) for a head's share of
 * content against pair bias. Every ramp takes even steps in OKLab. A view shows at most four data hues
 * plus magenta. SHADING is the key light and edge mix of the 3D scenes.
 */

/* The same values as viz.css; tests/test_viz_docs.py parses this object between the markers. */
// palette:begin
export const PALETTE = {
  "light": {
    "surface": "#ffffff", "surface-alt": "#fbfbfc", "ink": "#1f2328", "ink-soft": "#5b636d", "ink-faint": "#8b939d", "line": "#d6dae0", "line-strong": "#aeb5bd", "grid": "#eceef1",
    "pair-tint": "#eaf9fd", "pair-soft": "#cfecf3", "pair-fill": "#6097a4", "pair-face": "#367b89", "pair": "#004b58", "pair-ink": "#003d47", "pair-on-fill": "#1f2328",
    "single-tint": "#fdf5e7", "single-soft": "#f9e3bd", "single-fill": "#e9c176", "single-face": "#e1b255", "single": "#d19a12", "single-ink": "#5a4000", "single-on-fill": "#1f2328",
    "ca-tint": "#f2f6ff", "ca-soft": "#dce6ff", "ca-fill": "#87a3ec", "ca-face": "#6c8adf", "ca": "#4462c3", "ca-ink": "#273d8a", "ca-on-fill": "#1f2328",
    "lat-tint": "#fff3f0", "lat-soft": "#ffddd7", "lat-fill": "#ea9587", "lat-face": "#dc7b6c", "lat": "#c15344", "lat-ink": "#792218", "lat-on-fill": "#1f2328",
    "cond-tint": "#f4f6f8", "cond-soft": "#e4e6ea", "cond-fill": "#aaaeb5", "cond-face": "#959aa1", "cond": "#737981", "cond-ink": "#41454b", "cond-on-fill": "#1f2328",
    "recycle-tint": "#eff9ed", "recycle-soft": "#d5efd0", "recycle-fill": "#6e9b64", "recycle-face": "#4a7f3f", "recycle": "#145100", "recycle-ink": "#0e4100", "recycle-on-fill": "#1f2328",
    "ligand-tint": "#fff3ec", "ligand-soft": "#f7e1d4", "ligand-fill": "#bb957c", "ligand-face": "#a57a5d", "ligand": "#80512f", "ligand-ink": "#613a1f", "ligand-on-fill": "#1f2328",
    "nucleic-tint": "#f8f3ff", "nucleic-soft": "#eedfff", "nucleic-fill": "#a386c3", "nucleic-face": "#7c599e", "nucleic": "#4e2472", "nucleic-ink": "#3e1c5b", "nucleic-on-fill": "#1f2328",
    "given-tint": "#fff1f9", "given-soft": "#ffdaf0", "given-fill": "#de78bc", "given-face": "#cc55a7", "given": "#a80c84", "given-ink": "#7a005f", "given-on-fill": "#1f2328",
    "designed": "#4462c3", "context": "#8f949b", "weight": "#afb5bd", "input": "#d2dae3", "padding": "#fbfbfc", "el-n": "#294288", "el-o": "#862724",
    "cat-0": "#aeb5bd", "cat-1": "#1f2328", "cat-2": "#80512f", "cat-3": "#4e2472",
    "seq-0": "#f1f4f7", "seq-1": "#cacdd1", "seq-2": "#a4a7ac", "seq-3": "#7f8388", "seq-4": "#5c6066", "seq-5": "#3c4046", "seq-6": "#1e2228",
    "att-0": "#fbfcfe", "att-1": "#f2f4f6", "att-2": "#e5e8ec", "att-3": "#cdd1d7", "att-4": "#8e939a", "att-5": "#535860", "att-6": "#20242b",
    "div-0": "#ab8be3", "div-1": "#bda6ea", "div-2": "#cfc1f0", "div-3": "#e2dcf5", "div-4": "#f5f7f9", "div-5": "#caebda", "div-6": "#9ddebc", "div-7": "#69d19e", "div-8": "#00c380",
    "pair-0": "#ebf6f8", "pair-1": "#bcd0d5", "pair-2": "#8dacb3", "pair-3": "#618892", "pair-4": "#336672", "pair-5": "#004551", "pair-6": "#00242b",
    "single-0": "#faf3e5", "single-1": "#efdab3", "single-2": "#e3c07f", "single-3": "#d7a642", "single-4": "#c18e0e", "single-5": "#a37705", "single-6": "#876200",
    "ca-0": "#eff3ff", "ca-1": "#c3d0f2", "ca-2": "#99aee5", "ca-3": "#708bd6", "ca-4": "#4a68c6", "ca-5": "#31499c", "ca-6": "#1b2d6f",
    "lat-0": "#ffefed", "lat-1": "#f3c8c1", "lat-2": "#e4a297", "lat-3": "#d37b6d", "lat-4": "#c15344", "lat-5": "#983c30", "lat-6": "#71261d",
    "cond-0": "#f2f3f5", "cond-1": "#d0d3d7", "cond-2": "#b0b4b9", "cond-3": "#90959c", "cond-4": "#727880", "cond-5": "#575c62", "cond-6": "#3d4146",
    "attn": "#20242b", "accent": "#1f2328", "accent-bg": "#eceef1", "select": "#1f2328", "hover": "#1f2328", "focus": "#a80c84"
  },
  "dark": {
    "surface": "#1c2027", "surface-alt": "#222831", "ink": "#e9edf2", "ink-soft": "#a9b2bd", "ink-faint": "#79828d", "line": "#3a424c", "line-strong": "#59626d", "grid": "#272d35",
    "pair-tint": "#1c2f31", "pair-soft": "#1d4145", "pair-fill": "#006069", "pair-face": "#1c6c75", "pair": "#1794a1", "pair-ink": "#c0e6eb", "pair-on-fill": "#e9edf2",
    "single-tint": "#332a1a", "single-soft": "#48381a", "single-fill": "#b28235", "single-face": "#c49548", "single": "#f0c168", "single-ink": "#eedcbb", "single-on-fill": "#1c2027",
    "ca-tint": "#242b3b", "ca-soft": "#2b3a58", "ca-fill": "#4b69a8", "ca-face": "#6886c5", "ca": "#82aaff", "ca-ink": "#cedeff", "ca-on-fill": "#e9edf2",
    "lat-tint": "#392621", "lat-soft": "#542f26", "lat-fill": "#8e4333", "lat-face": "#a15746", "lat": "#d87059", "lat-ink": "#ffd2c7", "lat-on-fill": "#e9edf2",
    "cond-tint": "#2a2b2d", "cond-soft": "#383b3e", "cond-fill": "#81878d", "cond-face": "#b9bec6", "cond": "#b8bec7", "cond-ink": "#dbdee2", "cond-on-fill": "#1c2027",
    "recycle-tint": "#262f1f", "recycle-soft": "#304022", "recycle-fill": "#30500a", "recycle-face": "#335212", "recycle": "#4b781a", "recycle-ink": "#d1e6c3", "recycle-on-fill": "#e9edf2",
    "ligand-tint": "#302a25", "ligand-soft": "#433830", "ligand-fill": "#7b6554", "ligand-face": "#988170", "ligand": "#c3a48c", "ligand-ink": "#e8dbd1", "ligand-on-fill": "#e9edf2",
    "nucleic-tint": "#2d2838", "nucleic-soft": "#3f3453", "nucleic-fill": "#5c4482", "nucleic-face": "#66508c", "nucleic": "#8d6cc2", "nucleic-ink": "#e3d7fc", "nucleic-on-fill": "#e9edf2",
    "given-tint": "#352630", "given-soft": "#522b47", "given-fill": "#964680", "given-face": "#b2629b", "given": "#e879c9", "given-ink": "#fdceed", "given-on-fill": "#e9edf2",
    "designed": "#82aaff", "context": "#666b71", "weight": "#8d939a", "input": "#3a424a", "padding": "#222831", "el-n": "#bad1ff", "el-o": "#fbb7ae",
    "cat-0": "#59626d", "cat-1": "#e9edf2", "cat-2": "#c3a48c", "cat-3": "#8d6cc2",
    "seq-0": "#32363c", "seq-1": "#4d5157", "seq-2": "#6a6e74", "seq-3": "#898d92", "seq-4": "#a9acb1", "seq-5": "#cacdd1", "seq-6": "#eceff2",
    "att-0": "#22252b", "att-1": "#2a2e34", "att-2": "#353a40", "att-3": "#4c5056", "att-4": "#7f848a", "att-5": "#babec4", "att-6": "#eff2f6",
    "div-0": "#ac9acd", "div-1": "#8a7ea4", "div-2": "#69637d", "div-3": "#4a4958", "div-4": "#2d3135", "div-5": "#3a514a", "div-6": "#467360", "div-7": "#509677", "div-8": "#58bb8e",
    "pair-0": "#1b3235", "pair-1": "#205258", "pair-2": "#1f747e", "pair-3": "#2698a4", "pair-4": "#6cb6c0", "pair-5": "#a3d6dc", "pair-6": "#d8f5f8",
    "single-0": "#372c18", "single-1": "#5a4827", "single-2": "#7f6638", "single-3": "#a78649", "single-4": "#d0a75a", "single-5": "#f2ca80", "single-6": "#fbedd4",
    "ca-0": "#242e42", "ca-1": "#3a4b6d", "ca-2": "#51699c", "ca-3": "#6a8acd", "ca-4": "#84abff", "ca-5": "#b5ceff", "ca-6": "#e7efff",
    "lat-0": "#402620", "lat-1": "#6b3b31", "lat-2": "#995242", "lat-3": "#ca6954", "lat-4": "#e5917e", "lat-5": "#f3bdb0", "lat-6": "#ffe9e4",
    "cond-0": "#2c2e30", "cond-1": "#474a4e", "cond-2": "#65686d", "cond-3": "#84888f", "cond-4": "#a4a9b1", "cond-5": "#c7ccd3", "cond-6": "#edeff1",
    "attn": "#eff2f6", "accent": "#e9edf2", "accent-bg": "#2d333c", "select": "#e9edf2", "hover": "#e9edf2", "focus": "#e879c9"
  }
};
// palette:end

/** The CSS custom property of a role: `cssVar('pair')` is '--viz-pair'. */
const cssVar = (role) => `--viz-${role}`;

/** A role in one of its tones: `toned('pair', 'face')` is 'pair-face'; any other spec is returned as given. */
export const toned = (spec, tone) => (typeof spec === 'string' && `${spec}-${tone}` in PALETTE.light ? `${spec}-${tone}` : spec);

const series = (name, n = 7) => Array.from({ length: n }, (_, i) => `${name}-${i}`);
/** Colour ramps by name, as lists of palette tokens from low to high, spaced evenly. */
export const RAMPS = {
  sequential: series('seq'),
  attention: series('att'),
  diverging: series('div', 9),
  pair: series('pair'),
  single: series('single'),
  ca: series('ca'),
  lat: series('lat'),
  cond: series('cond'),
  balance: ['single', 'div-4', 'pair'],
};

/** The 3D scenes' lighting: ambient share, how far box edges pull toward the ink, and the key light. */
export const SHADING = { ambient: 0.72, edgeMix: 0.4, light: [-0.45, 0.8, 0.4] };

let probe = null;
const probeContext = () => {
  if (!probe) probe = document.createElement('canvas').getContext('2d');
  return probe;
};

/** Parse '#rgb', '#rrggbb', 'rgb(...)' or 'rgba(...)' into [r, g, b, a] with channels in 0..1, or null. */
export function parseColor(text) {
  const value = String(text).trim();
  let match = value.match(/^#([0-9a-f]{3}|[0-9a-f]{6}|[0-9a-f]{8})$/i);
  if (match) {
    let hex = match[1];
    if (hex.length === 3) hex = [...hex].map((c) => c + c).join('');
    const n = (i) => parseInt(hex.slice(i, i + 2), 16) / 255;
    return [n(0), n(2), n(4), hex.length === 8 ? n(6) : 1];
  }
  match = value.match(/^rgba?\(([^)]+)\)$/i);
  if (match) {
    const raw = match[1].split(/[\s,/]+/).filter(Boolean);
    const channel = (p) => (p.endsWith('%') ? parseFloat(p) / 100 : parseFloat(p) / 255);
    const parts = [...raw.slice(0, 3).map(channel), raw.length > 3 ? parseFloat(raw[3]) / (raw[3].endsWith('%') ? 100 : 1) : 1];
    if (raw.length < 3 || parts.some((p) => !Number.isFinite(p))) return null;
    return parts.map((p) => Math.min(1, Math.max(0, p)));
  }
  return null;
}

/**
 * Resolve a colour spec to [r, g, b, a] in 0..1.
 * @param {string | number[]} spec  a role ('pair'), a custom property ('--wf-ink'), any CSS colour, or
 *   [r, g, b(, a)] in 0..1
 * @param {Element} [element]  where custom properties are read (default <html>)
 * @param {'light' | 'dark'} [mode]  PALETTE entry used when the stylesheet does not define the role
 */
export function resolveColor(spec, element = null, mode = 'light') {
  if (Array.isArray(spec) || ArrayBuffer.isView(spec)) return [spec[0], spec[1], spec[2], spec.length > 3 ? spec[3] : 1];
  let text = String(spec ?? '').trim();
  const role = text in PALETTE.light ? text : text.startsWith('--viz-') ? text.slice(6) : null;
  if (role || text.startsWith('--')) {
    const name = role ? cssVar(role) : text;
    const fromCss = typeof getComputedStyle === 'function' ? getComputedStyle(element || document.documentElement).getPropertyValue(name).trim() : '';
    text = fromCss || (role && PALETTE[mode][role]) || '#000000';
  }
  const direct = parseColor(text);
  if (direct) return direct;
  if (typeof document === 'undefined') return [0, 0, 0, 1];
  const ctx = probeContext();
  ctx.fillStyle = '#000000';
  ctx.fillStyle = text;
  return parseColor(ctx.fillStyle) || [0, 0, 0, 1];
}

/** A resolved colour as '#rrggbb'. */
export function hex(spec, element = null, mode = 'light') {
  const [r, g, b] = resolveColor(spec, element, mode);
  return `#${[r, g, b].map((c) => Math.round(Math.min(1, Math.max(0, c)) * 255).toString(16).padStart(2, '0')).join('')}`;
}

/** Mix two [r, g, b(, a)] colours: f = 0 gives a, f = 1 gives b. */
export const mix = (a, b, f) => a.map((value, i) => value * (1 - f) + (b[i] ?? 1) * f);

/**
 * A colour ramp as a lookup table of `n` RGBA bytes per entry (Uint8Array of length 4n).
 * @param {string | Array<string | number[]>} ramp  a RAMPS name or a list of colour specs, spaced evenly
 * @param {object} [options]  element and mode as for resolveColor; n entries (default 256)
 */
export function rampTable(ramp, { element = null, mode = 'light', n = 256 } = {}) {
  const stops = (typeof ramp === 'string' ? RAMPS[ramp] : ramp).map((spec) => resolveColor(spec, element, mode));
  if (!stops || stops.length < 1) throw new Error(`unknown ramp ${ramp}`);
  const table = new Uint8Array(4 * n);
  for (let i = 0; i < n; i++) {
    const t = n === 1 ? 0 : (i / (n - 1)) * (stops.length - 1);
    const k = Math.min(stops.length - 2, Math.floor(t));
    const color = stops.length === 1 ? stops[0] : mix(stops[Math.max(0, k)], stops[Math.max(0, k) + 1], t - Math.max(0, k));
    for (let c = 0; c < 4; c++) table[4 * i + c] = Math.round(Math.min(1, Math.max(0, color[c])) * 255);
  }
  return table;
}
