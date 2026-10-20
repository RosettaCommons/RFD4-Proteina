/* The docs theme and the reader's motion preference, for the visualisation pages.
 *
 * pydata-sphinx-theme sets `data-theme` on <html> after the first paint on a first visit; until
 * then `data-mode` decides, and the system preference decides the automatic mode (as the explainer
 * and the explorer do). One MutationObserver serves every listener. Importing this module has no
 * side effects, so node can import it (and the modules built on it) to check their exports.
 */

const hasDom = typeof document !== 'undefined' && typeof matchMedia === 'function';
const darkQuery = hasDom ? matchMedia('(prefers-color-scheme: dark)') : null;
const motionQuery = hasDom ? matchMedia('(prefers-reduced-motion: reduce)') : null;

/** 'light' or 'dark', as the docs theme currently shows. */
export function theme() {
  if (!hasDom) return 'light';
  const { theme: resolved, mode } = document.documentElement.dataset;
  if (resolved === 'dark' || resolved === 'light') return resolved;
  if (mode === 'dark' || mode === 'light') return mode;
  return darkQuery.matches ? 'dark' : 'light';
}

const themeListeners = new Set();
let shown = null;
let watching = false;
const follow = () => {
  const wanted = theme();
  if (wanted === shown) return;
  shown = wanted;
  for (const listener of [...themeListeners]) listener(wanted);
};

/**
 * Call `listener(theme)` after every change of the docs theme.
 * @param {(theme: 'light' | 'dark') => void} listener
 * @returns {() => void} unsubscribe
 */
export function onTheme(listener) {
  themeListeners.add(listener);
  if (hasDom && !watching) {
    watching = true;
    shown = theme();
    new MutationObserver(follow).observe(document.documentElement, {
      attributes: true,
      attributeFilter: ['data-theme', 'data-mode'],
    });
    darkQuery.addEventListener('change', follow);
  }
  return () => themeListeners.delete(listener);
}

/** True when the reader asks for reduced motion: jump, do not tween. */
export const reducedMotion = () => Boolean(motionQuery?.matches);

/**
 * Call `listener(reduced)` when the motion preference changes.
 * @returns {() => void} unsubscribe
 */
export function onReducedMotion(listener) {
  if (!motionQuery) return () => {};
  const handler = () => listener(motionQuery.matches);
  motionQuery.addEventListener('change', handler);
  return () => motionQuery.removeEventListener('change', handler);
}

/**
 * A canvas font in the story pages' type, from arch.css's --arch-sans or --arch-mono read on `element`:
 * `canvasFont(el, {size: 12})` is '400 12px "IBM Plex Sans", system-ui, …'. Use 14 (labels) or 12 (ticks).
 * @param {Element | null} [element]  where the custom properties are read (default <body>)
 * @param {{size?: number, weight?: number, italic?: boolean, mono?: boolean}} [options]  size in CSS px
 */
export function canvasFont(element = null, { size = 14, weight = 400, italic = false, mono = false } = {}) {
  const read = hasDom ? getComputedStyle(element || document.body).getPropertyValue(mono ? '--arch-mono' : '--arch-sans').trim() : '';
  return `${italic ? 'italic ' : ''}${weight} ${size}px ${read || (mono ? 'ui-monospace, monospace' : 'system-ui, sans-serif')}`;
}

/**
 * Resolves once the story pages' faces have loaded, or after `timeout` ms, whichever is first: a canvas
 * draws with whatever face is ready and does not redraw when a web font arrives.
 */
export function fontsReady(timeout = 2000) {
  if (!hasDom || !document.fonts) return Promise.resolve();
  const faces = ['400 1em', '600 1em', 'italic 400 1em'].map((f) => document.fonts.load(`${f} "IBM Plex Sans"`));
  faces.push(document.fonts.load('400 1em "IBM Plex Mono"'));
  return Promise.race([Promise.all(faces).catch(() => {}), new Promise((done) => setTimeout(done, timeout))]);
}
