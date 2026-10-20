/* Element helpers and the parts of a page's one instrument, styled by ../arch.css and viz.css.
 *
 * A page has at most one instrument with one or two controls; every other figure is a drawing whose
 * marks answer the pointer. Each control is a native input or button with a label (visible, or for
 * screen readers only), so it works with a keyboard and a screen reader; values that change are
 * announced through the figure's one polite <output>. Nothing here starts playing by itself.
 *
 *   el(tag, attributes, ...children), svgEl(...)       create an HTML or SVG element (null attributes skipped)
 *   uniqueId(prefix), paintRange(input)                an element id; fill a slider's track up to its thumb
 *   range({label, values | min/max/step, value, format, onInput, bare})
 *       a slider; `values` steps through a list (token lengths 64..2048). `bare` (the instrument's main
 *       slider) hides its label and value from view: the readout states the value in words
 *   toggle({label, name, options: [{value, text}], value, onChange})
 *       quiet tabs, one pressed: a short discrete choice (the atlas's batch sizes 1..32)
 *   button(text, onClick, {label})                      a quiet text button
 *   playButton({label, onToggle, keys})                 the play/pause icon button (aria-pressed)
 *   controlRow(items)                                   the centred row under a drawing
 *   readout()                                           the figure's one-line aria-live <output>
 *   rampLegend({ramp, label, ticks})                    a colour-ramp bar with at most three ticks
 */

import { RAMPS } from './palette.js';

let serial = 0;

/** A unique element id with a readable prefix. */
export const uniqueId = (prefix = 'viz') => `${prefix}-${++serial}`;

/** Create an HTML element: `el('button', {type: 'button'}, 'Next')`. */
export function el(tag, attributes = {}, ...children) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(attributes)) {
    if (value === null || value === undefined || value === false) continue;
    node.setAttribute(key, value === true ? '' : String(value));
  }
  node.append(...children.flat().filter((child) => child !== null && child !== undefined));
  return node;
}

/** The same for an SVG element (circle, path, text, ...): a string child becomes a text node. */
export function svgEl(tag, attributes = {}, ...children) {
  const node = document.createElementNS('http://www.w3.org/2000/svg', tag);
  for (const [key, value] of Object.entries(attributes)) {
    if (value === null || value === undefined || value === false) continue;
    node.setAttribute(key, value === true ? '' : String(value));
  }
  node.append(...children.flat().filter((child) => child !== null && child !== undefined));
  return node;
}

/** Paint a slider's track up to its thumb (the --arch-p stop of arch.css's gradient). */
export function paintRange(input) {
  const min = Number(input.min || 0), max = Number(input.max || 100);
  const share = max > min ? (Number(input.value) - min) / (max - min) : 0;
  input.style.setProperty('--arch-p', `${(100 * Math.min(1, Math.max(0, share))).toFixed(2)}%`);
}

const valueOf = typeof HTMLInputElement === 'undefined' ? null : Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value');
/** Keep a slider's fill in step with its value, whether a reader or a script moves it. */
function trackRange(input) {
  Object.defineProperty(input, 'value', {
    configurable: true,
    get: () => valueOf.get.call(input),
    set: (next) => {
      valueOf.set.call(input, next);
      paintRange(input);
    },
  });
  input.addEventListener('input', () => paintRange(input));
  paintRange(input);
}

/**
 * A slider with its label and value.
 * @param {object} spec
 * @param {string} spec.label  its accessible name, and the visible label unless `bare`
 * @param {Array<number|string>} [spec.values]  discrete values in order (the slider steps through them)
 * @param {number} [spec.min], [spec.max], [spec.step]  a continuous range instead
 * @param {number|string} [spec.value]
 * @param {(value) => string} [spec.format]  the visible and spoken value
 * @param {(value) => void} [spec.onInput]
 * @param {boolean} [spec.bare]  label and value for screen readers only (the figure's main slider)
 * @returns {{element: HTMLElement, input: HTMLInputElement, value: () => any, set: (value, notify?) => void}}
 */
export function range({ label, values = null, min = 0, max = 1, step = 0.01, value, format = String, onInput = () => {}, bare = false }) {
  const id = uniqueId('viz-range');
  const discrete = Array.isArray(values);
  const toIndex = (v) => Math.max(0, values.indexOf(v));
  const input = el('input', {
    type: 'range',
    id,
    min: discrete ? 0 : min,
    max: discrete ? values.length - 1 : max,
    step: discrete ? 1 : step,
    value: discrete ? toIndex(value ?? values[0]) : value ?? min,
  });
  const shown = el('span', { class: `viz-range-value arch-field-value${bare ? ' arch-sr-only' : ''}`, 'aria-hidden': 'true' });
  const current = () => (discrete ? values[Number(input.value)] : Number(input.value));
  const paint = () => {
    const text = format(current());
    shown.textContent = text;
    input.setAttribute('aria-valuetext', text);
  };
  trackRange(input);
  input.addEventListener('input', () => {
    paint();
    onInput(current());
  });
  paint();
  const element = el(
    'div',
    { class: 'viz-range arch-field arch-field-range' },
    el('label', { for: id, class: `arch-field-label${bare ? ' arch-sr-only' : ''}` }, label),
    el('span', { class: 'arch-field-control' }, input, shown),
  );
  return {
    element,
    input,
    value: current,
    set(next, notify = false) {
      input.value = String(discrete ? toIndex(next) : next);
      paint();
      if (notify) onInput(current());
    },
  };
}

/**
 * Quiet tabs of which one is pressed (aria-pressed); the arrow keys move between them. Each is a text
 * button with no border or fill; the pressed one is ink with a 2 px bar under it (arch.css .arch-tabs).
 * `label` is printed before the tabs; `name`, or else `label`, is the group's accessible name.
 * @param {object} spec
 * @param {string} spec.label
 * @param {string} [spec.name]
 * @param {Array<{value: string, text: string}>} spec.options
 * @param {string} [spec.value]  the pressed option (the first by default)
 * @param {(value: string) => void} [spec.onChange]  called when a reader presses another option
 * @returns {{element: HTMLElement, value: () => string, set: (value: string, notify?: boolean) => void}}
 */
export function toggle({ label, name = label, options, value, onChange = () => {} }) {
  const group = el('div', { class: 'arch-tabs', role: 'group', 'aria-label': name });
  let current = value ?? options[0].value;
  const set = (next, notify = false) => {
    current = next;
    options.forEach((option, i) => buttons[i].setAttribute('aria-pressed', String(option.value === next)));
    if (notify) onChange(next);
  };
  const buttons = options.map((option) => {
    const b = el('button', { type: 'button', 'aria-pressed': String(option.value === current) }, option.text);
    b.addEventListener('click', () => option.value !== current && set(option.value, true));
    group.append(b);
    return b;
  });
  group.addEventListener('keydown', (event) => {
    const i = buttons.indexOf(document.activeElement);
    if (i < 0 || !['ArrowLeft', 'ArrowRight'].includes(event.key)) return;
    event.preventDefault();
    const j = Math.min(buttons.length - 1, Math.max(0, i + (event.key === 'ArrowRight' ? 1 : -1)));
    buttons[j].focus();
    if (options[j].value !== current) set(options[j].value, true);
  });
  const text = el('span', { class: 'arch-field-label', 'aria-hidden': 'true' }, label);
  const element = el('div', { class: 'arch-field' }, text, group);
  return { element, value: () => current, set };
}

/** A quiet text button. */
export function button(text, onClick, { label = null } = {}) {
  const b = el('button', { type: 'button', class: 'arch-button', 'aria-label': label }, text);
  b.addEventListener('click', onClick);
  return b;
}

/**
 * The play/pause button: a drawn icon on a grey square, aria-pressed while playing. `onToggle(playing)`
 * runs on a click; `set(playing)` shows the state without calling it (for playback that ends by itself).
 * @param {{label?: string, onToggle?: (playing: boolean) => void, keys?: string}} [spec]  keys: aria-keyshortcuts
 * @returns {{element: HTMLButtonElement, playing: () => boolean, set: (playing: boolean) => void}}
 */
export function playButton({ label = 'Play', onToggle = () => {}, keys = null } = {}) {
  const element = el('button', { type: 'button', class: 'arch-play', 'aria-label': label, 'aria-pressed': 'false', title: keys ? `${label} (${keys})` : label, 'aria-keyshortcuts': keys });
  let playing = false;
  const set = (next) => {
    playing = Boolean(next);
    element.setAttribute('aria-pressed', String(playing));
  };
  element.addEventListener('click', () => {
    set(!playing);
    onToggle(playing);
  });
  return { element, playing: () => playing, set };
}

/**
 * The control row under a drawing, its items centred (play, then the one slider). Posters omit the row.
 * @param {Array<HTMLElement | {element: HTMLElement}>} items
 */
export function controlRow(items) {
  const row = el('div', { class: 'arch-row' }, ...items.map((item) => item.element || item));
  return el('div', { class: 'arch-controls', 'data-viz-poster-omit': 'collapse' }, row);
}

/** The figure's readout: one line, announced politely; write a short phrase to it when the view changes. */
export const readout = () => el('output', { class: 'arch-readout', 'aria-live': 'polite' });

/**
 * A colour-ramp legend: the quantity named above an 8 px bar, and at most three ticks under it.
 * @param {{ramp: string | string[], label: string, ticks?: Array<string | {at: number, text: string}>, width?: number}} spec
 *   ramp: a RAMPS name or a list of palette tokens; ticks as texts spread over the bar (min, middle, max)
 *   or {at, text} with at in 0..1 (the ▲ uniform mark of an attention ramp, say)
 */
export function rampLegend({ ramp, label, ticks = [], width = 200 }) {
  const stops = (typeof ramp === 'string' ? RAMPS[ramp] : ramp).map((token) => `var(--viz-${token})`);
  if (!stops.length) throw new Error(`unknown ramp ${ramp}`);
  if (ticks.length > 3) throw new Error('a ramp legend carries at most three ticks');
  const bar = el('span', { class: 'arch-ramp-bar', 'aria-hidden': 'true', style: `--arch-ramp: linear-gradient(to right, ${stops.join(', ')})` });
  const placed = ticks.map((tick, i) => (typeof tick === 'string' ? { at: ticks.length === 1 ? 0.5 : i / (ticks.length - 1), text: tick } : tick));
  const marks = placed.map(({ at, text }) =>
    el('span', { class: at <= 0 ? 'arch-ramp-start' : at >= 1 ? 'arch-ramp-end' : null, style: `left: ${(100 * at).toFixed(2)}%` }, text),
  );
  return el('div', { class: 'arch-ramp', style: `width: ${width}px` }, el('span', {}, label), bar, el('div', { class: 'arch-ramp-ticks' }, ...marks));
}
