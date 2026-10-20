/* One fragment of the latent-space instrument (page-latents.js) as a stick model in SVG: the perturbed window
 * as thick ink sticks, coral where an atom has moved more than MOVED_A from its unperturbed decode, the
 * unperturbed window as a thin ghost behind it, and the context residues as thin grey sticks. In the arrow
 * modes each window token also carries its arrow: the lab-frame direction its latent points in. Sticks are
 * drawn back to front and fade with depth. A mouse or pen drag turns the model; touch scrolls the page, so a
 * phone shows the model at its home orientation (the exporter's axes, the fragment's longest axis across).
 */

import { svgEl } from './lib/controls.js';
import { mix, resolveColor } from './lib/palette.js';
import { MOVED_A } from './latents-model.js';

const rgb = ([r, g, b]) => `rgb(${Math.round(r * 255)}, ${Math.round(g * 255)}, ${Math.round(b * 255)})`;
/** Length (Å) of an arrow of unperturbed length. */
const ARROW_A = 2.4;

const multiply = (a, b) => a.map((row) => [0, 1, 2].map((j) => row[0] * b[0][j] + row[1] * b[1][j] + row[2] * b[2][j]));
/** The view turn; export_latent_space.view_rotation is the same matrix, so the arrow modes turn in this frame. */
const turn = (yaw, pitch) => {
  const cy = Math.cos(yaw), sy = Math.sin(yaw), cp = Math.cos(pitch), sp = Math.sin(pitch);
  return multiply([[1, 0, 0], [0, cp, -sp], [0, sp, cp]], [[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]]);
};

/**
 * Draw one fragment into `host`.
 * @param {HTMLElement} host
 * @param {ReturnType<import('./latents-model.js').fragment>} frag
 * @param {{label: string}} options  the drawing's accessible name, before the state's description
 * @returns {{show: (state: object, options?: {arrows?: Array|null}) => void, draw: () => void, home: () => void}}
 */
export function moleculeView(host, frag, { label }) {
  const svg = svgEl('svg', { class: 'latents-molecule', role: 'img' });
  host.replaceChildren(svg);
  const cxyz = frag.context.positions;
  const axes = frag.axes;

  const rest = frag.rest.positions();
  const drawn = frag.rest.present();
  const points = [];
  for (let i = 0; i < rest.length; i += 3) if (drawn[i / 3]) points.push([rest[i], rest[i + 1], rest[i + 2]]);
  for (let i = 0; i < cxyz.length; i += 3) points.push([cxyz[i], cxyz[i + 1], cxyz[i + 2]]);
  const mean = [0, 1, 2].map((k) => points.reduce((s, p) => s + p[k], 0) / points.length);
  const reach = Math.max(...points.map((p) => Math.hypot(p[0] - mean[0], p[1] - mean[1], p[2] - mean[2])));
  const extent = [0, 1].map((k) => Math.max(...points.map((p) => Math.abs(axes[k].reduce((s, a, i) => s + a * (p[i] - mean[i]), 0)))));

  let yaw = 0, pitch = 0;
  let state = frag.rest;
  let arrows = null;

  function draw() {
    const width = host.clientWidth;
    if (!width) return;
    const height = Math.round(Math.min(300, Math.max(170, (width * extent[1]) / extent[0] + 24)));
    svg.setAttribute('width', String(width));
    svg.setAttribute('height', String(height));
    svg.setAttribute('viewBox', `0 0 ${width} ${height}`);
    const pad = 10;
    /* One scale for every orientation: the home view fills the box, and a turned model stays inside its reach. */
    const scale = Math.min((width / 2 - pad) / extent[0], (height / 2 - pad) / extent[1], (Math.min(width, height) / 2 - pad) / (0.8 * reach));
    const R = multiply(turn(yaw, pitch), axes);
    const project = (x, y, z) => {
      const d = [x - mean[0], y - mean[1], z - mean[2]];
      const p = R.map((row) => row[0] * d[0] + row[1] * d[1] + row[2] * d[2]);
      return [width / 2 + scale * p[0], height / 2 - scale * p[1], p[2] / reach];
    };

    const ground = resolveColor('surface', host);
    const colours = {
      moved: resolveColor('lat', host),
      still: resolveColor('ink', host),
      ghost: resolveColor('line-strong', host),
      context: resolveColor('context', host),
      arrow: resolveColor('lat-ink', host),
    };
    const at = state.positions();
    const moved = state.displacement();
    const bonds = state.bonds();
    const sticks = [];
    const add = (a, b, kind, w) => sticks.push({ a, b, kind, w, z: (a[2] + b[2]) / 2 });
    const W = (i) => project(at[3 * i], at[3 * i + 1], at[3 * i + 2]);
    const C = (i) => project(cxyz[3 * i], cxyz[3 * i + 1], cxyz[3 * i + 2]);
    for (const [a, b] of bonds.context) add(C(a), C(b), 'context', 1.4);
    if (state !== frag.rest) {
      const R0 = (i) => project(rest[3 * i], rest[3 * i + 1], rest[3 * i + 2]);
      for (const [a, b] of frag.rest.bonds().window) add(R0(a), R0(b), 'ghost', 1.2);
    }
    for (const [a, b] of bonds.window) add(W(a), W(b), moved[a] > MOVED_A || moved[b] > MOVED_A ? 'moved' : 'still', 2.8);
    for (const [a, k] of bonds.cross) add(W(a), C(k), moved[a] > MOVED_A ? 'moved' : 'still', 2.8);
    sticks.sort((p, q) => q.z - p.z);

    const parts = sticks.map(({ a, b, kind, w, z }) => {
      /* z runs from -1 (near) to 1 (far) over the fragment's reach: far sticks thin and fade toward the ground. */
      const far = Math.min(1, Math.max(0, (z + 1) / 2));
      const colour = rgb(mix(colours[kind], ground, (kind === 'moved' || kind === 'still' ? 0.35 : 0.5) * far));
      return svgEl('line', { x1: a[0].toFixed(1), y1: a[1].toFixed(1), x2: b[0].toFixed(1), y2: b[1].toFixed(1), stroke: colour, 'stroke-width': (w * (1.15 - 0.3 * far)).toFixed(2), 'stroke-linecap': 'round' });
    });
    for (const { atom, vector } of arrows ?? []) {
      const o = [at[3 * atom], at[3 * atom + 1], at[3 * atom + 2]];
      const a = project(...o);
      const b = project(o[0] + ARROW_A * vector[0], o[1] + ARROW_A * vector[1], o[2] + ARROW_A * vector[2]);
      const dx = b[0] - a[0], dy = b[1] - a[1], n = Math.hypot(dx, dy);
      const stroke = rgb(colours.arrow);
      parts.push(svgEl('line', { class: 'latents-arrow', x1: a[0].toFixed(1), y1: a[1].toFixed(1), x2: b[0].toFixed(1), y2: b[1].toFixed(1), stroke, 'stroke-width': 1.6, 'stroke-dasharray': '3 2' }));
      if (n > 4) {
        const ux = dx / n, uy = dy / n, h = 6;
        const tip = `${b[0].toFixed(1)},${b[1].toFixed(1)}`;
        const l = `${(b[0] - h * ux + 0.5 * h * uy).toFixed(1)},${(b[1] - h * uy - 0.5 * h * ux).toFixed(1)}`;
        const r = `${(b[0] - h * ux - 0.5 * h * uy).toFixed(1)},${(b[1] - h * uy + 0.5 * h * ux).toFixed(1)}`;
        parts.push(svgEl('polygon', { class: 'latents-arrow', points: `${tip} ${l} ${r}`, fill: stroke }));
      }
    }
    svg.replaceChildren(...parts);
  }

  let pending = 0;
  const schedule = () => {
    if (pending) return;
    pending = requestAnimationFrame(() => {
      pending = 0;
      draw();
    });
  };
  new ResizeObserver(schedule).observe(host);

  let drag = null;
  svg.addEventListener('pointerdown', (event) => {
    if (event.pointerType === 'touch' || event.button !== 0) return;
    drag = { x: event.clientX, y: event.clientY, yaw, pitch };
    svg.setPointerCapture(event.pointerId);
    svg.classList.add('is-turning');
  });
  svg.addEventListener('pointermove', (event) => {
    if (!drag) return;
    yaw = drag.yaw + (event.clientX - drag.x) * 0.01;
    pitch = Math.max(-1.4, Math.min(1.4, drag.pitch + (event.clientY - drag.y) * 0.01));
    schedule();
  });
  const end = () => {
    drag = null;
    svg.classList.remove('is-turning');
  };
  svg.addEventListener('pointerup', end);
  svg.addEventListener('pointercancel', end);
  svg.addEventListener('dblclick', () => home());

  function home() {
    yaw = 0;
    pitch = 0;
    schedule();
  }

  return {
    show(next, options = {}) {
      state = next;
      arrows = options.arrows ?? null;
      const unit = frag.units ?? (frag.group === 'protein' ? 'residues' : 'nucleotides');
      svg.setAttribute('aria-label', `${label}: ${state.rms.toFixed(2)} Å RMS displacement of the perturbed ${unit}, ${state.max.toFixed(2)} Å at most; ${state.rmsdTrue.toFixed(2)} Å from the deposited structure.`);
      draw();
    },
    draw,
    home,
  };
}
