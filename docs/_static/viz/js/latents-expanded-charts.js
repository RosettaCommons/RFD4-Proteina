/* Measured rate concentration, decoder sensitivity and identity conditioning. */
import { el, svgEl, toggle, controlRow } from './lib/controls.js';

export function informationPlots(host, groups, examples, names, order, sigmas, onPick) {
  let mode = 'budget', channel = 3;
  const panels = el('div', { class: 'expanded-chart-panels' });
  const description = el('p', { class: 'viz-caption' });
  host.append(controlRow([toggle({ label: 'Explore', value: mode, options: [
    { value: 'budget', text: 'Rate budget' }, { value: 'sensitivity', text: 'Decoder sensitivity' },
    { value: 'noise', text: 'Noise tolerance' },
  ], onChange: (value) => { mode = value; draw(); } })]), panels, description);
  const text = (x, y, value, attrs = {}) => svgEl('text', { x, y, fill: 'var(--wf-ink-soft)', ...attrs }, value);
  const line = (x1, y1, x2, y2, attrs = {}) => svgEl('line', { x1, y1, x2, y2, stroke: 'var(--wf-line)', ...attrs });
  function draw() {
    panels.replaceChildren();
    const maxRate = Math.ceil(Math.max(...order.flatMap(g => groups[g].kl_nats)));
    const maxMove = Math.max(...order.flatMap(g => examples[g].ablation_displacement_A));
    const maxNoiseError = Math.ceil(Math.max(...order.flatMap(g =>
      [examples[g].rest.rmsd_true, ...examples[g].noise[32].flat().map(f => f.rmsd_true)])));
    for (const g of order) {
      const stats = groups[g];
      const sorted = stats.kl_nats.map((rate, c) => ({ rate, c })).sort((a, b) => b.rate - a.rate);
      let sum = 0;
      const cumulative = [0, ...sorted.map(({ rate }) => (sum += rate) / stats.total_kl_nats)];
      const half = cumulative.findIndex(v => v >= .5);
      const figure = el('figure', { class: 'expanded-small-plot' });
      figure.append(el('figcaption', {}, el('strong', {}, names[g]),
        el('span', {}, `${stats.total_kl_nats.toFixed(1)} nats/token · ${stats.active_units}/32 active`)));
      const svg = svgEl('svg', { viewBox: '0 0 310 190', role: 'img',
        'aria-label': mode === 'budget' ? `${names[g]}: ${half} highest-rate channels carry half the KL rate`
          : mode === 'noise' ? `${names[g]}: decoded atom-slot RMSD versus noise scale, three draws`
          : `${names[g]}: per-channel KL rate against atom movement after ablation` });
      const x0 = 38, x1 = 296, y0 = 148, y1 = 16;
      svg.append(line(x0, y1, x0, y0), line(x0, y0, x1, y0));
      if (mode === 'budget') {
        svg.append(line(x0, y0, x1, y1, { 'stroke-dasharray': '3 4' }));
        [0, .5, 1].forEach(v => {
          const y = y0 - v * (y0 - y1);
          svg.append(line(x0, y, x1, y), text(x0 - 7, y + 4, `${v * 100}%`, { 'text-anchor': 'end' }));
        });
        [0, 8, 16, 24, 32].forEach(n => svg.append(text(x0 + n / 32 * (x1 - x0), y0 + 17, n, { 'text-anchor': 'middle' })));
        svg.append(svgEl('path', { d: cumulative.map((v, i) => `${i ? 'L' : 'M'}${x0 + i / 32 * (x1 - x0)},${y0 - v * (y0 - y1)}`).join(' '), fill: 'none', stroke: 'var(--viz-lat-ink)', 'stroke-width': 2 }));
        const rank = sorted.findIndex(v => v.c === channel) + 1;
        svg.append(svgEl('circle', { cx: x0 + rank / 32 * (x1 - x0), cy: y0 - cumulative[rank] * (y0 - y1), r: 4, fill: 'var(--wf-ink)' }),
          text((x0 + x1) / 2, 185, 'Channels, largest rate first', { 'text-anchor': 'middle' }));
      } else if (mode === 'noise') {
        const scales = [0, ...sigmas];
        const traces = examples[g].noise[32].map(draw => [examples[g].rest.rmsd_true, ...draw.map(f => f.rmsd_true)]);
        const mean = scales.map((_, i) => traces.reduce((sum, trace) => sum + trace[i], 0) / traces.length);
        const x = v => x0 + v / scales.at(-1) * (x1 - x0);
        const y = v => y0 - v / maxNoiseError * (y0 - y1);
        [0, maxNoiseError / 2, maxNoiseError].forEach(v => {
          svg.append(line(x0, y(v), x1, y(v)), text(x0 - 7, y(v) + 4, v.toFixed(1), { 'text-anchor': 'end' }));
        });
        scales.forEach(v => svg.append(text(x(v), y0 + 17, v, { 'text-anchor': 'middle' })));
        const path = values => values.map((v, i) => `${i ? 'L' : 'M'}${x(scales[i])},${y(v)}`).join(' ');
        traces.forEach(values => svg.append(svgEl('path', { d: path(values), fill: 'none', stroke: 'var(--viz-lat-ink)', 'stroke-width': 1, opacity: .25 })));
        svg.append(svgEl('path', { d: path(mean), fill: 'none', stroke: 'var(--viz-lat-ink)', 'stroke-width': 2 }));
        mean.forEach((v, i) => {
          const dot = svgEl('circle', { cx: x(scales[i]), cy: y(v), r: 3.5, fill: 'var(--viz-lat-ink)',
            'data-noise-context': g, 'data-noise-sigma': scales[i], 'data-noise-rmsd': v });
          dot.append(svgEl('title', {}, `σ ${scales[i]}: mean ${v.toFixed(2)} Å over ${traces.length} draws`)); svg.append(dot);
        });
        svg.append(text((x0 + x1) / 2, 185, 'Noise σ, prior standard deviations', { 'text-anchor': 'middle' }), text(x0, 10, 'Atom-slot RMSD, Å'));
      } else {
        [0, .5, 1].forEach(v => {
          const y = y0 - v * (y0 - y1);
          svg.append(line(x0, y, x1, y), text(x0 - 7, y + 4, (v * maxMove).toFixed(1), { 'text-anchor': 'end' }));
        });
        [0, maxRate / 2, maxRate].forEach(v => svg.append(text(x0 + v / maxRate * (x1 - x0), y0 + 17, v.toFixed(1), { 'text-anchor': 'middle' })));
        stats.kl_nats.forEach((rate, c) => {
          const movement = examples[g].ablation_displacement_A[c];
          const active = stats.mean_variance[c] > .01;
          const dot = svgEl('circle', { cx: x0 + rate / maxRate * (x1 - x0), cy: y0 - movement / maxMove * (y0 - y1),
            r: c === channel ? 5 : 3.5, fill: active ? 'var(--viz-lat-ink)' : 'var(--wf-ink-faint)',
            stroke: c === channel ? 'var(--wf-ink)' : 'none', 'stroke-width': 2, tabindex: 0, role: 'button',
            'aria-label': `${names[g]}, channel ${c}: ${rate.toFixed(2)} nats, ${movement.toFixed(2)} Å movement`,
            'aria-pressed': String(c === channel) });
          dot.append(svgEl('title', {}, `Channel ${c}: ${rate.toFixed(2)} nats · ${movement.toFixed(2)} Å · ${active ? 'active' : 'inactive'}`));
          dot.addEventListener('click', () => onPick(c));
          dot.addEventListener('keydown', e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); onPick(c); } });
          svg.append(dot);
        });
        svg.append(text((x0 + x1) / 2, 185, 'KL rate, nats/token', { 'text-anchor': 'middle' }), text(x0, 10, 'Ablation movement, Å'));
      }
      figure.append(svg, el('p', { class: 'viz-caption' }, mode === 'budget'
        ? `${half} channels carry half the rate; ${stats.n_tokens.toLocaleString()} tokens measured.`
        : mode === 'noise' ? `${examples[g].names.join('–')}: one window, all 32 channels perturbed.`
        : `Channel ${channel}: ${stats.kl_nats[channel].toFixed(2)} nats · ${examples[g].ablation_displacement_A[channel].toFixed(2)} Å movement.`));
      panels.append(figure);
    }
    description.textContent = mode === 'budget'
      ? 'Curves accumulate KL from the largest-rate channels. The dashed diagonal is equal allocation; the dot locates the picked channel. Higher rate is not necessarily more usable information.'
      : mode === 'noise' ? 'Thin lines are three seeded draws; the thick line is their arithmetic mean, not a confidence interval. Each dot is a decoder evaluation; connecting lines guide the eye. Fixed anchors, identities and deposited slots. This is a noise-response curve, not a rate–distortion frontier or a molecular-validity score.'
      : 'Same axes in every panel. Click a dot to pick its channel above. Coral means Var(μ) > 0.01. Ablation zeros that channel across one input’s matching tokens; sensitivity is not a chemical-validity score.';
  }
  draw();
  return { set(c) { channel = c; draw(); } };
}

export function identityPlot(host, groups, names, order) {
  function draw() {
    const width = host.clientWidth || 660;
    const svg = svgEl('svg', { viewBox: `0 0 ${width} 310`, role: 'img', 'aria-label': 'Identity recovery with native conditioning and after clearing supplied identity' });
    const x0 = width < 480 ? 122 : 180, x1 = width - 25;
    for (const value of [0, 25, 50, 75, 100]) {
      const x = x0 + value / 100 * (x1 - x0);
      svg.append(svgEl('line', { x1: x, y1: 20, x2: x, y2: 264, stroke: 'var(--wf-line)' }),
        svgEl('text', { x, y: 287, 'text-anchor': 'middle', fill: 'var(--wf-ink-soft)' }, `${value}%`));
    }
    order.forEach((g, i) => {
      const s = groups[g], y = 32 + i * 44;
      const native = x0 + s.native_sequence_accuracy * (x1 - x0), cleared = x0 + s.cleared_sequence_accuracy * (x1 - x0);
      svg.append(svgEl('text', { x: x0 - 14, y: y + 5, 'text-anchor': 'end', fill: 'var(--wf-ink)' }, names[g]),
        svgEl('line', { x1: cleared, y1: y, x2: native, y2: y, stroke: 'var(--wf-ink-soft)', 'stroke-width': 2 }));
      for (const [x, value, label, fill] of [[native, s.native_sequence_accuracy, 'native', 'var(--viz-lat-ink)'], [cleared, s.cleared_sequence_accuracy, 'cleared', 'var(--wf-surface)']]) {
        const dot = svgEl('circle', { cx: x, cy: y, r: 5, fill, stroke: 'var(--viz-lat-ink)', 'stroke-width': 2 });
        dot.append(svgEl('title', {}, `${names[g]}, ${label}: ${(100 * value).toFixed(1)}%`)); svg.append(dot);
      }
    });
    host.replaceChildren(svg);
  }
  new ResizeObserver(draw).observe(host); draw();
}
