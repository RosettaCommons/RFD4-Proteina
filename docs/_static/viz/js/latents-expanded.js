/* The expanded AE uses the reference instrument's controls, stick renderer and visual language.
 * Its states are separately decoded by export_expanded_walks.py; z16 arrow roles do not transfer. */
import { definePage } from './lib/page.js';
import { controlRow, el, range, toggle, button } from './lib/controls.js';
import { informationPlots, identityPlot } from './latents-expanded-charts.js';
import { moleculeView } from './latents-molecule.js';
import { rampTable } from './lib/palette.js';

const GROUPS = ['aa', 'dna', 'rna', 'ptm', 'ncaa', 'glycan'];
const NAMES = { aa: 'Protein', dna: 'DNA', rna: 'RNA', ptm: 'PTMs', ncaa: 'Noncanonical AAs', glycan: 'Glycans' };
const fmt = (n) => n.toFixed(2);
// Manuscript style/figure_theme.json: sequential palette used in Figures 4 and 5.
const HEATMAP_COLORS = rampTable(['#CFF6EF', '#32B39B', '#186C70', '#124F52'], { n: 7 });

function fragment(example, group) {
  const rest = example.rest.xyz.flat();
  for (const rows of [example.shift, ...example.noise]) for (const row of rows) for (const frame of row) {
    const bytes = Uint8Array.from(atob(frame.xyz_delta), (c) => c.charCodeAt(0));
    const offsets = new DataView(bytes.buffer);
    frame.xyz = example.rest.xyz.map((p, i) => p.map((v, k) => v + .01 * offsets.getInt16(2 * (3*i+k), true)));
  }
  const wrap = (frame) => {
    const positions = frame.xyz.flat();
    const displacement = frame.xyz.map((p, i) => Math.hypot(...p.map((v, k) => v - example.rest.xyz[i][k])));
    return { positions: () => positions, present: () => frame.xyz.map(() => true),
      displacement: () => displacement, bonds: () => example.bonds, rms: frame.rms,
      max: Math.max(...displacement), rmsdTrue: frame.rmsd_true };
  };
  return { group: group === 'aa' ? 'protein' : group, units: 'tokens', axes: example.axes,
    context: { positions: example.context_xyz.flat() }, rest: wrap(example.rest), wrap, positions: rest };
}

export default definePage({
  slug: 'expanded', styles: ['latents.css', 'expanded.css'],
  async mount(stage, ctx) {
    const response = await fetch(ctx.asset('expanded-data/expanded-walks.json.gz'));
    if (!response.ok) throw Error('Could not load decoded states');
    const walks = await new Response(response.body.pipeThrough(new DecompressionStream('gzip'))).json();
    const state = { channel: 5, mode: 'shift', shift: 1, sigma: 1, draw: 0, all: false };
    const title = el('span', { class: 'arch-title-name' });
    const claim = el('span');
    const heat = el('div', { class: 'expanded-grid', role: 'group', 'aria-label': 'Pick a latent channel' });
    const rows = [];
    const top = Math.max(...GROUPS.flatMap((g) => walks.groups[g].kl_nats));
    heat.append(el('span', {}), ...Array.from({ length: 32 }, (_, c) => el('span', { class: 'expanded-column' }, String(c))));
    for (const group of GROUPS) {
      heat.append(el('span', { class: 'expanded-row-name' }, NAMES[group]));
      walks.groups[group].kl_nats.forEach((value, channel) => {
        const b = el('button', { type: 'button', class: 'expanded-cell',
          'aria-label': `${NAMES[group]}, channel ${channel}, ${fmt(value)} nats`, 'aria-pressed': 'false',
          title: `${NAMES[group]} · channel ${channel} · ${fmt(value)} nats` });
        const color = 4 * Math.min(6, Math.round(value / top * 6));
        b.style.background = `rgb(${HEATMAP_COLORS.subarray(color, color + 3).join(',')})`;
        b.addEventListener('click', () => { state.channel = channel; update(); });
        rows.push({ b, channel }); heat.append(b);
      });
    }
    const views = el('div', { class: 'latents-views' });
    const models = {}, readouts = {}, sequences = {}, panes = {}, fragments = {};
    for (const group of GROUPS) {
      const example = walks.examples[group];
      const host = el('div', { class: 'latents-view-model' });
      readouts[group] = el('span', { class: 'latents-view-value' });
      sequences[group] = el('span', { class: 'latents-view-seq' });
      panes[group] = el('div', { class: 'latents-view' },
        el('p', { class: 'latents-view-head' }, el('span', { class: 'latents-view-name' }, NAMES[group]),
          el('span', { class: 'latents-view-what' }, ` · ${example.names.join('–')}`)), host, readouts[group], sequences[group]);
      views.append(panes[group]); fragments[group] = fragment(example, group);
      models[group] = moleculeView(host, fragments[group], { label: `${NAMES[group]}, ${example.names.join(', ')}` });
    }
    const mode = toggle({ label: 'Perturb', options: [
      { value: 'shift', text: 'Shift a channel' }, { value: 'noise', text: 'Add noise' }],
      value: state.mode, onChange: (v) => { stop(); state.mode = v; update(); } });
    const shift = range({ label: 'Shift, in prior standard deviations', values: walks.shifts, value: state.shift,
      format: (v) => `${v > 0 ? '+' : ''}${v} SD`, onInput: (v) => { stop(); state.shift = v; update(); } });
    const sigma = range({ label: 'Noise, in prior standard deviations', values: [0, ...walks.sigmas], value: state.sigma,
      format: (v) => `σ ${v}`, onInput: (v) => { state.sigma = v; update(); } });
    const noiseTarget = toggle({ label: 'Noise on', options: [
      { value: 'one', text: 'the picked channel' }, { value: 'all', text: 'all 32' }], value: 'one',
      onChange: (v) => { state.all = v === 'all'; update(); } });
    const draw = toggle({ label: 'Draw', options: [0, 1, 2].map((v) => ({ value: String(v), text: String(v + 1) })), value: '0',
      onChange: (v) => { state.draw = Number(v); update(); } });
    let timer = null;
    const play = button('Walk the channel', () => {
      if (timer) return stop();
      play.textContent = 'Pause';
      timer = setInterval(() => { state.shift = walks.shifts[(walks.shifts.indexOf(state.shift) + 1) % walks.shifts.length]; shift.set(state.shift); update(); }, 450);
    });
    function stop() { clearInterval(timer); timer = null; play.textContent = 'Walk the channel'; }
    document.addEventListener('visibilitychange', () => { if (document.hidden) stop(); });
    const shiftRow = controlRow([shift, play]);
    const noiseRow = controlRow([noiseTarget, sigma, draw]);
    const description = el('p', { class: 'latents-roles', 'aria-live': 'polite' });
    const app = el('div', { class: 'latents-app', 'data-viz-poster-region': '' },
      el('p', { class: 'arch-title' }, title, ' ', claim), el('div', { class: 'expanded-grid-scroll' }, heat),
      el('p', { class: 'viz-caption expanded-key' }, `KL to the prior · 0–${fmt(top)} nats per token. Click a column to compare it.`),
      el('div', { class: 'latents-controls' }, controlRow([mode]), shiftRow, noiseRow), views, description);
    stage.append(app);
    stage.querySelector(":scope > .viz-fallback-note")?.remove();
    const plots = informationPlots(document.querySelector('#expanded-rate-plots'), walks.groups, walks.examples, NAMES, GROUPS, walks.sigmas, (c) => { state.channel = c; update(); });
    identityPlot(document.querySelector('#expanded-identity-plot'), walks.groups, NAMES, GROUPS);
    const stats = document.querySelector('#expanded-channel-stats');
    function current(g) {
      const ex = walks.examples[g];
      if (state.mode === 'shift') return ex.shift[state.channel][walks.shifts.indexOf(state.shift)];
      return state.sigma === 0 ? ex.rest : ex.noise[state.all ? 32 : state.channel][state.draw][walks.sigmas.indexOf(state.sigma)];
    }
    function update() {
      heat.style.setProperty('--expanded-selected-column', String(state.channel + 2));
      for (const { b, channel } of rows) b.setAttribute('aria-pressed', String(channel === state.channel));
      for (const g of GROUPS) {
        const frame = current(g), example = walks.examples[g];
        models[g].show(fragments[g].wrap(frame));
        readouts[g].textContent = `moves ${fmt(frame.rms)} Å · ${fmt(frame.rmsd_true)} Å from deposited slots`;
        sequences[g].replaceChildren(...frame.types.map((t, i) => el('span', { class: t !== example.names[i] ? 'is-changed' : null }, `${t} `)),
          el('span', { class: 'latents-view-line latents-view-note' }, example.conditioned.every(Boolean) ? 'Identity supplied to the decoder' : 'Identity not supplied for every token'));
      }
      shiftRow.hidden = state.mode !== 'shift'; noiseRow.hidden = state.mode !== 'noise';
      title.textContent = state.mode === 'shift' ? `Channel ${state.channel}, shifted ${state.shift > 0 ? '+' : ''}${state.shift} SD`
        : `Noise of ${state.sigma} SD on ${state.all ? 'all 32 channels' : `channel ${state.channel}`}`;
      claim.textContent = 'The same intervention, decoded in six molecular contexts.';
      description.textContent = 'Drag a fragment to turn it. Teal marks atoms moved by more than 0.2 Å; thin sticks show the unperturbed decode and nearby context.';
      if (stats) stats.replaceChildren(...GROUPS.map((g) => {
        const s = walks.groups[g], c = state.channel;
        return el('tr', {}, el('th', { scope: 'row' }, NAMES[g]), el('td', {}, fmt(s.kl_nats[c])),
          el('td', {}, fmt(s.mean_variance[c])), el('td', {}, s.mean_variance[c] > .01 ? 'yes' : 'no'),
          el('td', {}, fmt(walks.examples[g].ablation_displacement_A[c])));
      }));
      plots.set(state.channel);
      document.querySelectorAll('[data-expanded-channel]').forEach((e) => { e.textContent = state.channel; });
    }
    ctx.onTheme(() => { for (const g of GROUPS) models[g].draw(); });
    update();
  },
});
