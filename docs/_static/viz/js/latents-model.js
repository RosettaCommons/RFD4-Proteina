/* The latent-space page's numbers (docs/architecture/viz/latents.md), read from latents-space.json,
 * latents-perturb.json, latents-noise.json and latents-arrows.json
 * (docs/architecture/viz/tools/export_latent_space.py). Pure functions without a document, so
 * tests/test_viz_latents.py imports this module in node.
 *
 * A token's latent is 16 channels, each a Gaussian posterior N(mu, sigma^2) with the KL cost
 * 0.5 (sigma^2 + mu^2 - 1) - ln sigma nats against the prior N(0, 1). The instrument's fragments store the
 * decode at the encoder means and one offset per exactly decoded state; every atom of a window token is a
 * slot of some residue type the token decodes to, shown while the token decodes to a type that has the slot.
 */

import { take, unpack } from '../../explainer/js/lib/data.js';

export const GROUPS = ['protein', 'dna', 'rna'];
export const GROUP_NAMES = { protein: 'Protein', dna: 'DNA', rna: 'RNA' };
export const UNITS = { protein: 'residue', dna: 'nucleotide', rna: 'nucleotide' };
export const D = 16;
/** A channel is read when setting it to the prior's mean moves an atom class by more than this RMS (Å). */
export const READ_A = 0.05;
/** A bond is drawn as moved when one of its atoms is displaced by more than this (Å). */
export const MOVED_A = 0.2;
/** The instrument opens on the channel that is empty for proteins and full for nucleotides, shifted by one. */
export const DEFAULT = { channel: 12, delta: 1 };
/** Names of the arrows the exporter fits, as the figures print them. */
export const ARROW_NAMES = {
  side_chain: 'side chain',
  backbone: 'backbone',
  third: 'third arrow',
  base: 'base',
  phosphate: 'phosphate',
  sugar_frame: 'sugar frame',
};

/** KL of N(mu, sigma^2) from N(0, 1), in nats. */
export const kl = (mu, sigma) => 0.5 * (sigma * sigma + mu * mu - 1) - Math.log(sigma);

/** The spread of the means at which a channel of posterior width `sigma` costs `nats` on average (NaN if none). */
export function isoKlSpread(nats, sigma) {
  const s2 = 2 * (nats + Math.log(sigma)) + 1 - sigma * sigma;
  return s2 >= 0 ? Math.sqrt(s2) : NaN;
}

/** One molecule type's 16 channels: rates, width, spread, whether the decoder reads it, rotation share. */
export function channels(space, g) {
  const group = space.groups[g];
  return group.rates.channels.map((c, i) => {
    const moved = Math.max(...group.ablation.displacement_A[i]);
    return {
      index: i,
      kl: c.kl_nats,
      information: Math.max(0, c.information_nats),
      mismatch: c.prior_mismatch_nats,
      sigma: c.sigma_geo,
      spread: c.sd_mu,
      activity: c.activity,
      active: c.active,
      moved,
      read: moved > READ_A,
      invariant: group.rotation.invariant_share[i],
      percentiles: c.mu_percentiles_5_50_95,
    };
  });
}

/** Totals of one molecule type: nats per token, their split, and the counts of active and read channels. */
export function totals(space, g) {
  const r = space.groups[g].rates;
  const cs = channels(space, g);
  return {
    tokens: r.n_tokens,
    kl: r.kl_total_nats,
    information: cs.reduce((a, c) => a + c.information, 0),
    mismatch: cs.reduce((a, c) => a + c.mismatch, 0),
    active: cs.filter((c) => c.active).length,
    read: cs.filter((c) => c.read).length,
  };
}

/** The aggregate posterior q(z_c) of channel `c` on the exporter's grid: {x, q}. */
export function density(space, g, c) {
  const field = space.groups[g].rates.aggregate_density;
  return { x: space.density_grid, q: Array.from(take(unpack(field), c)) };
}

/** The standard normal density. */
export const prior = (x) => Math.exp(-0.5 * x * x) / Math.sqrt(2 * Math.PI);

/** The arrows of one molecule type: channels, name, the direction that fits best and its R^2. */
export function arrows(space, g) {
  return Object.entries(space.groups[g].rotation.arrows).map(([key, a]) => ({
    key,
    name: ARROW_NAMES[key] ?? key.replace(/_/g, ' '),
    channels: a.channels,
    direction: a.best_direction,
    r2: a.best_r2,
    r2ByType: a.best_r2_by_type,
    frameR2: a.frame_r2,
  }));
}

/** Quantiles of `values` (sorted in place) at the fractions `ps`, by linear interpolation. */
export function quantiles(values, ps) {
  values.sort((a, b) => a - b);
  return ps.map((p) => {
    const t = p * (values.length - 1);
    const k = Math.floor(t);
    return k + 1 < values.length ? values[k] + (values[k + 1] - values[k]) * (t - k) : values[k];
  });
}

/** The length of the 1-4-10 arrow per residue type: 5th, 25th, 50th, 75th and 95th percentiles and counts. */
export function identityArrow(space, g) {
  const a = space.groups[g].rotation.identity_arrow;
  const length = unpack(a.length).values;
  const residue = unpack(a.residue).values;
  const rows = a.residues.map((name, k) => {
    const values = [];
    for (let i = 0; i < length.length; i++) if (residue[i] === k) values.push(length[i]);
    return { name, n: values.length, q: values.length ? quantiles(values, [0.05, 0.25, 0.5, 0.75, 0.95]) : null };
  });
  return { channels: a.channels, rows: rows.filter((r) => r.n), eta2: space.groups[g].rotation.identity_from_arrow_length };
}

/**
 * What each channel is for one molecule type, from the data: "at the prior" (inactive), the arrow it belongs
 * to, "unread" (active, but setting it to 0 moves no atom class by READ_A), "residue type" (it keeps most of its
 * variance under rotation and residue types explain most of it), "turn-invariant", or "turns".
 */
export function roles(space, g) {
  const rotation = space.groups[g].rotation;
  const arrowOf = {};
  for (const [key, a] of Object.entries(rotation.arrows)) {
    const name = ARROW_NAMES[key] ?? key.replace(/_/g, ' ');
    for (const c of a.channels) arrowOf[c] = name.endsWith('arrow') || a.channels.length !== 3 ? name : `${name} arrow`;
  }
  return channels(space, g).map((c) => {
    if (!c.active) return 'at the prior';
    if (arrowOf[c.index]) return arrowOf[c.index];
    if (!c.read) return 'unread';
    if (c.invariant > 0.5) return rotation.identity_by_channel[c.index] > 0.5 ? 'residue type' : 'turn-invariant';
    return 'turns';
  });
}

/**
 * Reconstruction of one molecule type over all its tokens with noise on channel `target` (0-15, or 16 for all
 * of them): {sigmas, rmsd, recovery}, the unperturbed point at sigma 0 first.
 */
export function noiseCurve(space, g, target) {
  const group = space.groups[g];
  const n = group.noise;
  return {
    sigmas: [0, ...n.sigmas],
    rmsd: [group.reconstruction.rmsd_all_matched, ...n.rmsd_all_matched[target]],
    recovery: [group.reconstruction.recovery, ...n.recovery[target]],
  };
}

/** RMSD (Å, symmetric pairs matched) after setting the k cheapest channels to 0, k = 0..16, and their order. */
export function rateDistortion(space, g) {
  const group = space.groups[g];
  const cs = channels(space, g);
  const order = group.ablation.cumulative_order;
  const points = [{ k: 0, removed: 0, rmsd: group.reconstruction.rmsd_all_matched }];
  let removed = 0;
  group.ablation.cumulative.forEach((s, i) => {
    removed += cs[order[i]].kl;
    points.push({ k: i + 1, removed, rmsd: s.rmsd_all_matched });
  });
  return { order, points };
}

/* ------------------------------------------------------------------ the instrument's fragments */

const PEPTIDE = ['C', 'N'];
const PHOSPHODIESTER = ["O3'", 'P'];

/** The metrics every set of states stores, by the name a state handle gives them. */
const METRICS = {
  rms: 'window_rms_A',
  max: 'window_max_A',
  elsewhere: 'elsewhere_max_A',
  rmsdTrue: 'rmsd_true_A',
  recovered: 'recovered',
  link: 'link_dev_A',
  follow: 'follow_deg',
};

/**
 * One fragment of latents-perturb.json, decoded: the window's atoms in any exactly decoded state, and the
 * context drawn around them. The shift states come with the file; `addSets` adds every set of states of an
 * example, such as those of latents-noise.json (`noise`) or latents-arrows.json (`turn`, `stretch`, with the
 * arrows' lab-frame directions).
 * @returns {{group, input, deltas, window, atoms, context, axes, links, rest, state(name, ...index),
 *   has(name), shape(name), addSets(example), atomName(handle, a)}}
 */
export function fragment(perturb, g) {
  const e = perturb.examples[g];
  const topo = perturb.topology;
  const names = perturb.residue_types;
  const deltas = perturb.deltas;
  const base = unpack(e.xyz).values;
  const A = e.atoms.token.length;
  const W = e.window.res_id.length;
  const atomOf = new Map(e.atoms.token.map((j, a) => [`${j}:${e.atoms.slot[a]}`, a]));
  const nameOf = (type, slot) => {
    const t = topo[type];
    const k = t ? t.slots.indexOf(slot) : -1;
    return k >= 0 ? t.names[k] : null;
  };

  const ctx = e.context;
  const cxyz = unpack(ctx.xyz).values;
  const contextAtom = new Map(ctx.token.map((k, i) => [`${k}:${ctx.slot[i]}`, i]));
  const contextBonds = [];
  ctx.residue.forEach((type, k) => {
    for (const [s1, s2] of topo[type]?.bonds ?? []) {
      const a = contextAtom.get(`${k}:${s1}`), b = contextAtom.get(`${k}:${s2}`);
      if (a !== undefined && b !== undefined) contextBonds.push([a, b]);
    }
  });

  /* Residues in chain order, window and context together, for the bonds between consecutive residues. */
  const residues = [
    ...e.window.res_id.map((res, j) => ({ chain: e.window.chain[j], res, kind: 'window', j })),
    ...ctx.res_id.map((res, k) => ({ chain: ctx.chain[k], res, kind: 'context', k })),
  ].sort((a, b) => (a.chain === b.chain ? a.res - b.res : a.chain < b.chain ? -1 : 1));
  const link = g === 'protein' ? PEPTIDE : PHOSPHODIESTER;
  const slotOf = (type, name) => {
    const t = topo[type];
    const k = t ? t.names.indexOf(name) : -1;
    return k >= 0 ? t.slots[k] : -1;
  };

  /** Which window atoms exist for the residue types `t`: the slot belongs to the type its token decodes to. */
  const presentFor = (t) => Array.from({ length: A }, (_, a) => nameOf(t[e.atoms.token[a]], e.atoms.slot[a]) !== null);

  /**
   * Bonds for the residue types `t`: `window` pairs of window atoms, `cross` pairs [window atom, context atom]
   * and the context's own pairs, which never change.
   */
  function bondsFor(t) {
    const shown = presentFor(t);
    const inWindow = [];
    t.forEach((type, w) => {
      for (const [s1, s2] of topo[type]?.bonds ?? []) {
        const a = atomOf.get(`${w}:${s1}`), b = atomOf.get(`${w}:${s2}`);
        if (a !== undefined && b !== undefined && shown[a] && shown[b]) inWindow.push([a, b]);
      }
    });
    const cross = [];
    for (let i = 0; i + 1 < residues.length; i++) {
      const p = residues[i], q = residues[i + 1];
      if (p.chain !== q.chain || q.res !== p.res + 1 || (p.kind === 'context' && q.kind === 'context')) continue;
      const end = (r, name) => {
        if (r.kind === 'window') {
          const a = atomOf.get(`${r.j}:${slotOf(t[r.j], name)}`);
          return a !== undefined && shown[a] ? { kind: 'window', i: a } : null;
        }
        const k = contextAtom.get(`${r.k}:${slotOf(ctx.residue[r.k], name)}`);
        return k !== undefined ? { kind: 'context', i: k } : null;
      };
      const a = end(p, link[0]), b = end(q, link[1]);
      if (!a || !b) continue;
      if (a.kind === 'window' && b.kind === 'window') inWindow.push([a.i, b.i]);
      else cross.push(a.kind === 'window' ? [a.i, b.i] : [b.i, a.i]);
    }
    return { window: inWindow, cross, context: contextLinks };
  }

  /* Links between consecutive context residues, fixed. */
  const contextLinks = [...contextBonds];
  for (let i = 0; i + 1 < residues.length; i++) {
    const p = residues[i], q = residues[i + 1];
    if (p.kind !== 'context' || q.kind !== 'context' || p.chain !== q.chain || q.res !== p.res + 1) continue;
    const a = contextAtom.get(`${p.k}:${slotOf(ctx.residue[p.k], link[0])}`);
    const b = contextAtom.get(`${q.k}:${slotOf(ctx.residue[q.k], link[1])}`);
    if (a !== undefined && b !== undefined) contextLinks.push([a, b]);
  }

  /* The sets of states: packed offsets, decoded types and metrics, each with its leading shape. */
  const sets = {};
  const lab = {};
  function addSets(example) {
    for (const [name, doc] of Object.entries(example)) {
      if (!doc?.offsets || !doc.decoded_type) continue;
      const set = { offsets: unpack(doc.offsets), decoded: unpack(doc.decoded_type) };
      for (const [key, field] of Object.entries(METRICS)) if (doc[field]) set[key] = unpack(doc[field]);
      set.shape = set.offsets.shape.slice(0, -2);
      sets[name] = set;
    }
    if (example.lab_rest) {
      Object.assign(lab, { anchor: example.anchor, maps: example.maps, rest: unpack(example.lab_rest), turn: unpack(example.lab_turn), stretch: unpack(example.lab_stretch) });
    }
  }
  addSets(e);

  /** A lab-frame arrow per window token, from its representative atom (`atom`), in unperturbed lengths. */
  const arrowsOf = (flat) => lab.anchor.map((atom, j) => ({ atom, vector: [flat[3 * j], flat[3 * j + 1], flat[3 * j + 2]] }));

  /**
   * One state: its residue types, atoms, displacement from the unperturbed decode (Å), bonds, metrics
   * (rms, max, elsewhere, rmsdTrue in Å, recovered tokens, link in Å, follow in degrees where stored) and, for
   * the arrow sets, the arrows.
   */
  function handle(key, types, offsets, metrics, arrows = null) {
    let positions = null;
    return {
      key,
      ...metrics,
      arrows,
      types: () => types,
      positions() {
        if (!positions) {
          positions = new Float32Array(3 * A);
          for (let i = 0; i < 3 * A; i++) positions[i] = base[i] + (offsets ? offsets[i] : 0);
        }
        return positions;
      },
      displacement() {
        const out = new Float32Array(A);
        if (offsets) for (let a = 0; a < A; a++) out[a] = Math.hypot(offsets[3 * a], offsets[3 * a + 1], offsets[3 * a + 2]);
        return out;
      },
      present: () => presentFor(types),
      bonds: () => bondsFor(types),
    };
  }

  const rest = handle('rest', e.rest.decoded, null, {
    rms: 0,
    max: 0,
    elsewhere: 0,
    rmsdTrue: e.rest.rmsd_true_A,
    recovered: e.rest.recovered,
    link: e.rest.link_dev_A,
  });

  /** The state at `index` of set `name` (shift [channel, delta], noise [target, sigma, draw], ...). */
  function state(name, ...index) {
    const set = sets[name];
    if (!set) throw new Error(`the ${g} fragment has no ${name} states`);
    if (index.length !== set.shape.length || index.some((i, k) => !(Number.isInteger(i) && i >= 0 && i < set.shape[k]))) {
      throw new RangeError(`no ${name} state ${index.join(', ')}`);
    }
    const metrics = {};
    for (const key of Object.keys(METRICS)) if (set[key]) metrics[key] = take(set[key], ...index)[0];
    const types = Array.from(take(set.decoded, ...index), (t) => names[t]);
    let arrows = null;
    if (name === 'turn' || name === 'stretch') arrows = arrowsOf(take(lab[name], ...index));
    return handle(`${name}:${index.join(',')}`, types, take(set.offsets, ...index), metrics, arrows);
  }

  return {
    group: g,
    input: e.input,
    deltas,
    window: e.window,
    atoms: e.atoms,
    axes: e.axes,
    links: e.links,
    context: { ...ctx, positions: cxyz },
    rest,
    state,
    has: (name) => name in sets,
    shape: (name) => sets[name]?.shape.slice(),
    addSets,
    /** The rest pose's arrows of arrow `a` (index into TURN_ARROWS), once latents-arrows.json is added. */
    restArrows: (a) => arrowsOf(take(lab.rest, a)),
    arrowMap: (key) => lab.maps?.[key],
    atomName: (h, a) => nameOf(h.types()[e.atoms.token[a]], e.atoms.slot[a]),
  };
}

/* ------------------------------------------------------------------ words */

const ARTICLE = { protein: 'the protein', dna: 'the DNA', rna: 'the RNA' };
const joinAnd = (items) => (items.length < 2 ? items.join('') : `${items.slice(0, -1).join(', ')} and ${items.at(-1)}`);

/** The instrument's claim for one state, from the RMS displacement (Å) of each fragment. */
export function shiftClaim(moved) {
  const still = GROUPS.filter((g) => !(moved[g] >= MOVED_A));
  const moves = GROUPS.filter((g) => moved[g] >= MOVED_A);
  if (!moves.length) return `No fragment moves by ${MOVED_A} Å.`;
  if (!still.length) return 'It moves all three fragments.';
  return `It moves ${joinAnd(moves.map((g) => ARTICLE[g]))} and leaves ${joinAnd(still.map((g) => ARTICLE[g]))} as ${still.length > 1 ? 'they were' : 'it was'}.`;
}

const LETTERS = {
  ALA: 'A', ARG: 'R', ASN: 'N', ASP: 'D', CYS: 'C', GLN: 'Q', GLU: 'E', GLY: 'G', HIS: 'H', ILE: 'I',
  LEU: 'L', LYS: 'K', MET: 'M', PHE: 'F', PRO: 'P', SER: 'S', THR: 'T', TRP: 'W', TYR: 'Y', VAL: 'V',
  DA: 'A', DC: 'C', DG: 'G', DT: 'T', A: 'A', C: 'C', G: 'G', U: 'U',
};

/** One-letter code of a residue type ("?" for any other token). */
export const oneLetter = (name) => LETTERS[name] ?? '?';

/** A shift in prior standard deviations as text: "+1", "−0.5", "0". */
export function shiftText(delta) {
  if (delta === 0) return '0';
  const text = Math.abs(delta).toString();
  return `${delta > 0 ? '+' : '−'}${text}`;
}
