/* Framed architecture diagrams outside the explorer (the architecture notebook).
 *
 * An <iframe class="ncd-frame" data-ncd="<slug>"> shows docs/_static/architecture/ncd/<slug>.html.
 * This script gives each frame the docs theme and keeps it in step with the theme switcher, sizes
 * the frame to the diagram at the width of the frame, and sends a click on a deeper block to that
 * block's diagram in the explorer; the diagram opens a new tab itself when the reader asks for one.
 * The messages are listed at the top of ncd/ncd-embed.js.
 *
 * The frames load lazily. A frame whose diagram has reported its state is asked to redraw in a new
 * theme; any other frame is loaded again with the new theme in its URL, which also covers frames
 * that have not started loading.
 *
 * Each diagram's size is kept for the session, and every frame whose diagram has reported one is
 * given its height before it loads, so a reload or a return to the page finds the reader's place.
 */
(() => {
  'use strict';

  const frames = () => Array.from(document.querySelectorAll('iframe.ncd-frame[data-ncd]'));
  if (!frames().length) return;

  const MIN_HEIGHT = 240;
  const MAX_HEIGHT_FRACTION = 0.85;
  const here = document.currentScript?.src || document.querySelector('script[src*="ncd-frames.js"]')?.src;
  const staticBase = new URL('.', here || location.href);
  const explorer = new URL('../../architecture/index.html', staticBase);
  /* pydata-sphinx-theme sets `data-theme` after this script has run on a first visit; until then
     the mode decides, and the system preference decides the automatic mode. */
  const darkQuery = matchMedia('(prefers-color-scheme: dark)');
  const theme = () => {
    const { theme: resolved, mode } = document.documentElement.dataset;
    if (resolved === 'dark' || resolved === 'light') return resolved;
    if (mode === 'dark' || mode === 'light') return mode;
    return darkQuery.matches ? 'dark' : 'light';
  };
  const themeOf = (frame) => new URL(frame.getAttribute('src') || '', location.href).searchParams.get('theme');
  /* Frames whose diagram has reported its state, and so answers messages. */
  const answering = new WeakSet();

  const sizeKey = (frame) => `ncd-size:${frame.dataset.ncd}`;
  const fitHeight = (frame, width, height) => {
    const wanted = height * Math.min(1, frame.clientWidth / width) + 4;
    const tallest = window.innerHeight * MAX_HEIGHT_FRACTION;
    frame.style.height = `${Math.round(Math.max(MIN_HEIGHT, Math.min(wanted, tallest)))}px`;
  };
  /* Session storage may be unavailable, as in some private windows; the frames then keep their
     placeholder height until they load. */
  const storedSize = (frame) => {
    try {
      const size = JSON.parse(sessionStorage.getItem(sizeKey(frame)) || 'null');
      return Array.isArray(size) && size[0] > 0 && size[1] > 0 ? size : null;
    } catch {
      return null;
    }
  };
  const storeSize = (frame, width, height) => {
    try {
      sessionStorage.setItem(sizeKey(frame), JSON.stringify([width, height]));
    } catch {
      /* The size is only a head start for the next visit. */
    }
  };

  /* A replaced iframe adds no entry to the session history, where a changed `src` would. */
  const load = (frame) => {
    const url = new URL(`ncd/${frame.dataset.ncd}.html`, staticBase);
    url.searchParams.set('theme', theme());
    const fresh = frame.cloneNode(false);
    fresh.src = url.href;
    frame.replaceWith(fresh);
  };

  let shown = theme();
  for (const frame of frames()) {
    if (themeOf(frame) !== shown) load(frame);
  }
  for (const frame of frames()) {
    const size = storedSize(frame);
    if (size) fitHeight(frame, ...size);
  }

  window.addEventListener('message', (event) => {
    const frame = frames().find((candidate) => candidate.contentWindow === event.source);
    if (!frame || !event.data || typeof event.data !== 'object') return;
    const data = event.data;
    if (data.type === 'ncd:state') {
      answering.add(frame);
      if (data.theme && data.theme !== theme()) frame.contentWindow.postMessage({ type: 'ncd:set-theme', theme: theme() }, '*');
      if (data.width > 0 && data.height > 0) {
        fitHeight(frame, data.width, data.height);
        storeSize(frame, data.width, data.height);
      }
    } else if (data.type === 'ncd:navigate') {
      const slug = new URL(data.href).pathname.split('/').pop().replace(/\.html$/, '');
      location.href = `${explorer.href}#view=${slug}`;
    }
  });

  /* pydata-sphinx-theme rewrites `data-theme` on <html>, sometimes to the value it already had. */
  const followTheme = () => {
    const wanted = theme();
    if (wanted === shown) return;
    shown = wanted;
    for (const frame of frames()) {
      if (answering.has(frame)) frame.contentWindow?.postMessage({ type: 'ncd:set-theme', theme: wanted }, '*');
      else if (themeOf(frame) !== wanted) load(frame);
    }
  };
  new MutationObserver(followTheme).observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme', 'data-mode'] });
  darkQuery.addEventListener('change', followTheme);
})();
