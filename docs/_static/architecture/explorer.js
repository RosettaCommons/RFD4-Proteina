/* The architecture explorer on docs/architecture/index.rst.
 *
 * One stage shows one diagram at a time. The root is the overview map drawn inline in the page;
 * every other diagram is a neural circuit diagram page under `ncd/`, framed in an iframe. The
 * manifest (`manifest.js`,) holds the tree of
 * diagrams and the prose of the panel under the stage.
 *
 * Navigation is the URL hash (`#view=<id>`), so the browser's back and forward buttons walk the
 * levels. Every link to a level is a real `#view=` link, which a modified or middle click opens in
 * a new tab. A hash without `view=` is an ordinary anchor of the page and leaves the stage as it
 * is. The framed page talks to this one with the messages listed at the top of ncd/ncd-embed.js.
 *
 * A new iframe replaces the old one on every navigation: changing the `src` of an iframe adds an
 * entry to the session history, and the back button would then rewind the frame before the level.
 *
 * Focus moves to the stage only after a navigation from the stage itself (the map or a block in a
 * diagram), from a card or from a link elsewhere on the page, and on into the new diagram's frame
 * once that has loaded, so the diagram's keys work at once; the frame is asked to show its focus
 * ring when a key made the navigation. The tree and the breadcrumbs keep focus. A polite live
 * region in the toolbar announces the diagram shown. While the explorer is expanded over the page,
 * Tab and Shift+Tab cycle through the explorer alone, and closing it returns focus to where it was
 * before.
 */
(() => {
  'use strict';

  const manifest = window.RFD4_ARCHITECTURE;
  const root = document.getElementById('arch-explorer');
  if (!manifest || !root) return;

  const staticBase = new URL('.', document.currentScript?.src || location.href);
  const diagrams = manifest.diagrams;
  const $ = (selector) => root.querySelector(selector);
  const toolbar = $('.arch-toolbar');
  const stage = $('.arch-stage');
  const map = $('.arch-map');
  let frame = $('.arch-frame');
  const crumbs = $('.arch-crumbs');
  const tree = $('.arch-tree');
  const panel = $('.arch-panel');
  const zoomLabel = $('[data-zoom="fit"]');
  const openLink = $('.arch-open');
  const expand = $('.arch-expand');

  let current = null;
  let frameState = null;
  /* Whether `id` names a diagram of the manifest, rather than a property every object inherits. */
  const known = (id) => typeof id === 'string' && Object.hasOwn(diagrams, id);

  const element = (tag, attributes = {}, ...children) => {
    const node = document.createElement(tag);
    for (const [key, value] of Object.entries(attributes)) {
      if (key === 'html') node.innerHTML = value;
      else if (key === 'text') node.textContent = value;
      else node.setAttribute(key, value);
    }
    node.append(...children);
    return node;
  };
  const live = $('.arch-live') || element('p', { class: 'arch-live', 'aria-live': 'polite' });
  if (!live.isConnected) crumbs.after(live);

  /* pydata-sphinx-theme sets `data-theme` after this script has run on a first visit; until then
     the mode decides, and the system preference decides the automatic mode. */
  const darkQuery = matchMedia('(prefers-color-scheme: dark)');
  const theme = () => {
    const { theme: resolved, mode } = document.documentElement.dataset;
    if (resolved === 'dark' || resolved === 'light') return resolved;
    if (mode === 'dark' || mode === 'light') return mode;
    return darkQuery.matches ? 'dark' : 'light';
  };
  let shownTheme = theme();

  const pageUrl = (id) => new URL(`${diagrams[id].page}?theme=${theme()}`, staticBase).href;
  const lineage = (id) => {
    const chain = [];
    for (let at = id; at; at = diagrams[at].parent) chain.unshift(at);
    return chain;
  };
  const byPage = (href) => {
    const path = new URL(href, location.href).pathname;
    return Object.keys(diagrams).find((id) => diagrams[id].page && path.endsWith(`/${diagrams[id].page}`));
  };
  /* A click the browser should handle itself: a modified or non-primary click opens a new tab. */
  const modified = (event) =>
    event.defaultPrevented || event.button !== 0 || event.metaKey || event.ctrlKey || event.shiftKey || event.altKey;
  /* Whether a key, rather than a pointer, made the click: Enter on a link, which Firefox reports with
     a click count of one. */
  let pressedKey = false;
  document.addEventListener('keydown', () => (pressedKey = true), true);
  document.addEventListener('pointerdown', () => (pressedKey = false), true);
  const byKey = (event) => event.detail === 0 || pressedKey;
  const link = (id, label, className, focus = null) => {
    const anchor = element('a', { href: `#view=${id}`, class: className || '' });
    anchor.textContent = label ?? diagrams[id].short ?? diagrams[id].title;
    anchor.addEventListener('click', (event) => {
      if (modified(event)) return;
      event.preventDefault();
      go(id, { focus, keyboard: byKey(event) });
    });
    return anchor;
  };

  /* --------------------------------------------------------------- the tree */

  const buildTree = () => {
    const branch = (id) => {
      const item = element('li', { 'data-id': id });
      item.append(link(id, diagrams[id].short ?? diagrams[id].title, 'arch-tree-link'));
      const children = diagrams[id].children || [];
      if (children.length) item.append(element('ul', {}, ...children.map(branch)));
      return item;
    };
    tree.replaceChildren(element('p', { class: 'arch-tree-title', text: 'All diagrams' }), element('ul', {}, branch(manifest.root)));
  };

  /* -------------------------------------------------------------- the panel */

  const renderPanel = (id) => {
    const info = diagrams[id];
    const depth = lineage(id).length - 1;
    const parts = [
      element(
        'div',
        { class: 'arch-panel-head' },
        element('span', { class: 'arch-level', text: depth === 0 ? 'Overview' : `Level ${depth}` }),
        element('h3', { class: 'arch-panel-title', text: info.title }),
      ),
    ];
    if (info.summary) parts.push(element('div', { class: 'arch-summary', html: info.summary }));
    if (info.highlights?.length) {
      parts.push(
        element(
          'ul',
          { class: 'arch-highlights' },
          ...info.highlights.map((h) =>
            element('li', { class: `arch-highlight arch-highlight-${h.kind || 'note'}` }, element('span', { class: 'arch-highlight-label', text: h.label }), element('span', { html: h.html })),
          ),
        ),
      );
    }
    if (info.facts?.length) {
      const rows = info.facts.map(([name, value]) => element('tr', {}, element('th', { html: name }), element('td', { html: value })));
      parts.push(element('table', { class: 'arch-facts' }, element('tbody', {}, ...rows)));
    }
    const children = info.children || [];
    if (children.length) {
      parts.push(element('p', { class: 'arch-subhead', text: 'Go deeper' }));
      parts.push(
        element(
          'div',
          { class: 'arch-cards' },
          ...children.map((child) => {
            const card = link(child, '', 'arch-card', 'stage');
            card.replaceChildren(
              element('span', { class: 'arch-card-title', text: diagrams[child].title }),
              element('span', { class: 'arch-card-body', text: diagrams[child].blurb || '' }),
            );
            return card;
          }),
        ),
      );
    }
    if (info.code?.length) {
      parts.push(element('p', { class: 'arch-subhead', text: 'In the code' }));
      parts.push(
        element(
          'ul',
          { class: 'arch-code' },
          ...info.code.map((ref) => element('li', {}, element('a', { href: ref.url, target: '_blank', rel: 'noopener', text: ref.label }))),
        ),
      );
    }
    panel.replaceChildren(...parts);
  };

  /* -------------------------------------------------------------- the stage */

  const post = (data) => {
    if (!frame.hidden) frame.contentWindow?.postMessage(data, '*');
  };
  /* The Fit button shows the zoom, and its accessible name starts with what it shows. */
  const setZoomLabel = () => {
    const percent = frameState ? `${Math.round(frameState.scale * 100)}%` : null;
    const name = percent ? `${percent} zoom: fit the diagram to the frame` : 'Fit the diagram to the frame';
    if (zoomLabel.textContent !== (percent ?? 'Fit')) zoomLabel.textContent = percent ?? 'Fit';
    if (zoomLabel.getAttribute('aria-label') !== name) zoomLabel.setAttribute('aria-label', name);
  };
  /* The frame that takes the focus from the stage once it has loaded, and whether it shows its ring. */
  let handOff = null;
  const focusFrame = (ring) => {
    frame.dataset.ncdFocus = ring ? 'visible' : 'quiet';
    frame.focus({ preventScroll: true });
  };
  const replaceFrame = (src, title = 'Interactive neural circuit diagram') => {
    const fresh = frame.cloneNode(false);
    fresh.removeAttribute('src');
    fresh.removeAttribute('data-ncd-focus');
    fresh.hidden = src === null;
    fresh.title = title;
    if (src !== null) fresh.src = src;
    fresh.addEventListener('load', () => {
      fresh.contentWindow?.postMessage({ type: 'ncd:hello' }, '*');
      if (handOff?.frame !== fresh) return;
      const { ring } = handOff;
      handOff = null;
      if (document.activeElement === stage) focusFrame(ring);
    });
    frame.replaceWith(fresh);
    frame = fresh;
  };

  const show = (id, announce = true) => {
    const info = diagrams[id];
    current = id;
    root.dataset.view = id;
    for (const item of tree.querySelectorAll('li')) {
      const here = item.dataset.id === id;
      item.classList.toggle('arch-current', here);
      const anchor = item.querySelector(':scope > .arch-tree-link');
      if (here) anchor.setAttribute('aria-current', 'page');
      else anchor.removeAttribute('aria-current');
    }
    revealInTree();
    crumbs.replaceChildren(
      ...lineage(id).flatMap((step, index, all) => {
        const label = diagrams[step].short ?? diagrams[step].title;
        const node =
          index === all.length - 1
            ? element('span', { class: 'arch-crumb-current', 'aria-current': 'page', tabindex: '-1', text: label })
            : link(step, label, 'arch-crumb', 'crumb');
        return index ? [element('span', { class: 'arch-crumb-sep', text: '›' }), node] : [node];
      }),
    );
    renderPanel(id);
    frameState = null;
    if (info.kind === 'map') {
      map.hidden = false;
      replaceFrame(null);
      root.classList.add('arch-on-map');
      openLink.hidden = true;
    } else {
      map.hidden = true;
      root.classList.remove('arch-on-map');
      shownTheme = theme();
      replaceFrame(pageUrl(id), `Interactive neural circuit diagram: ${info.title}`);
      openLink.hidden = false;
      openLink.href = pageUrl(id);
    }
    setZoomLabel();
    if (announce) live.textContent = `Showing: ${info.title}`;
  };

  /* Scroll the tree, and only the tree, to the current diagram when its link is out of view. */
  const revealInTree = () => {
    const anchor = tree.querySelector('.arch-tree-link[aria-current="page"]');
    if (!anchor || tree.clientHeight === 0 || tree.scrollHeight <= tree.clientHeight) return;
    const top = tree.getBoundingClientRect().top + tree.clientTop;
    const shown = anchor.getBoundingClientRect();
    if (shown.top >= top && shown.bottom <= top + tree.clientHeight) return;
    tree.scrollTop += shown.top - top - (tree.clientHeight - shown.height) / 2;
  };

  /* Bring the toolbar out from under pydata's sticky header after a navigation, if it is not
     already in full view. */
  const reveal = () => {
    if (root.classList.contains('arch-expanded')) return;
    const header = document.querySelector('.bd-header');
    const headerBottom = header ? Math.max(0, header.getBoundingClientRect().bottom) : 0;
    const bar = toolbar.getBoundingClientRect();
    if (bar.top >= headerBottom && bar.bottom <= innerHeight) return;
    const behavior = matchMedia('(prefers-reduced-motion: reduce)').matches ? 'auto' : 'smooth';
    window.scrollTo({ top: scrollY + root.getBoundingClientRect().top - headerBottom - 8, behavior });
  };

  /* `focus` is 'stage', 'crumb' (the current breadcrumb, which replaces the one clicked) or null;
     `keyboard` says a key made the navigation, so the diagram that takes focus shows its ring. */
  const go = (id, { replace = false, focus = null, keyboard = false } = {}) => {
    if (!known(id)) id = manifest.root;
    const hash = `#view=${id}`;
    if (replace) history.replaceState({ archView: id }, '', hash);
    else if (location.hash !== hash) history.pushState({ archView: id }, '', hash);
    const shown = id !== current;
    if (shown) show(id, !replace);
    if (replace) return;
    reveal();
    if (focus === 'stage') {
      stage.focus({ preventScroll: true });
      handOff = shown && !frame.hidden ? { frame, ring: keyboard } : null;
      if (!shown && !frame.hidden) focusFrame(keyboard);
    } else if (focus === 'crumb') {
      crumbs.querySelector('.arch-crumb-current')?.focus({ preventScroll: true });
    }
  };

  /* The diagram named by `#view=`, the root for an unknown name, and null for any other hash. */
  const fromHash = () => {
    const view = new URLSearchParams(location.hash.slice(1)).get('view');
    if (view === null) return null;
    return known(view) ? view : manifest.root;
  };
  const remember = (id) => history.replaceState({ ...(history.state || {}), archView: id }, '');

  /* Links in the overview map carry `data-arch` with the diagram they open. */
  root.addEventListener('click', (event) => {
    const target = event.target instanceof Element && event.target.closest('[data-arch]');
    if (!target || modified(event)) return;
    event.preventDefault();
    go(target.dataset.arch, { focus: 'stage', keyboard: byKey(event) });
  });
  document.addEventListener('click', (event) => {
    const target = event.target instanceof Element && event.target.closest('a[href^="#view="]');
    if (!target || root.contains(target) || modified(event)) return;
    event.preventDefault();
    go(new URLSearchParams(target.getAttribute('href').slice(1)).get('view'), { focus: 'stage', keyboard: byKey(event) });
  });

  window.addEventListener('message', (event) => {
    if (event.source !== frame.contentWindow || !event.data || typeof event.data !== 'object') return;
    const data = event.data;
    if (data.type === 'ncd:navigate') {
      const id = byPage(data.href);
      if (id) go(id, { focus: 'stage', keyboard: data.keyboard === true });
    } else if (data.type === 'ncd:state') {
      frameState = data;
      setZoomLabel();
      if (data.theme && data.theme !== theme()) post({ type: 'ncd:set-theme', theme: theme() });
    } else if (data.type === 'ncd:escape') {
      if (root.classList.contains('arch-expanded')) setExpanded(false);
    }
  });

  for (const button of root.querySelectorAll('[data-zoom]')) {
    button.addEventListener('click', () => {
      const kind = button.dataset.zoom;
      post(kind === 'fit' ? { type: 'ncd:zoom', fit: true } : { type: 'ncd:zoom', step: kind === 'in' ? 1 : -1 });
    });
  }

  /* While expanded, a guard at each end of the explorer takes the focus that Tab or Shift+Tab moves
     out of it and hands it to the other end. */
  const focusable = () =>
    Array.from(root.querySelectorAll('a[href], button:not([disabled]), iframe, [tabindex]:not([tabindex="-1"])')).filter(
      (node) =>
        !node.classList.contains('arch-focus-guard') && node.getClientRects().length > 0 && getComputedStyle(node).visibility === 'visible',
    );
  const guard = (toEnd) => {
    const node = element('span', { class: 'arch-focus-guard' });
    node.addEventListener('focus', () => {
      const nodes = focusable();
      (toEnd ? nodes.at(-1) : nodes[0])?.focus();
    });
    return node;
  };
  const guards = [guard(true), guard(false)];
  root.prepend(guards[0]);
  root.append(guards[1]);
  let focusBefore = null;
  const setExpanded = (on) => {
    const was = root.classList.contains('arch-expanded');
    root.classList.toggle('arch-expanded', on);
    document.documentElement.classList.toggle('arch-no-scroll', on);
    expand.textContent = on ? 'Close' : 'Expand';
    for (const node of guards) {
      if (on) node.setAttribute('tabindex', '0');
      else node.removeAttribute('tabindex');
    }
    if (on && !was) {
      focusBefore = root.contains(document.activeElement) ? document.activeElement : expand;
    } else if (!on && was) {
      const back = focusBefore?.isConnected && focusBefore.getClientRects().length > 0 ? focusBefore : expand;
      focusBefore = null;
      back.focus({ preventScroll: true });
    }
    revealInTree();
  };
  expand.addEventListener('click', () => setExpanded(!root.classList.contains('arch-expanded')));
  document.addEventListener('keydown', (event) => {
    if (event.key === 'Escape' && root.classList.contains('arch-expanded')) setExpanded(false);
  });

  /* The docs theme switcher rewrites `data-theme` on <html>, sometimes to the value it already
     had; the framed diagram is asked to follow only when the theme changes. */
  const followTheme = () => {
    const wanted = theme();
    if (wanted === shownTheme) return;
    shownTheme = wanted;
    post({ type: 'ncd:set-theme', theme: wanted });
    if (!openLink.hidden && current) openLink.href = pageUrl(current);
  };
  new MutationObserver(followTheme).observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme', 'data-mode'] });
  darkQuery.addEventListener('change', followTheme);

  /* A history entry of an ordinary anchor records the diagram shown when it was made. */
  const follow = () => {
    const recorded = history.state?.archView;
    const id = fromHash() ?? (known(recorded) ? recorded : location.hash.length > 1 ? null : manifest.root);
    if (id === null) remember(current);
    else if (id !== current) show(id);
  };
  window.addEventListener('popstate', follow);
  window.addEventListener('hashchange', follow);

  buildTree();
  for (const anchor of root.querySelectorAll('a[data-arch]:not([href])')) anchor.setAttribute('href', `#view=${anchor.dataset.arch}`);
  const initial = fromHash();
  if (initial) {
    go(initial, { replace: true });
  } else {
    const recorded = history.state?.archView;
    show(known(recorded) ? recorded : manifest.root, false);
    remember(current);
  }
})();
