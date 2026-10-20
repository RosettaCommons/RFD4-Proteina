/* Helpers loaded by every architecture diagram page, before the tsncd bundle.
 *
 * tsncd draws one neural circuit diagram from the JSON message embedded in the page. This
 * script adds what the docs need around that figure, and talks to the page that frames it (the
 * explorer, explorer.js, or the notebook, ncd-frames.js) with `postMessage`.
 *
 * From the parent page:
 *   {type: 'ncd:hello'}                  answered with 'ncd:state' once the figure is ready.
 *   {type: 'ncd:set-theme', theme}       'light' or 'dark'. The figure is drawn again unless it
 *                                        already shows that theme; a theme that arrives before the
 *                                        figure is ready is applied once it is.
 *   {type: 'ncd:zoom', scale|step|fit}   an absolute scale, one step in (1) or out (-1), or Fit.
 * To the parent page:
 *   {type: 'ncd:state', scale, width, height, fit, theme}
 *                                        on every change of the view. `width` and `height` are the
 *                                        figure's size at scale 1, and `theme` is the theme drawn.
 *   {type: 'ncd:navigate', href, keyboard}
 *                                        a block with a deeper diagram was followed in place, with a
 *                                        key when `keyboard` is true. A new tab is opened by this page
 *                                        itself (see drill-down) and is not announced.
 *   {type: 'ncd:escape'}                 Escape was pressed with no inspection box open.
 * A parent page that focuses the frame may first set `data-ncd-focus` on the frame element to
 * 'visible' or 'quiet', which shows or hides the focus ring around the frame (see keyboard).
 *
 * theme       `?theme=light|dark` picks the theme before tsncd reads the message. A script in the
 *             head of the page paints the page in that
 *             theme before its first paint, and this script paints it again on start-up.
 * title       tsncd names the tab after the diagram; the page keeps the title it was written with.
 * camera      the figure is one transformed canvas. Dragging with a mouse, a pen or one finger
 *             pans, as do the arrow keys while the page itself has focus; Ctrl/Cmd + wheel or a
 *             pinch zooms, on a touch screen or on a trackpad (Safari's gesture events), and so do
 *             + and -. Fit, or 0, shows the whole figure.
 * drill-down  a block whose references hold a link labelled "Open diagram" is a door to a deeper
 *             diagram and shows a pointer cursor. A mouse click on it, or on that link in its box,
 *             sends 'ncd:navigate', or follows the link when the page is opened on its own, and a
 *             Shift click on the block locks its box. A Ctrl, Cmd, Alt or middle click, or a Shift
 *             click on the link, opens a new tab from this page, within the click, which popup
 *             blockers allow: a framed page opens the explorer at the deeper diagram,
 *             `../../../architecture/index.html#view=<slug>` from this page's URL (the pages live in
 *             `_static/architecture/ncd/`), and a page opened on its own opens the deeper page in the
 *             theme it shows. tsncd opens boxes on hover only for a mouse, so a tap with a finger or
 *             a pen opens the box of a door, and a second tap on the block while that box is open, or
 *             a tap on "Open diagram" in it, follows the door.
 * keyboard    every region tsncd opens a box for is also a button of one toolbar, laid over the
 *             figure in the figure's order, so Tab reaches the figure as one stop. The arrow keys,
 *             Home and End move between the regions and the camera follows; a region's box opens
 *             when its button takes focus, as on hover. Enter follows a door, as a click does, and
 *             locks the box of any other region; Shift+Enter locks a door's box, and Ctrl, Cmd or
 *             Alt+Enter opens the door in a new tab. Tab goes on into the open box and its links, and
 *             Escape closes it. A ring marks the focused region, and the whole frame while the page
 *             itself has keyboard focus.
 * boxes       tsncd places a box up to 1030 px wide beside the pointer, which in a frame covers
 *             the block it describes. Each box is instead docked beside the region it opened from,
 *             or above or below it, where the frame has room, and otherwise beside the point it
 *             opened at. It is as wide as that room, at most as tall as the frame, and scrolls. A
 *             diagram wider than its box is scaled down to the box's width but never below half its
 *             size, and a formula is never cut, so a box is docked beside a region only where its
 *             formula and that diagram fit; where nothing fits, the box spans the frame in the half
 *             away from the region and the diagram scrolls sideways inside it. A locked box may cover
 *             its region, which a click no longer needs to reach, unless a tap opened it on a door,
 *             which a second tap follows: it goes above or below the region only where its text and a
 *             real part of its diagram fit, and spans all of the frame where the half is too short for
 *             them. A box grows around where it was placed, so a pointer
 *             inside it stays there when its diagram arrives, unless the diagram needs a wider or
 *             taller place than the text did. A resize of the frame docks every box again beside
 *             where its region is now, and a box is docked again when it is locked or unlocked.
 *
 * tsncd hit-tests the pointer against rectangles measured in the untransformed drawing, from
 * `clientX` and `clientY`, and places inspection boxes from `pageX` and `pageY`. The client
 * coordinates of pointer events inside a scaled drawing (the figure under the camera, or a diagram
 * scaled into its box) are therefore rewritten into drawing units; page coordinates are left alone.
 * The toolbar opens and locks boxes with the pointer events a mouse would send to its region.
 */
(() => {
  'use strict';

  const DRILLDOWN_LABEL = 'Open diagram';
  const MIN_SCALE = 0.15;
  const MAX_SCALE = 3;
  /* The largest zoom change of one wheel event, which is one press of a zoom button. */
  const WHEEL_STEP = 1.25;
  const WHEEL_LINE_PX = 16;
  const framed = window.parent !== window;
  const message = document.getElementById('tsncd-embedded-message');
  if (!message) return;

  const send = (data) => {
    if (framed) window.parent.postMessage(data, '*');
  };
  /* A Ctrl, Cmd or Alt click asks for a new tab, as a middle click does. */
  const wantsNewTab = (event) => event.ctrlKey || event.metaKey || event.altKey;
  const within = (value, low, high) => Math.max(low, Math.min(high, value));

  /* ------------------------------------------------------------------ theme */

  const asDark = (theme) => (theme === 'dark' ? true : theme === 'light' ? false : null);
  const withTheme = (dark) => {
    const parsed = JSON.parse(message.textContent);
    parsed.settings = { ...(parsed.settings || {}), darkMode: dark };
    message.textContent = JSON.stringify(parsed);
    return parsed;
  };
  const paintSurface = (dark) => {
    document.documentElement.dataset.ncdTheme = dark ? 'dark' : 'light';
    document.documentElement.style.colorScheme = dark ? 'dark' : 'light';
    document.body.style.backgroundColor = dark ? '#1e1e1e' : '#ffffff';
    document.body.style.color = dark ? '#dedede' : '#000000';
  };
  const initialTheme = asDark(new URL(location.href).searchParams.get('theme'));
  if (initialTheme !== null) withTheme(initialTheme);
  /* tsncd draws the dark theme unless the message says `darkMode: false`. */
  let shownDark = initialTheme;
  const showsDark = () => {
    if (shownDark === null) shownDark = JSON.parse(message.textContent).settings?.darkMode !== false;
    return shownDark;
  };
  let pendingDark = null;
  paintSurface(showsDark());

  /* ------------------------------------------------------------------ title */

  const ownTitle = document.title;
  const keepTitle = () => {
    if (document.title !== ownTitle) document.title = ownTitle;
  };
  new MutationObserver(keepTitle).observe(document.head, { childList: true, subtree: true, characterData: true });

  /* ------------------------------------------------------------- drill-down */

  let blockText = null;
  const blocks = () => {
    if (blockText === null) blockText = JSON.parse(message.textContent).auxiliary?.blocks || {};
    return blockText;
  };
  const doorOf = (key) => {
    const info = Object.hasOwn(blocks(), key) ? blocks()[key] : null;
    const reference = info?.references?.find((r) => r.url && r.label.startsWith(DRILLDOWN_LABEL));
    return reference ? { href: reference.url } : null;
  };
  /* The explorer showing the diagram page at `url`, found from this page's own URL. */
  const explorerAt = (url) => {
    const explorer = new URL('../../../architecture/index.html', location.href);
    explorer.hash = `view=${url.pathname.split('/').pop().replace(/\.html$/, '')}`;
    return explorer.href;
  };
  /* Called within the click or key press, so that the browser lets a new tab open. */
  const follow = (door, newTab = false, keyboard = false) => {
    const url = new URL(door.href, location.href);
    if (framed) {
      if (newTab) window.open(explorerAt(url), '_blank', 'noopener');
      else send({ type: 'ncd:navigate', href: url.href, keyboard });
      return;
    }
    // A standalone page takes its theme from the URL, so the next page keeps the one shown here.
    url.searchParams.set('theme', showsDark() ? 'dark' : 'light');
    if (newTab) window.open(url.href, '_blank', 'noopener');
    else location.href = url.href;
  };
  /* The "Open diagram" link of an inspection box that `target` is or is inside, if there is one. */
  const drilldownLink = (target) => {
    const link = target instanceof Element ? target.closest('.inspection-box-references a') : null;
    return link && link.textContent.trim().startsWith(DRILLDOWN_LABEL) ? link : null;
  };

  /* ----------------------------------------------------------------- camera */

  const root = document.documentElement;
  let canvas = null;
  let figure = null;
  let ready = false;
  let scale = 1;
  let x = 0;
  let y = 0;
  let width = 0;
  let height = 0;
  let autoFit = true;
  let pan = null;
  let suppressClick = false;
  let pinch = null;

  const clamp = (value) => within(value, MIN_SCALE, MAX_SCALE);
  const fitScale = () => (width && height ? clamp(Math.min(1, innerWidth / width, innerHeight / height)) : 1);
  /* A figure smaller than the frame sits in its middle, which leaves room on every side for the
     inspection boxes; a larger one can be panned until its edges meet the frame's. */
  const constrain = () => {
    const shownWidth = width * scale;
    const shownHeight = height * scale;
    x = shownWidth <= innerWidth ? (innerWidth - shownWidth) / 2 : within(x, innerWidth - shownWidth, 0);
    y = shownHeight <= innerHeight ? (innerHeight - shownHeight) / 2 : within(y, innerHeight - shownHeight, 0);
  };
  const report = () => send({ type: 'ncd:state', scale, width, height, fit: autoFit, theme: showsDark() ? 'dark' : 'light' });
  const paint = () => {
    constrain();
    canvas.style.transform = `translate(${x}px, ${y}px) scale(${scale})`;
    /* A finger drags the figure only while it overflows the frame; otherwise it scrolls the page. */
    const overflows = width * scale > innerWidth + 1 || height * scale > innerHeight + 1;
    canvas.style.touchAction = overflows ? 'none' : '';
    placeKeys();
    report();
  };
  const zoomAbout = (next, px, py) => {
    const target = clamp(next);
    const fx = (px - x) / scale;
    const fy = (py - y) / scale;
    scale = target;
    x = px - fx * scale;
    y = py - fy * scale;
    paint();
  };
  const fit = () => {
    autoFit = true;
    scale = fitScale();
    x = 0;
    y = 0;
    paint();
  };
  const measure = () => {
    width = canvas.offsetWidth;
    height = canvas.offsetHeight;
    if (autoFit) fit();
    else paint();
  };

  const installCamera = () => {
    figure = document.getElementById('diagram');
    if (!figure) return false;
    canvas = document.createElement('div');
    canvas.className = 'ncd-camera';
    const padding = getComputedStyle(document.body).padding;
    canvas.style.cssText =
      `position:absolute;left:0;top:0;width:max-content;padding:${padding};` +
      'transform-origin:0 0;will-change:transform;';
    for (const node of document.querySelectorAll('#page-heading, #diagram')) canvas.append(node);
    document.body.prepend(canvas);
    const style = document.createElement('style');
    /* The body clips the camera, or a phone's browser widens the page to the unscaled figure and
       the camera, fitting that page, shows the figure off screen. */
    style.textContent = `
      html, body { overflow: hidden !important; }
      html { -webkit-tap-highlight-color: transparent; --ncd-focus: #a80c84; }
      html[data-ncd-theme="dark"] { --ncd-focus: #e879c9; }
      body { position: relative; margin: 0; padding: 0 !important; min-height: 100vh; }
      html[data-ncd-panning], html[data-ncd-panning] * { cursor: grabbing !important; user-select: none !important; }
      html[data-ncd-focused]::after {
        content: ''; position: fixed; inset: 0; z-index: 2147483001; pointer-events: none;
        outline: 2px solid var(--ncd-focus); outline-offset: -2px;
      }
      .ncd-keys { position: fixed; inset: 0; z-index: 1; overflow: hidden; pointer-events: none; }
      .ncd-keys .ncd-key {
        position: absolute; box-sizing: border-box; margin: 0; padding: 0; border: 0; border-radius: 4px;
        background: transparent; pointer-events: none;
      }
      .ncd-keys .ncd-key:focus { outline: 2px solid var(--ncd-focus); outline-offset: 2px; }
    `;
    document.head.append(style);
    return true;
  };

  const inFigure = (target) => target instanceof Element && figure !== null && figure.contains(target);
  const interactive = (target) =>
    target instanceof Element && target.closest('a, button, input, select, textarea, .inspection-box');

  /* The regions of the figure itself, in figure units from the top left of `#diagram`. tsncd's
     `regions()` also lists the diagrams drawn inside open boxes, which lead nowhere. */
  let rootRegions = null;
  const captureRegions = () => {
    rootRegions = null;
    if (!window.tsncd || figure === null || window.tsncd.openBoxes() > 0) return;
    const origin = figure.getBoundingClientRect();
    rootRegions = window.tsncd
      .regions()
      .filter((r) => r.kind !== 'none')
      .map((r) => ({ kind: r.kind, key: r.key, left: r.left - origin.left, top: r.top - origin.top, width: r.width, height: r.height }));
  };
  /* The smallest region of the figure of one of `kinds` under a point in figure units, which is
     the one tsncd opens a box for. */
  const regionAtFigure = (fx, fy, kinds = ['block', 'expansion']) => {
    const hits = rootRegions.filter(
      (r) => kinds.includes(r.kind) && fx >= r.left && fx <= r.left + r.width && fy >= r.top && fy <= r.top + r.height,
    );
    return hits.reduce((a, b) => (a && a.width * a.height <= b.width * b.height ? a : b), null);
  };
  /* The same, for a point in client coordinates. */
  const regionAt = (clientX, clientY, kinds = ['block', 'expansion']) => {
    if (rootRegions === null) captureRegions();
    if (rootRegions === null) return null;
    const origin = figure.getBoundingClientRect();
    return regionAtFigure((clientX - origin.left) / scale, (clientY - origin.top) / scale, kinds);
  };
  const blockAt = (clientX, clientY) => regionAt(clientX, clientY, ['block']);
  /* The door under a pointer event on the figure, with the key of its block. */
  const doorAt = (event) => {
    if (!ready || !inFigure(event.target)) return null;
    const region = blockAt(event.clientX, event.clientY);
    const door = region && doorOf(region.key);
    return door ? { ...door, key: region.key } : null;
  };

  /* The scale each diagram inside an inspection box is drawn at, set when it is fitted to its box. */
  const drawingScale = new WeakMap();
  const scaledDrawing = (target) => {
    if (!(target instanceof Element)) return null;
    if (inFigure(target)) return ready && scale !== 1 ? { node: figure, zoom: scale } : null;
    const diagram = target.closest('.inspection-box-diagram');
    const zoom = diagram ? drawingScale.get(diagram) : undefined;
    return zoom !== undefined && zoom !== 1 ? { node: diagram, zoom } : null;
  };
  const toDrawing = (event) => {
    const drawing = scaledDrawing(event.target);
    if (!drawing) return;
    const bounds = drawing.node.getBoundingClientRect();
    Object.defineProperties(event, {
      clientX: { value: bounds.left + (event.clientX - bounds.left) / drawing.zoom },
      clientY: { value: bounds.top + (event.clientY - bounds.top) / drawing.zoom },
    });
  };

  /* The pointer, or the point the toolbar opened a box at, in client coordinates. A tap has no move
     before it, so the press and the click record it too, and its box is docked beside the finger. */
  let pointer = { x: innerWidth / 2, y: innerHeight / 2 };
  let pointerTarget = null;
  /* The kind of pointer pressed last, which tells a tap from a mouse click. */
  let pointerType = 'mouse';
  const tapping = () => pointerType === 'touch' || pointerType === 'pen';
  const endPan = () => {
    if (!pan) return;
    if (pan.moved) {
      suppressClick = true;
      setTimeout(() => {
        suppressClick = false;
      }, 0);
    }
    pan = null;
    root.removeAttribute('data-ncd-panning');
  };
  document.addEventListener(
    'pointerdown',
    (event) => {
      pointer = { x: event.clientX, y: event.clientY };
      pointerTarget = event.target;
      pointerType = event.pointerType || 'mouse';
      lastInput = 'pointer';
      showFrameRing(false);
      if (!ready || !event.isPrimary || event.button !== 0 || interactive(event.target)) return;
      pan = { id: event.pointerId, x: event.clientX, y: event.clientY, moved: false };
    },
    true,
  );
  document.addEventListener(
    'pointermove',
    (event) => {
      pointer = { x: event.clientX, y: event.clientY };
      pointerTarget = event.target;
      if (!pan || event.pointerId !== pan.id) return;
      if (pinch || !(event.buttons & 1)) {
        endPan();
        return;
      }
      const dx = event.clientX - pan.x;
      const dy = event.clientY - pan.y;
      if (!pan.moved && Math.hypot(dx, dy) < 4) return;
      if (!pan.moved) {
        pan.moved = true;
        root.setAttribute('data-ncd-panning', '');
        try {
          root.setPointerCapture(event.pointerId);
        } catch {
          /* The pointer has already been released. */
        }
      }
      event.stopImmediatePropagation();
      x += dx;
      y += dy;
      pan.x = event.clientX;
      pan.y = event.clientY;
      paint();
    },
    true,
  );
  document.addEventListener('pointerup', endPan, true);
  document.addEventListener('pointercancel', endPan, true);
  /* Registered after the pan, which reads the real coordinates; tsncd's handlers read these. */
  document.addEventListener('pointermove', toDrawing, true);

  /* On the root element, not the document: WebKit on a touch screen sends a tap's click only to an
     element that listens for clicks, and a tap beside the figure must still close its boxes. */
  root.addEventListener(
    'click',
    (event) => {
      if (suppressClick) {
        suppressClick = false;
        event.preventDefault();
        event.stopImmediatePropagation();
        return;
      }
      if (keyOf(event.target)) return;
      pointer = { x: event.clientX, y: event.clientY };
      pointerTarget = event.target;
      /* A click the toolbar sends reaches tsncd as a mouse's would. A browser that also sends 'click'
         for a middle click leaves it to 'auxclick', which opens one tab. */
      if (event.button !== 0 || synthetic.has(event)) {
        toDrawing(event);
        return;
      }
      const newTab = wantsNewTab(event);
      const link = drilldownLink(event.target);
      if (link) {
        event.preventDefault();
        event.stopImmediatePropagation();
        const byKey = event.detail === 0 || lastInput === 'key';
        follow({ href: link.getAttribute('href') }, newTab || event.shiftKey, byKey);
        return;
      }
      /* A tap on a door opens its box first; the box shows what the door leads to. */
      const door = doorAt(event);
      if (door && (newTab || (!event.shiftKey && (!tapping() || boxOpenFor(door.key))))) {
        event.stopImmediatePropagation();
        follow(door, newTab);
        return;
      }
      toDrawing(event);
    },
    true,
  );
  /* A middle click fires 'auxclick', whose default, a new tab for the link, is replaced by ours. */
  document.addEventListener(
    'auxclick',
    (event) => {
      if (event.button !== 1) return;
      const link = drilldownLink(event.target);
      const door = link ? { href: link.getAttribute('href') } : doorAt(event);
      if (!door) return;
      event.preventDefault();
      event.stopImmediatePropagation();
      follow(door, true);
    },
    true,
  );
  /* Middle-click autoscroll, on by default on Windows, starts in the page around a frame and takes
     the click; a middle press on a door does not start it. */
  document.addEventListener(
    'mousedown',
    (event) => {
      if (event.button === 1 && (drilldownLink(event.target) || doorAt(event))) event.preventDefault();
    },
    true,
  );
  /* tsncd shows a pointer over every region, since a click locks its box; only a door keeps it. */
  document.addEventListener('pointermove', (event) => {
    if (!ready || pan || !inFigure(event.target)) return;
    const region = blockAt(pointer.x, pointer.y);
    const door = region !== null && doorOf(region.key) !== null;
    figure.title = door ? 'Click to open this block as its own diagram' : '';
    figure.style.cursor = door ? 'pointer' : 'default';
  });

  document.addEventListener(
    'wheel',
    (event) => {
      if (!ready || (event.target instanceof Element && event.target.closest('.inspection-box'))) return;
      const unitX = event.deltaMode === 1 ? WHEEL_LINE_PX : event.deltaMode === 2 ? innerWidth : 1;
      const unitY = event.deltaMode === 1 ? WHEEL_LINE_PX : event.deltaMode === 2 ? innerHeight : 1;
      if (event.ctrlKey || event.metaKey) {
        event.preventDefault();
        autoFit = false;
        const factor = within(Math.exp(-event.deltaY * unitY * 0.01), 1 / WHEEL_STEP, WHEEL_STEP);
        zoomAbout(scale * factor, event.clientX, event.clientY);
        return;
      }
      /* Pan only while the figure can move that way, so the docs page scrolls past a diagram
         that already shows everything instead of trapping the wheel. */
      const beforeX = x;
      const beforeY = y;
      x -= event.deltaX * unitX;
      y -= event.deltaY * unitY;
      constrain();
      if (x === beforeX && y === beforeY) return;
      event.preventDefault();
      paint();
    },
    { passive: false, capture: true },
  );

  /* A pinch zooms about the point between the fingers and moves with it. */
  const touchPoint = (touches) => ({
    x: (touches[0].clientX + touches[1].clientX) / 2,
    y: (touches[0].clientY + touches[1].clientY) / 2,
    d: Math.hypot(touches[0].clientX - touches[1].clientX, touches[0].clientY - touches[1].clientY),
  });
  document.addEventListener(
    'touchstart',
    (event) => {
      if (!ready || event.touches.length !== 2) return;
      event.preventDefault();
      endPan();
      const at = touchPoint(event.touches);
      pinch = { d: at.d, scale, fx: (at.x - x) / scale, fy: (at.y - y) / scale };
    },
    { passive: false },
  );
  document.addEventListener(
    'touchmove',
    (event) => {
      if (!pinch || event.touches.length !== 2) return;
      event.preventDefault();
      const now = touchPoint(event.touches);
      autoFit = false;
      scale = clamp((pinch.scale * now.d) / Math.max(1, pinch.d));
      x = now.x - pinch.fx * scale;
      y = now.y - pinch.fy * scale;
      paint();
    },
    { passive: false },
  );
  const endPinch = (event) => {
    if (event.touches.length < 2) pinch = null;
  };
  document.addEventListener('touchend', endPinch);
  document.addEventListener('touchcancel', endPinch);
  /* Safari reports a pinch on a trackpad as gesture events rather than a Ctrl + wheel, and a pinch
     on a touch screen as gesture events beside the touches, which zoom already. */
  let gesture = null;
  document.addEventListener('gesturestart', (event) => {
    if (!ready || pinch) return;
    event.preventDefault();
    const known = Number.isFinite(event.clientX) && Number.isFinite(event.clientY);
    gesture = { scale, x: known ? event.clientX : innerWidth / 2, y: known ? event.clientY : innerHeight / 2 };
  });
  document.addEventListener('gesturechange', (event) => {
    if (!gesture || pinch) return;
    event.preventDefault();
    autoFit = false;
    zoomAbout(gesture.scale * event.scale, gesture.x, gesture.y);
  });
  document.addEventListener('gestureend', () => {
    gesture = null;
  });

  /* Registered before tsncd's own listener, so the open boxes are counted before it closes them:
     the first Escape closes the boxes, the next one reaches the parent page. */
  document.addEventListener('keydown', (event) => {
    if (event.key === 'Escape' && (window.tsncd?.openBoxes() ?? 0) === 0) send({ type: 'ncd:escape' });
  });
  document.addEventListener('keydown', (event) => {
    if (!ready || event.defaultPrevented || event.ctrlKey || event.metaKey || event.altKey) return;
    if (interactive(event.target) && !keyOf(event.target)) return;
    const step = { ArrowLeft: [60, 0], ArrowRight: [-60, 0], ArrowUp: [0, 60], ArrowDown: [0, -60] }[event.key];
    if (step) {
      event.preventDefault();
      x += step[0];
      y += step[1];
      paint();
    } else if (event.key === '+' || event.key === '=') {
      autoFit = false;
      zoomAbout(scale * 1.2, innerWidth / 2, innerHeight / 2);
    } else if (event.key === '-') {
      autoFit = false;
      zoomAbout(scale / 1.2, innerWidth / 2, innerHeight / 2);
    } else if (event.key === '0') {
      fit();
    }
  });

  /* tsncd lays text out from measured boxes, which a scaled camera would shrink, so the figure is
     drawn again at scale 1 and the view restored afterwards. */
  const redraw = async (dark) => {
    const parsed = withTheme(dark);
    shownDark = dark;
    const view = { scale, x, y, autoFit };
    paintSurface(dark);
    ready = false;
    canvas.style.transform = 'none';
    canvas.style.visibility = 'hidden';
    try {
      await window.tsncd.render(parsed.data, parsed.settings, parsed.auxiliary);
      await window.tsncd.settled();
    } finally {
      ready = true;
      captureRegions();
      width = canvas.offsetWidth;
      height = canvas.offsetHeight;
      if (view.autoFit) {
        fit();
      } else {
        ({ scale, x, y } = view);
        autoFit = false;
        paint();
      }
      canvas.style.visibility = '';
      buildKeys();
      applyPendingTheme();
    }
  };
  const setTheme = (dark) => {
    if (dark === null) return;
    if (!ready || !canvas || !window.tsncd) {
      pendingDark = dark;
      return;
    }
    pendingDark = null;
    if (dark !== showsDark()) redraw(dark);
  };
  const applyPendingTheme = () => {
    const dark = pendingDark;
    pendingDark = null;
    if (dark !== null) setTheme(dark);
  };

  window.addEventListener('message', (event) => {
    if (event.source !== window.parent || !event.data || typeof event.data !== 'object') return;
    const data = event.data;
    if (data.type === 'ncd:set-theme') {
      setTheme(asDark(data.theme));
    } else if (data.type === 'ncd:zoom' && ready) {
      if (data.fit) {
        fit();
      } else {
        autoFit = false;
        const next = Number.isFinite(data.scale) ? data.scale : scale * (data.step > 0 ? 1.25 : 0.8);
        zoomAbout(next, innerWidth / 2, Math.min(innerHeight, height * scale) / 2);
      }
    } else if (data.type === 'ncd:hello' && ready) {
      report();
    }
  });

  /* ------------------------------------------------------- inspection boxes */

  /* Each box is docked as the top of this file describes; the rule below wins over the position
     and size tsncd writes on the box. */
  const BOX_GAP = 14;
  const BOX_MARGIN = 8;
  const BOX_MIN_SIDE = 280;
  const BOX_MIN_HEIGHT = 200;
  const BOX_MIN_DIAGRAM = 160;
  const BOX_PADDING_X = 28;
  /* The smallest scale a diagram in a box is drawn at; a box too narrow for it scrolls sideways. */
  const BOX_MIN_ZOOM = 0.5;
  const boxStyle = document.createElement('style');
  boxStyle.textContent = `
    .inspection-box.ncd-docked {
      left: var(--ncd-left) !important;
      top: var(--ncd-top) !important;
      max-width: var(--ncd-width) !important;
      max-height: var(--ncd-height) !important;
      overflow: auto !important;
      scrollbar-gutter: stable;
    }
    .inspection-box.ncd-docked .katex-display {
      position: relative;
    }
    /* Firefox lets a box scroll over the unscaled size of a scaled diagram in its flow, but only
       over the drawn size of a float. */
    .inspection-box.ncd-docked > .inspection-box-diagram {
      float: left;
    }
    /* Links a finger can hit without opening the one beside it. */
    @media (pointer: coarse) {
      .inspection-box-references a {
        display: inline-block;
        padding: 5px 0;
      }
    }
  `;
  document.head.append(boxStyle);
  const anchors = new WeakMap();
  const margins = new WeakMap();
  const setVar = (node, name, value) => {
    if (node.style.getPropertyValue(name) !== value) node.style.setProperty(name, value);
  };
  /* Whether a box opened by a press on the block `key` is open. */
  const boxOpenFor = (key) =>
    Array.from(document.querySelectorAll('.inspection-box')).some((box) => anchors.get(box)?.block === key);

  const diagramOf = (box) => Array.from(box.children).find((child) => child.classList.contains('inspection-box-diagram'));
  /* The margins tsncd gave `diagram` for the ink outside it, [top, right, bottom, left]. */
  const marginsOf = (diagram) => {
    let margin = margins.get(diagram);
    if (!margin) {
      const style = diagram.style;
      margin = [style.marginTop, style.marginRight, style.marginBottom, style.marginLeft].map((m) => parseFloat(m) || 0);
      margins.set(diagram, margin);
    }
    return margin;
  };
  /* The width of the widest display formula of a box as drawn, which tsncd lets scroll sideways
     inside a box narrower than it. */
  const formulaWidth = (box) => {
    let widest = 0;
    for (const formula of box.querySelectorAll('.katex-display .katex-html')) {
      let left = Infinity;
      let right = -Infinity;
      for (const part of formula.children) {
        const bounds = part.getBoundingClientRect();
        if (bounds.width > 0) {
          left = Math.min(left, bounds.left);
          right = Math.max(right, bounds.right);
        }
      }
      if (right > left) widest = Math.max(widest, right - left);
    }
    return widest;
  };
  /* The narrowest box that shows its formulas whole and its diagram at BOX_MIN_ZOOM, and at least
     BOX_MIN_SIDE. */
  const readableWidth = (box, wanted) => {
    const chrome = box.offsetWidth - box.clientWidth + BOX_PADDING_X;
    let needed = Math.max(BOX_MIN_SIDE, Math.ceil(formulaWidth(box) + chrome));
    const diagram = diagramOf(box);
    if (diagram) {
      const [, right, , left] = marginsOf(diagram);
      needed = Math.max(needed, Math.ceil((left + diagram.offsetWidth + right) * BOX_MIN_ZOOM + chrome));
    }
    return Math.min(wanted, needed);
  };
  /* Whether `box` may cover the region it describes: only once it is locked, as its lock note says,
     since a click on the region of an open box must reach the region, and never over a tapped door. */
  const mayCover = (box) => {
    const opened = anchors.get(box);
    if (opened.tap && opened.block !== null && doorOf(opened.block)) return false;
    return (box.querySelector('.inspection-box-lock')?.textContent ?? '').includes('unlock');
  };
  /* The least height a box that may cover its region is docked above or below it at: its text and a
     real part of its diagram, BOX_MIN_DIAGRAM or a third of the frame, or BOX_MIN_HEIGHT without one. */
  const readableHeight = (box) => {
    const diagram = diagramOf(box);
    if (!diagram) return BOX_MIN_HEIGHT;
    const drawn = diagram.getBoundingClientRect();
    const above = drawn.top - box.getBoundingClientRect().top + box.scrollTop;
    const part = Math.min(drawn.height, Math.max(BOX_MIN_DIAGRAM, innerHeight / 3));
    return Math.max(BOX_MIN_HEIGHT, Math.min(box.scrollHeight, Math.ceil(above + part)));
  };

  /* Scale the diagram tsncd drew for a 1000 px box down to the box's width, margins included, but
     not below BOX_MIN_ZOOM, and take back the room the transform leaves with negative margins. */
  const fitDiagram = (box) => {
    const diagram = diagramOf(box);
    if (!diagram) return;
    const [top, right, bottom, left] = marginsOf(diagram);
    const drawnWidth = diagram.offsetWidth;
    const drawnHeight = diagram.offsetHeight;
    const room = box.clientWidth - BOX_PADDING_X;
    const wanted = room > 0 ? Math.min(1, room / (left + drawnWidth + right)) : 1;
    const zoom = Math.max(BOX_MIN_ZOOM, Math.floor(wanted * 1000) / 1000);
    if (drawingScale.get(diagram) === zoom) return;
    drawingScale.set(diagram, zoom);
    diagram.style.transformOrigin = '0 0';
    diagram.style.transform = zoom === 1 ? '' : `scale(${zoom})`;
    const shrunk = 1 - zoom;
    diagram.style.margin =
      `${top * zoom}px ${right * zoom - drawnWidth * shrunk}px ` +
      `${bottom * zoom - drawnHeight * shrunk}px ${left * zoom}px`;
  };

  /* Where a box goes around `around`, a rectangle in client coordinates: beside it on the side with
     more room when that is `need` wide, else above or below it when that is `tall`. */
  const placement = (around, wanted, need, tall, prefer = null) => {
    const room = {
      right: innerWidth - around.right - BOX_GAP - BOX_MARGIN,
      left: around.left - BOX_GAP - BOX_MARGIN,
      below: innerHeight - around.bottom - BOX_GAP - BOX_MARGIN,
      above: around.top - BOX_GAP - BOX_MARGIN,
    };
    const first = prefer ?? (room.right >= wanted || room.right >= room.left ? 'right' : 'left');
    const sides = first === 'right' ? ['right', 'left'] : ['left', 'right'];
    const side = sides.find((name) => room[name] >= need);
    if (side) return { side, width: Math.min(wanted, room[side]), height: innerHeight - 2 * BOX_MARGIN };
    const vertical = room.below >= room.above ? 'below' : 'above';
    if (room[vertical] < tall) return null;
    return { side: vertical, width: Math.min(wanted, innerWidth - 2 * BOX_MARGIN), height: room[vertical] };
  };
  /* The full width of the frame, in the half of it away from `around` or in all the room on that
     side where that is more, and all of the frame where that is not `tall`; the box may cover `around`. */
  const spanning = (around, wanted, tall) => {
    const lower = around.top + around.bottom < innerHeight;
    const room = lower ? innerHeight - around.bottom - BOX_GAP - BOX_MARGIN : around.top - BOX_GAP - BOX_MARGIN;
    const half = Math.max(room, innerHeight / 2 - BOX_MARGIN);
    const full = half < tall;
    const height = full ? innerHeight - 2 * BOX_MARGIN : half;
    return { side: lower ? 'lower' : 'upper', width: Math.min(wanted, innerWidth - 2 * BOX_MARGIN), height, full };
  };

  /* The rectangle a region of the figure covers on the screen now. The figure may have been drawn
     again since the box opened, so the region is looked up again by its key, nearest where it was. */
  const onScreen = (region) => {
    const away = (r) => Math.hypot(r.left - region.left, r.top - region.top);
    const found =
      rootRegions === null || rootRegions.includes(region)
        ? region
        : rootRegions
            .filter((r) => r.kind === region.kind && r.key === region.key)
            .reduce((a, b) => (a && away(a) <= away(b) ? a : b), null) ?? region;
    const origin = figure.getBoundingClientRect();
    const left = origin.left + found.left * scale;
    const top = origin.top + found.top * scale;
    return { left, top, right: left + found.width * scale, bottom: top + found.height * scale };
  };

  const dock = (box) => {
    const opened = anchors.get(box);
    const origin = ready && opened.at ? figure.getBoundingClientRect() : null;
    const at = origin ? { x: origin.left + opened.at.x * scale, y: origin.top + opened.at.y * scale } : opened;
    const point = { x: within(at.x, 0, innerWidth), y: within(at.y, 0, innerHeight) };
    const spot = { left: point.x, right: point.x, top: point.y, bottom: point.y };
    const wanted = parseFloat(box.style.width) || box.offsetWidth;
    const need = readableWidth(box, wanted);
    const covering = mayCover(box);
    const tall = covering ? readableHeight(box) : BOX_MIN_HEIGHT;
    /* Around the region the box describes, so a click on the region stays on it; around the point
       when the region leaves no room, on the side of the point that covers less of the region. */
    const region = origin && opened.region ? onScreen(opened.region) : null;
    const away = region && (point.x < (region.left + region.right) / 2 ? 'left' : 'right');
    let around = region;
    let place = region && placement(region, wanted, need, tall);
    if (!place) {
      around = spot;
      place = placement(spot, wanted, need, tall, away);
    }
    if (!place) {
      around = region || spot;
      place = spanning(around, wanted, covering ? tall : 0);
    }
    setVar(box, '--ncd-width', `${Math.floor(place.width)}px`);
    setVar(box, '--ncd-height', `${Math.floor(Math.max(120, place.height))}px`);
    fitDiagram(box);
    const shownWidth = box.offsetWidth;
    const shownHeight = box.offsetHeight;
    let boxX = within(point.x - shownWidth / 2, BOX_MARGIN, innerWidth - shownWidth - BOX_MARGIN);
    let boxY = within(point.y - shownHeight / 3, BOX_MARGIN, innerHeight - shownHeight - BOX_MARGIN);
    if (place.side === 'right') boxX = around.right + BOX_GAP;
    else if (place.side === 'left') boxX = around.left - BOX_GAP - shownWidth;
    else if (place.side === 'below') boxY = around.bottom + BOX_GAP;
    else if (place.side === 'above') boxY = Math.max(BOX_MARGIN, around.top - BOX_GAP - shownHeight);
    else if (place.side === 'lower') boxY = innerHeight - BOX_MARGIN - (place.full ? shownHeight : place.height);
    else boxY = place.full ? BOX_MARGIN : Math.max(BOX_MARGIN, BOX_MARGIN + place.height - shownHeight);
    setVar(box, '--ncd-left', `${Math.round(boxX + scrollX)}px`);
    setVar(box, '--ncd-top', `${Math.round(boxY + scrollY)}px`);
  };
  /* Where a box opened: the pointer in client coordinates, whether a tap opened it and, for a box
     opened from the figure, the pointer in figure units, the region and the key of the block under it. */
  const openedAt = () => {
    const opened = { ...pointer, tap: tapping(), at: null, region: null, block: null };
    if (!ready || !inFigure(pointerTarget)) return opened;
    const origin = figure.getBoundingClientRect();
    opened.at = { x: (pointer.x - origin.left) / scale, y: (pointer.y - origin.top) / scale };
    opened.region = regionAt(pointer.x, pointer.y);
    opened.block = blockAt(pointer.x, pointer.y)?.key ?? null;
    return opened;
  };
  const watchBox = (box) => {
    if (anchors.has(box)) return;
    anchors.set(box, openedAt());
    box.classList.add('ncd-docked');
    dock(box);
    new MutationObserver(() => dock(box)).observe(box, { childList: true });
    const note = box.querySelector('.inspection-box-lock');
    if (note) new MutationObserver(() => dock(box)).observe(note, { childList: true, characterData: true, subtree: true });
    new ResizeObserver(() => dock(box)).observe(box);
  };
  new MutationObserver((records) => {
    for (const record of records) {
      for (const node of record.addedNodes) {
        if (node instanceof HTMLElement && node.classList.contains('inspection-box')) watchBox(node);
      }
    }
  }).observe(document.body, { childList: true });
  /* A box docked over a still pointer takes the pointer from its region. Firefox then reports the
     mouse entering the box before the pointer leaving the region, which tsncd would read as leaving. */
  document.addEventListener(
    'pointerenter',
    (event) => {
      const box = event.target;
      if (event.pointerType === 'mouse' && box instanceof Element && box.classList.contains('inspection-box')) {
        box.dispatchEvent(new MouseEvent('mouseenter'));
      }
    },
    true,
  );

  /* --------------------------------------------------------------- keyboard */

  /* Each region tsncd opens a box for is a button of the toolbar, as the top of this file describes.
     The buttons take no pointer events, so tsncd's hit-testing sees the figure alone. */
  const SYMBOLS = {
    alpha: 'α', beta: 'β', gamma: 'γ', delta: 'δ', epsilon: 'ε', kappa: 'κ', lambda: 'λ', mu: 'μ', pi: 'π',
    rho: 'ρ', sigma: 'σ', tau: 'τ', phi: 'φ', omega: 'ω', lvert: '|', rvert: '|', odot: '⊙', cdot: '·', times: '×',
  };
  /* A label written in LaTeX as plain text, for a screen reader: `\bold{W^{c}}_{t,\mathrm{CA}}`
     reads `W^c_(t,CA)`. */
  const plainText = (latex) => {
    let text = latex.replace(/\\[,;:! ]/g, ' ');
    let before;
    do {
      before = text;
      text = text
        .replace(/\\bar\{([^{}]*)\}/g, '$1\u0304')
        .replace(/\\[A-Za-z]+\{([^{}]*)\}/g, '$1')
        .replace(/([_^])\{([^{}]*)\}/g, (all, mark, inner) => (/^(\w+|\\[A-Za-z]+)$/.test(inner) ? mark + inner : `${mark}(${inner})`));
    } while (text !== before);
    return text
      .replace(/\\([A-Za-z]+)/g, (all, name) => (Object.hasOwn(SYMBOLS, name) ? SYMBOLS[name] : ` ${name} `))
      .replace(/[{}]/g, '')
      .replace(/\s+/g, ' ')
      .replace(/\(\s/g, '(')
      .trim();
  };
  const expansions = () => JSON.parse(message.textContent).auxiliary?.expansions || {};
  /* The name and the description a screen reader gives the button of `region`. */
  const describe = (region) => {
    if (region.kind === 'block') {
      const info = blocks()[region.key];
      const door = doorOf(region.key) ? 'Enter opens its own diagram. ' : '';
      return { name: plainText(info.title || ''), description: door + (info.description || '') };
    }
    const info = expansions()[region.key] || {};
    const label = info.latex ?? info.operator ?? '';
    const name = /^\w+$/.test(label) ? label : `${info.operator || ''} ${plainText(label)}`.trim();
    return { name, description: info.description || '' };
  };
  /* A point of `region`, in figure units, where no smaller region lies, so a pointer there opens its box. */
  const pointIn = (region) => {
    const at = [0.5, 0.3, 0.7, 0.15, 0.85];
    for (const fy of at) {
      for (const fx of at) {
        const px = region.left + region.width * fx;
        const py = region.top + region.height * fy;
        if (regionAtFigure(px, py) === region) return { x: px, y: py };
      }
    }
    return null;
  };

  let keys = null;
  let keyTargets = [];
  let keyAt = 0;
  /* The events the toolbar sends to the figure, which reach tsncd as a mouse's would. */
  const synthetic = new WeakSet();
  const keyOf = (node) => (keys !== null && node instanceof Element ? keyTargets.find((t) => t.button === node) : undefined);
  const rove = () => keyTargets.forEach((t, index) => t.button.setAttribute('tabindex', index === keyAt ? '0' : '-1'));
  /* Each button covers its region on the screen, inside the frame, so its outline rings the region. */
  const placeKey = (target, origin) => {
    const left = within(origin.left + target.region.left * scale, 3, innerWidth - 3);
    const top = within(origin.top + target.region.top * scale, 3, innerHeight - 3);
    const right = within(origin.left + (target.region.left + target.region.width) * scale, 3, innerWidth - 3);
    const bottom = within(origin.top + (target.region.top + target.region.height) * scale, 3, innerHeight - 3);
    const style = target.button.style;
    style.left = `${left}px`;
    style.top = `${top}px`;
    style.width = `${right - left}px`;
    style.height = `${bottom - top}px`;
  };
  let keysPlaced = true;
  const placeKeys = () => {
    if (!keysPlaced || !keyTargets.length) return;
    keysPlaced = false;
    requestAnimationFrame(() => {
      keysPlaced = true;
      const origin = figure.getBoundingClientRect();
      for (const target of keyTargets) placeKey(target, origin);
    });
  };
  const clientPoint = (target) => {
    const origin = figure.getBoundingClientRect();
    return { x: origin.left + target.point.x * scale, y: origin.top + target.point.y * scale };
  };
  /* Pan the camera just enough to show `region`, or centre it on a region larger than the frame. */
  const showRegion = (region) => {
    const shown = onScreen(region);
    const margin = 24;
    const shift = (low, high, size) => {
      if (high - low > size - 2 * margin) return size / 2 - (low + high) / 2;
      if (low < margin) return margin - low;
      return high > size - margin ? size - margin - high : 0;
    };
    const dx = shift(shown.left, shown.right, innerWidth);
    const dy = shift(shown.top, shown.bottom, innerHeight);
    if (!dx && !dy) return;
    x += dx;
    y += dy;
    paint();
  };
  let keyOpened = false;
  /* Open the box of `target` as a mouse resting on it would, and dock it beside this region even
     when a box of the same block is open beside another. */
  const openKey = (target) => {
    const at = clientPoint(target);
    const move = { bubbles: true, composed: true, pointerId: 1, pointerType: 'mouse', isPrimary: true, clientX: at.x, clientY: at.y };
    figure.dispatchEvent(new PointerEvent('pointermove', move));
    keyOpened = true;
    for (const box of document.querySelectorAll('.inspection-box')) {
      const region = anchors.get(box)?.region;
      if (!region || region === target.region || region.kind !== target.region.kind || region.key !== target.region.key) continue;
      anchors.set(box, openedAt());
      dock(box);
    }
  };
  const lockKey = (target) => {
    const at = clientPoint(target);
    const click = new MouseEvent('click', { bubbles: true, cancelable: true, composed: true, button: 0, clientX: at.x, clientY: at.y });
    synthetic.add(click);
    figure.dispatchEvent(click);
  };
  /* Close the boxes the toolbar opened, unless they are locked, as when a mouse leaves the figure. */
  const releaseKeys = () => {
    if (!keyOpened || figure === null) return;
    keyOpened = false;
    figure.dispatchEvent(new PointerEvent('pointerleave', { pointerId: 1, pointerType: 'mouse', isPrimary: true }));
  };
  const activateKey = (target, { newTab = false, lock = false } = {}) => {
    const door = target.region.kind === 'block' ? doorOf(target.region.key) : null;
    if (door && (newTab || !lock)) {
      follow(door, newTab, true);
      return;
    }
    openKey(target);
    lockKey(target);
  };
  const focusKey = (index) => {
    keyAt = within(index, 0, keyTargets.length - 1);
    rove();
    keyTargets[keyAt].button.focus({ preventScroll: true });
  };

  const installKeys = () => {
    keys = document.createElement('div');
    keys.className = 'ncd-keys';
    keys.setAttribute('role', 'toolbar');
    keys.setAttribute('aria-label', 'Blocks and operations of the diagram');
    const help = document.createElement('span');
    help.id = 'ncd-keys-help';
    help.hidden = true;
    help.textContent =
      'Arrow keys move between them and show each one’s details. Enter opens a block’s own diagram, where it has one, ' +
      'or keeps the details open. Tab goes into the details, and Escape closes them.';
    keys.setAttribute('aria-describedby', help.id);
    keys.append(help);
    canvas.after(keys);
    keys.addEventListener('focusin', (event) => {
      const target = keyOf(event.target);
      if (!target || !ready) return;
      keyAt = keyTargets.indexOf(target);
      rove();
      showFrameRing(false);
      showRegion(target.region);
      placeKey(target, figure.getBoundingClientRect());
      openKey(target);
    });
    keys.addEventListener('keydown', (event) => {
      const target = keyOf(event.target);
      if (!target || event.ctrlKey || event.metaKey || (event.altKey && event.key !== 'Enter')) return;
      const index = keyTargets.indexOf(target);
      const next = index + 1;
      const previous = index - 1;
      const move = { ArrowRight: next, ArrowDown: next, ArrowLeft: previous, ArrowUp: previous, Home: 0, End: Infinity }[event.key];
      if (move !== undefined && !event.altKey) {
        event.preventDefault();
        focusKey(move);
      }
    });
    /* Enter is read on key down, where its modifiers are known; a click comes from Space or from a
       screen reader. */
    keys.addEventListener('keydown', (event) => {
      const target = keyOf(event.target);
      if (!target || event.key !== 'Enter' || !ready) return;
      event.preventDefault();
      activateKey(target, { newTab: wantsNewTab(event), lock: event.shiftKey });
    });
    keys.addEventListener('click', (event) => {
      const target = keyOf(event.target);
      if (!target) return;
      event.stopPropagation();
      if (ready) activateKey(target);
    });
  };
  /* Called once the camera shows the figure; buttons whose region is unchanged are kept, so a
     redraw leaves the focus where it was. */
  const buildKeys = () => {
    if (rootRegions === null || canvas === null) return;
    if (keys === null) installKeys();
    const kept = new Set();
    const old = keyTargets;
    const focused = old.some((target) => target.button === document.activeElement);
    keyTargets = rootRegions
      .filter((region) => region.width > 0 && region.height > 0)
      .map((region) => ({ region, point: pointIn(region) }))
      .filter((target) => target.point !== null)
      .map((target, index) => {
        const same = old[index];
        const reused = same && same.region.kind === target.region.kind && same.region.key === target.region.key;
        const button = reused ? same.button : document.createElement('button');
        if (reused) kept.add(button);
        const note = reused ? same.note : document.createElement('span');
        const { name, description } = describe(target.region);
        button.type = 'button';
        button.className = 'ncd-key';
        button.setAttribute('aria-label', name);
        note.hidden = true;
        note.id = `ncd-key-note-${index}`;
        note.textContent = description;
        button.setAttribute('aria-describedby', note.id);
        return { ...target, button, note };
      });
    for (const target of old) {
      if (!kept.has(target.button)) {
        target.button.remove();
        target.note.remove();
      }
    }
    /* Moving a focused button in the document takes its focus, so an unchanged toolbar stays put. */
    if (kept.size !== old.length || keyTargets.length !== old.length) {
      keys.append(...keyTargets.flatMap((target) => [target.button, target.note]));
    }
    keyAt = within(keyAt, 0, Math.max(0, keyTargets.length - 1));
    rove();
    const origin = figure.getBoundingClientRect();
    for (const target of keyTargets) placeKey(target, origin);
    if (focused && keyTargets.length && !keyOf(document.activeElement)) focusKey(keyAt);
  };
  /* Focus leaving the toolbar and the boxes for anything else closes what the toolbar opened. */
  document.addEventListener('focusout', (event) => {
    const ours = (node) => keyOf(node) || (node instanceof Element && node.closest('.inspection-box'));
    if (ours(event.target) && !ours(event.relatedTarget)) releaseKeys();
  });

  /* The ring around the whole frame shows where the keys go while the page itself has focus: after
     keyboard focus enters the page or a key is pressed in it, and not after a press of the pointer. */
  let lastInput = null;
  const showFrameRing = (on) => {
    if (on) root.setAttribute('data-ncd-focused', '');
    else root.removeAttribute('data-ncd-focused');
  };
  const pageFocused = () =>
    framed && document.hasFocus() && [document.body, root, null].includes(document.activeElement);
  /* What the parent page asked for when it last focused this frame, read once from the frame element. */
  let asked = null;
  const readAsked = () => {
    const element = window.frameElement;
    if (element?.dataset.ncdFocus) {
      asked = element.dataset.ncdFocus;
      delete element.dataset.ncdFocus;
    }
  };
  /* Firefox focuses a frame's document, not its window, when Tab enters the frame. */
  window.addEventListener(
    'focus',
    (event) => {
      if (event.target !== window && event.target !== document) return;
      readAsked();
      showFrameRing(pageFocused() && (asked === 'visible' || (asked !== 'quiet' && lastInput !== 'pointer')));
    },
    true,
  );
  window.addEventListener(
    'blur',
    (event) => {
      if (event.target !== window && event.target !== document) return;
      asked = null;
      showFrameRing(false);
    },
    true,
  );
  document.addEventListener(
    'keydown',
    (event) => {
      if (!event.isTrusted) return;
      lastInput = 'key';
      if (pageFocused() && !['Shift', 'Control', 'Alt', 'Meta', 'Tab'].includes(event.key)) showFrameRing(true);
    },
    true,
  );

  window.addEventListener('resize', () => {
    if (ready) measure();
    for (const box of document.querySelectorAll('.inspection-box.ncd-docked')) dock(box);
  });

  /* The camera starts once tsncd's first draw has settled. A box opened while the page loaded was
     placed without the camera, so it is closed as Escape would close it. */
  const start = () => {
    if (!window.tsncd) {
      setTimeout(start, 50);
      return;
    }
    window.tsncd.settled().then(() => {
      if (!installCamera()) return;
      if (window.tsncd.openBoxes() > 0) document.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape' }));
      ready = true;
      captureRegions();
      new ResizeObserver(() => measure()).observe(canvas);
      measure();
      buildKeys();
      root.dataset.ncdReady = 'true';
      applyPendingTheme();
    });
  };
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', start);
  else start();
})();
