/* scene3d.js: a small WebGL2 renderer for the visualisation pages (docs/architecture/viz/).
 *
 * Content is kept in named sets, each drawn in one or a few instanced draw calls:
 *
 *   addBoxes(name, {positions, sizes, colors, alphas, edges, ids, pickable})
 *       axis-aligned boxes, centre + size per instance, flat-shaded with crisp anti-aliased edges
 *       (drawn in the fragment shader, so they stay one device pixel wide at any zoom). A set with
 *       any alpha < 1 is glass: sorted back to front every frame, back faces first.
 *   addPlane(name, {origin, u, v | box, face, values, width, height, domain, colormap, ...})
 *       a rectangle coloured by a small typed array (row-major, `height` rows of `width` values):
 *       an L x 768 slab face, an L x L pair map. Values go to an 8-bit texture read with texelFetch
 *       (crisp cells), colours come from a 256-entry ramp that follows the theme; NaN is empty.
 *       Cell grid lines appear once cells are 7 px wide. Picking reports the cell and its value.
 *   addLines(name, {paths | segments, colors, width, dash, onTop})
 *       screen-space lines `width` CSS px wide, clipped at the near plane.
 *   addTubes(name, {paths, radius, colors, joints, arrow}) and addSpheres(name, {positions, radius})
 *       shaded cylinders (with joint spheres and an arrowhead cone) and spheres, in world units.
 *   addLabel({position, text | html | node, anchor, offset, priority, maxDistance, interactive, leader})
 *       an HTML label pinned to a projected 3D point: real text in the small sans on a surface halo,
 *       crisp at any zoom, hidden behind the camera or off the canvas and, with `declutter`, where a
 *       label of higher priority overlaps; at most `maxLabels` show at once (8, or 5 on a canvas under
 *       600 px), the highest priorities first. `leader` draws a 1 px line from the point to the label.
 *
 * Colours are specs: a palette role ('pair', see palette.js), a custom property ('--wf-ink'), any
 * CSS colour, or [r, g, b(, a)] in 0..1; a set's `colors` is one spec or one per instance. Boxes
 * and spheres draw a role in its `-face` tone ('pair' is '--viz-pair-face'), tubes too except in the
 * dark theme, where they keep the base; a custom property or CSS colour is used as given. Shading
 * darkens a face without greying it, with palette.js's SHADING. Specs are resolved on the host
 * element, so they follow the docs theme: the scene restyles itself on every theme change (onTheme),
 * and restyle() re-reads them on demand.
 *
 * Camera: an orbit camera {target: [x, y, z], distance, yaw, pitch} (angles in degrees; yaw 0 looks
 * down -z, pitch > 0 looks down on the scene), perspective (fov degrees) or orthographic. Mouse:
 * drag orbits, Shift/right/middle drag pans, Ctrl/Cmd + wheel zooms at the pointer (a bare wheel
 * scrolls the page and shows a hint; `wheel: 'always'` zooms). Touch: one finger orbits (with
 * `touchAction: 'pan-y'`, the default, a vertical swipe scrolls the page), two fingers pinch and pan.
 * Keyboard on the focused canvas: arrows orbit 10 degrees (Shift pans), + and - zoom, 0 or Home
 * returns to the home view, Escape clears the selection. setCamera jumps, flyTo eases (a Promise,
 * true on arrival), fit/cameraFor frame bounds. Under prefers-reduced-motion flights jump.
 *
 * Picking is a CPU ray cast (boxes, spheres, tubes, planes; lines within a few pixels), so it is
 * exact and needs no GPU read-back. The scene dispatches `hover` and `select` (detail: a hit
 * {set, kind, index, id, point, t, and col/row/value on planes, segment on tubes and lines} or
 * null) and `view` (detail: the camera). With autoHover/autoSelect (default on) it also marks the
 * hovered and clicked instance: the hover colour rings a box, the selection colour a selected one.
 * set.setSelected(indices), set.setDimmed(indices) and plane.setHighlight(rect, 'hover'|'select')
 * mark instances from page code.
 *
 * It renders on demand (one frame per change, a loop only while a flight runs), sizes the
 * drawing buffer to the canvas at min(devicePixelRatio, 2) through a ResizeObserver, survives a
 * lost WebGL context (it keeps every array and rebuilds on restore) and throws an Error with
 * `code = 'webgl2-unavailable'` from the constructor without WebGL2, which page.js turns into the
 * poster and a short message. The overlay of labels is aria-hidden unless a label is interactive:
 * a page describes its scene in text and offers native controls for everything the scene does.
 *
 *   import { Scene3D } from './lib/scene3d.js';
 *   const scene = new Scene3D(host, {label: 'The trunk, one box per tensor', camera: {yaw: 30, pitch: 20}});
 *   const layers = scene.addBoxes('layers', {positions, sizes, colors: 'single', alphas: 0.9});
 *   scene.addLabel({position: [0, 1.2, 0], text: 'layer 1', anchor: 'bottom'});
 *   scene.addEventListener('select', (event) => inspect(event.detail));
 *   await scene.flyTo(scene.cameraFor(layers.bounds(), {yaw: 60}));
 */

import { onTheme, reducedMotion, theme as currentTheme } from './theme.js';
import { SHADING, rampTable, resolveColor, toned } from './palette.js';

const DEG = Math.PI / 180;
const MAX_DPR = 2;
const CLICK_SLOP_PX = 5;
const LINE_PICK_PX = 5;

/* ------------------------------------------------------------------ vectors and matrices
   Matrices are column-major Float32Array(16), as WebGL reads them. */
const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const add = (a, b) => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
const scale = (a, s) => [a[0] * s, a[1] * s, a[2] * s];
const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const length = (a) => Math.hypot(a[0], a[1], a[2]);
const normalize = (a) => scale(a, 1 / (length(a) || 1));

function lookAt(eye, target, up) {
  const z = normalize(sub(eye, target));
  const x = normalize(cross(up, z));
  const y = cross(z, x);
  return new Float32Array([x[0], y[0], z[0], 0, x[1], y[1], z[1], 0, x[2], y[2], z[2], 0, -dot(x, eye), -dot(y, eye), -dot(z, eye), 1]);
}
function perspective(fovy, aspect, near, far) {
  const f = 1 / Math.tan(fovy / 2);
  const nf = 1 / (near - far);
  return new Float32Array([f / aspect, 0, 0, 0, 0, f, 0, 0, 0, 0, (far + near) * nf, -1, 0, 0, 2 * far * near * nf, 0]);
}
function orthographic(halfW, halfH, near, far) {
  const nf = 1 / (near - far);
  return new Float32Array([1 / halfW, 0, 0, 0, 0, 1 / halfH, 0, 0, 0, 0, 2 * nf, 0, 0, 0, (far + near) * nf, 1]);
}
function multiply(a, b) {
  const out = new Float32Array(16);
  for (let c = 0; c < 4; c++) {
    for (let r = 0; r < 4; r++) {
      out[c * 4 + r] = a[r] * b[c * 4] + a[4 + r] * b[c * 4 + 1] + a[8 + r] * b[c * 4 + 2] + a[12 + r] * b[c * 4 + 3];
    }
  }
  return out;
}
function invert(m) {
  const [a00, a01, a02, a03, a10, a11, a12, a13, a20, a21, a22, a23, a30, a31, a32, a33] = m;
  const b00 = a00 * a11 - a01 * a10, b01 = a00 * a12 - a02 * a10, b02 = a00 * a13 - a03 * a10;
  const b03 = a01 * a12 - a02 * a11, b04 = a01 * a13 - a03 * a11, b05 = a02 * a13 - a03 * a12;
  const b06 = a20 * a31 - a21 * a30, b07 = a20 * a32 - a22 * a30, b08 = a20 * a33 - a23 * a30;
  const b09 = a21 * a32 - a22 * a31, b10 = a21 * a33 - a23 * a31, b11 = a22 * a33 - a23 * a32;
  const det = 1 / (b00 * b11 - b01 * b10 + b02 * b09 + b03 * b08 - b04 * b07 + b05 * b06);
  return new Float32Array([
    (a11 * b11 - a12 * b10 + a13 * b09) * det, (a02 * b10 - a01 * b11 - a03 * b09) * det,
    (a31 * b05 - a32 * b04 + a33 * b03) * det, (a22 * b04 - a21 * b05 - a23 * b03) * det,
    (a12 * b08 - a10 * b11 - a13 * b07) * det, (a00 * b11 - a02 * b08 + a03 * b07) * det,
    (a32 * b02 - a30 * b05 - a33 * b01) * det, (a20 * b05 - a22 * b02 + a23 * b01) * det,
    (a10 * b10 - a11 * b08 + a13 * b06) * det, (a01 * b08 - a00 * b10 - a03 * b06) * det,
    (a30 * b04 - a31 * b02 + a33 * b00) * det, (a21 * b02 - a20 * b04 - a23 * b00) * det,
    (a11 * b07 - a10 * b09 - a12 * b06) * det, (a00 * b09 - a01 * b07 + a02 * b06) * det,
    (a31 * b01 - a30 * b03 - a32 * b00) * det, (a20 * b03 - a21 * b01 + a22 * b00) * det,
  ]);
}
const transform = (m, [x, y, z]) => [
  m[0] * x + m[4] * y + m[8] * z + m[12],
  m[1] * x + m[5] * y + m[9] * z + m[13],
  m[2] * x + m[6] * y + m[10] * z + m[14],
  m[3] * x + m[7] * y + m[11] * z + m[15],
];

/** Ease in and out, cubic: the default for flights. */
export const easeInOutCubic = (t) => (t < 0.5 ? 4 * t * t * t : 1 - (-2 * t + 2) ** 3 / 2);

/* ------------------------------------------------------------------ bounds */
const emptyBounds = () => ({ min: [Infinity, Infinity, Infinity], max: [-Infinity, -Infinity, -Infinity] });
const grow = (b, p, r = 0) => {
  for (let k = 0; k < 3; k++) {
    b.min[k] = Math.min(b.min[k], p[k] - r);
    b.max[k] = Math.max(b.max[k], p[k] + r);
  }
  return b;
};
const unionBounds = (a, b) => grow(grow(a, b.min), b.max);
const boundsValid = (b) => b.min[0] <= b.max[0];

/**
 * The frame of one face of an axis-aligned box, for addPlane: as seen from outside the face, `u`
 * runs left to right and `v` top to bottom, so row 0 of a texture lies along the top edge. Top
 * (+y) and bottom (-y) faces read with their back (-z) edge up.
 * @param {{center: number[], size: number[]}} box
 * @param {'+x'|'-x'|'+y'|'-y'|'+z'|'-z'} face
 * @returns {{origin: number[], u: number[], v: number[]}}
 */
export function faceFrame({ center: [cx, cy, cz], size: [sx, sy, sz] }, face) {
  const [hx, hy, hz] = [sx / 2, sy / 2, sz / 2];
  const frames = {
    '+z': { origin: [cx - hx, cy + hy, cz + hz], u: [sx, 0, 0], v: [0, -sy, 0] },
    '-z': { origin: [cx + hx, cy + hy, cz - hz], u: [-sx, 0, 0], v: [0, -sy, 0] },
    '+x': { origin: [cx + hx, cy + hy, cz + hz], u: [0, 0, -sz], v: [0, -sy, 0] },
    '-x': { origin: [cx - hx, cy + hy, cz - hz], u: [0, 0, sz], v: [0, -sy, 0] },
    '+y': { origin: [cx - hx, cy + hy, cz - hz], u: [sx, 0, 0], v: [0, 0, sz] },
    '-y': { origin: [cx - hx, cy - hy, cz + hz], u: [sx, 0, 0], v: [0, 0, -sz] },
  };
  if (!frames[face]) throw new Error(`face must be one of ${Object.keys(frames)}, got ${face}`);
  return frames[face];
}

/** True when this browser can create a WebGL2 context. */
export function webgl2Available() {
  if (typeof document === 'undefined') return false;
  try {
    return Boolean(document.createElement('canvas').getContext('webgl2'));
  } catch {
    return false;
  }
}

/* ------------------------------------------------------------------ geometry */
function cube() {
  const faces = [
    [[1, 0, 0], [0, 1, 0], [0, 0, 1]], [[-1, 0, 0], [0, 1, 0], [0, 0, -1]],
    [[0, 1, 0], [0, 0, 1], [1, 0, 0]], [[0, -1, 0], [0, 0, -1], [1, 0, 0]],
    [[0, 0, 1], [1, 0, 0], [0, 1, 0]], [[0, 0, -1], [-1, 0, 0], [0, 1, 0]],
  ];
  const position = [], normal = [], index = [];
  faces.forEach(([n, a, b], f) => {
    for (const [s, t] of [[-1, -1], [1, -1], [1, 1], [-1, 1]]) {
      for (let k = 0; k < 3; k++) position.push(0.5 * (n[k] + s * a[k] + t * b[k]));
      normal.push(...n);
    }
    /* Wind every face counter-clockwise seen from outside. */
    const flip = dot(cross(a, b), n) < 0;
    const o = 4 * f;
    index.push(...(flip ? [o, o + 2, o + 1, o, o + 3, o + 2] : [o, o + 1, o + 2, o, o + 2, o + 3]));
  });
  return { position, normal, index };
}
function cylinder(n = 12) {
  const position = [], normal = [], index = [];
  for (let i = 0; i <= n; i++) {
    const a = (i / n) * 2 * Math.PI, c = Math.cos(a), s = Math.sin(a);
    position.push(c, 0, s, c, 1, s);
    normal.push(c, 0, s, c, 0, s);
    if (i < n) index.push(2 * i, 2 * i + 1, 2 * i + 2, 2 * i + 1, 2 * i + 3, 2 * i + 2);
  }
  return { position, normal, index };
}
function cone(n = 16) {
  const position = [], normal = [], index = [];
  for (let i = 0; i <= n; i++) {
    const a = (i / n) * 2 * Math.PI, c = Math.cos(a), s = Math.sin(a);
    position.push(c, 0, s, 0, 1, 0);
    normal.push(c, 0.5, s, c, 0.5, s);
    if (i < n) index.push(2 * i, 2 * i + 1, 2 * i + 2);
  }
  const base = position.length / 3;
  position.push(0, 0, 0);
  normal.push(0, -1, 0);
  for (let i = 0; i <= n; i++) {
    const a = (i / n) * 2 * Math.PI;
    position.push(Math.cos(a), 0, Math.sin(a));
    normal.push(0, -1, 0);
    if (i < n) index.push(base, base + 1 + i + 1, base + 1 + i);
  }
  return { position, normal, index };
}
function sphere(rings = 10, segments = 16) {
  const position = [], normal = [], index = [];
  for (let r = 0; r <= rings; r++) {
    const phi = (r / rings) * Math.PI;
    for (let s = 0; s <= segments; s++) {
      const theta = (s / segments) * 2 * Math.PI;
      const p = [Math.sin(phi) * Math.cos(theta), Math.cos(phi), Math.sin(phi) * Math.sin(theta)];
      position.push(...p);
      normal.push(...p);
      if (r < rings && s < segments) {
        const a = r * (segments + 1) + s, b = a + segments + 1;
        index.push(a, a + 1, b, a + 1, b + 1, b);
      }
    }
  }
  return { position, normal, index };
}

/* ------------------------------------------------------------------ shaders */
const HEADER = '#version 300 es\nprecision highp float;\nprecision highp int;\n';
const SHADE = `
uniform vec3 uLight; uniform float uAmbient; uniform vec3 uInk; uniform vec3 uBg;
uniform vec3 uHover; uniform vec3 uSelect; uniform float uDim;
vec3 shade(vec3 c, vec3 n) {
  float k = uAmbient + (1.0 - uAmbient) * max(dot(normalize(n), uLight), 0.0);
  vec3 d = c * k;
  float y = dot(d, vec3(0.2126, 0.7152, 0.0722));
  return clamp(mix(vec3(y), d, 1.0 + (1.0 - k) * 0.9), 0.0, 1.0);
}
`;
const BOX_VS = `${HEADER}
layout(location=0) in vec3 aPos; layout(location=1) in vec3 aNormal;
layout(location=2) in vec3 iCenter; layout(location=3) in vec3 iSize; layout(location=4) in vec4 iColor;
layout(location=5) in float iState;
uniform mat4 uViewProj;
out vec3 vLocal; out vec3 vNormal; out vec4 vColor; flat out float vState;
void main() {
  vLocal = aPos; vNormal = aNormal; vColor = iColor; vState = iState;
  gl_Position = uViewProj * vec4(iCenter + aPos * iSize, 1.0);
}`;
const BOX_FS = `${HEADER}${SHADE}
in vec3 vLocal; in vec3 vNormal; in vec4 vColor; flat in float vState;
uniform float uEdgeWidth; uniform float uEdgeMix; uniform float uEdgeAlpha;
out vec4 outColor;
void main() {
  vec3 n = normalize(vNormal);
  vec2 f = abs(n.x) > 0.5 ? vLocal.yz : abs(n.y) > 0.5 ? vLocal.xz : vLocal.xy;
  vec2 px = (0.5 - abs(f)) / max(fwidth(f), vec2(1e-7));
  float width = uEdgeWidth * (vState > 0.5 ? 2.5 : 1.0);
  float edge = width > 0.0 ? 1.0 - smoothstep(width - 0.5, width + 0.5, min(px.x, px.y)) : 0.0;
  vec3 base = shade(vColor.rgb, n);
  vec3 ink = mix(vColor.rgb, uInk, uEdgeMix);
  float alpha = vColor.a;
  float edgeAlpha = max(alpha, uEdgeAlpha);
  if (vState > 1.5) { ink = uSelect; base = mix(base, uSelect, 0.18); edgeAlpha = 1.0; }
  else if (vState > 0.5) { ink = uHover; base = mix(base, uHover, 0.12); edgeAlpha = 1.0; }
  else if (vState < -0.5) { alpha *= uDim; edgeAlpha *= uDim; base = mix(base, uBg, 0.55); ink = mix(ink, uBg, 0.55); }
  vec3 color = mix(base, ink, edge);
  float a = mix(alpha, edgeAlpha, edge);
  if (a < 0.002) discard;
  outColor = vec4(color * a, a);
}`;
const MESH_VS = `${HEADER}
layout(location=0) in vec3 aPos; layout(location=1) in vec3 aNormal;
layout(location=2) in vec3 iA; layout(location=3) in vec3 iB; layout(location=4) in float iRadius;
layout(location=5) in vec4 iColor; layout(location=6) in float iState;
uniform mat4 uViewProj; uniform int uKind;
out vec3 vNormal; out vec4 vColor; flat out float vState;
void main() {
  vec3 world; vec3 normal;
  if (uKind == 0) { world = iA + aPos * iRadius; normal = aNormal; }
  else {
    vec3 axis = iB - iA; float len = length(axis);
    vec3 y = len > 1e-9 ? axis / len : vec3(0.0, 1.0, 0.0);
    vec3 t = abs(y.y) < 0.99 ? vec3(0.0, 1.0, 0.0) : vec3(1.0, 0.0, 0.0);
    vec3 x = normalize(cross(t, y)); vec3 z = cross(x, y);
    world = iA + y * aPos.y * len + (x * aPos.x + z * aPos.z) * iRadius;
    normal = x * aNormal.x + y * aNormal.y + z * aNormal.z;
  }
  vNormal = normal; vColor = iColor; vState = iState;
  gl_Position = uViewProj * vec4(world, 1.0);
}`;
const MESH_FS = `${HEADER}${SHADE}
in vec3 vNormal; in vec4 vColor; flat in float vState;
out vec4 outColor;
void main() {
  vec3 c = shade(vColor.rgb, vNormal);
  float a = vColor.a;
  if (vState > 1.5) c = mix(c, uSelect, 0.45);
  else if (vState > 0.5) c = mix(c, uHover, 0.35);
  else if (vState < -0.5) { a *= uDim; c = mix(c, uBg, 0.55); }
  if (a < 0.002) discard;
  outColor = vec4(c * a, a);
}`;
const LINE_VS = `${HEADER}
layout(location=0) in vec2 aCorner;
layout(location=1) in vec3 iA; layout(location=2) in vec3 iB; layout(location=3) in vec4 iColor;
layout(location=4) in float iState;
uniform mat4 uView; uniform mat4 uProj; uniform vec2 uViewport; uniform float uWidth; uniform float uNear;
uniform bool uOrtho;
out vec4 vColor; out float vSide; out float vAlong; out float vHalf; flat out float vState;
void main() {
  vec4 a = uView * vec4(iA, 1.0); vec4 b = uView * vec4(iB, 1.0);
  vColor = iColor; vState = iState;
  if (!uOrtho) {
    if (a.z > -uNear && b.z > -uNear) { gl_Position = vec4(2.0, 2.0, 2.0, 1.0); return; }
    if (a.z > -uNear) a = mix(a, b, (a.z + uNear) / (a.z - b.z));
    if (b.z > -uNear) b = mix(b, a, (b.z + uNear) / (b.z - a.z));
  }
  vec4 ca = uProj * a; vec4 cb = uProj * b;
  vec2 sa = ca.xy / ca.w * uViewport * 0.5; vec2 sb = cb.xy / cb.w * uViewport * 0.5;
  vec2 d = sb - sa; float len = length(d); d = len > 1e-5 ? d / len : vec2(1.0, 0.0);
  vec2 n = vec2(-d.y, d.x);
  float half_ = 0.5 * uWidth * (iState > 0.5 ? 1.8 : 1.0) + 1.0;
  vec4 c = aCorner.x < 0.5 ? ca : cb;
  vec2 offset = n * aCorner.y * half_ + d * (aCorner.x * 2.0 - 1.0) * half_ * 0.5;
  c.xy += offset / (uViewport * 0.5) * c.w;
  gl_Position = c;
  vSide = aCorner.y * half_; vHalf = half_; vAlong = aCorner.x * len + (aCorner.x * 2.0 - 1.0) * half_ * 0.5;
}`;
const LINE_FS = `${HEADER}
in vec4 vColor; in float vSide; in float vAlong; in float vHalf; flat in float vState;
uniform vec2 uDash; uniform vec3 uHover; uniform vec3 uSelect; uniform vec3 uBg; uniform float uDim;
out vec4 outColor;
void main() {
  if (uDash.x > 0.0 && mod(vAlong, uDash.x + uDash.y) > uDash.x) discard;
  float a = vColor.a * (1.0 - smoothstep(vHalf - 1.0, vHalf, abs(vSide)));
  vec3 c = vColor.rgb;
  if (vState > 1.5) c = uSelect; else if (vState > 0.5) c = uHover;
  else if (vState < -0.5) { a *= uDim; c = mix(c, uBg, 0.55); }
  if (a < 0.002) discard;
  outColor = vec4(c * a, a);
}`;
const PLANE_VS = `${HEADER}
layout(location=0) in vec2 aCorner;
uniform mat4 uViewProj; uniform vec3 uOrigin; uniform vec3 uU; uniform vec3 uV;
out vec2 vUV;
void main() { vUV = aCorner; gl_Position = uViewProj * vec4(uOrigin + uU * aCorner.x + uV * aCorner.y, 1.0); }`;
const PLANE_FS = `${HEADER}
in vec2 vUV;
uniform sampler2D uValues; uniform sampler2D uLut; uniform ivec2 uSize;
uniform vec4 uEmpty; uniform float uAlpha; uniform float uBorder; uniform bool uGrid;
uniform vec3 uInk; uniform vec3 uBg; uniform vec3 uHover; uniform vec3 uSelect;
uniform vec4 uHi; uniform vec4 uSel;
out vec4 outColor;
float ring(vec2 g, vec2 fw, vec4 r) {
  if (r.x < 0.0 || g.x < r.x || g.x > r.z || g.y < r.y || g.y > r.w) return 0.0;
  vec2 d = min(g - r.xy, r.zw - g) / max(fw, vec2(1e-7));
  return 1.0 - smoothstep(1.5, 2.5, min(d.x, d.y));
}
void main() {
  vec2 g = vUV * vec2(uSize);
  ivec2 cell = clamp(ivec2(floor(g)), ivec2(0), uSize - 1);
  int code = int(texelFetch(uValues, cell, 0).r * 255.0 + 0.5);
  vec4 c = code == 0 ? uEmpty : vec4(texelFetch(uLut, ivec2(code, 0), 0).rgb, 1.0);
  vec2 fw = fwidth(g);
  if (uGrid) {
    vec2 px = min(fract(g), 1.0 - fract(g)) / max(fw, vec2(1e-7));
    float cellPx = 1.0 / max(fw.x, fw.y);
    float line = (1.0 - smoothstep(0.25, 0.9, min(px.x, px.y))) * smoothstep(7.0, 12.0, cellPx);
    c.rgb = mix(c.rgb, uBg, 0.7 * line);
  }
  vec2 edge = min(vUV, 1.0 - vUV) / max(fwidth(vUV), vec2(1e-7));
  float border = uBorder > 0.0 ? 1.0 - smoothstep(uBorder - 0.5, uBorder + 0.5, min(edge.x, edge.y)) : 0.0;
  c = mix(c, vec4(uInk, 1.0), border);
  c = mix(c, vec4(uHover, 1.0), ring(g, fw, uHi));
  c = mix(c, vec4(uSelect, 1.0), ring(g, fw, uSel));
  float a = c.a * uAlpha;
  if (a < 0.002) discard;
  outColor = vec4(c.rgb * a, a);
}`;

/* ------------------------------------------------------------------ WebGL helpers */
function program(gl, vsSource, fsSource) {
  const shader = (type, source) => {
    const s = gl.createShader(type);
    gl.shaderSource(s, source);
    gl.compileShader(s);
    if (!gl.getShaderParameter(s, gl.COMPILE_STATUS) && !gl.isContextLost()) throw new Error(`shader: ${gl.getShaderInfoLog(s)}`);
    return s;
  };
  const p = gl.createProgram();
  gl.attachShader(p, shader(gl.VERTEX_SHADER, vsSource));
  gl.attachShader(p, shader(gl.FRAGMENT_SHADER, fsSource));
  gl.linkProgram(p);
  if (!gl.getProgramParameter(p, gl.LINK_STATUS) && !gl.isContextLost()) throw new Error(`program: ${gl.getProgramInfoLog(p)}`);
  const uniforms = {};
  const count = gl.getProgramParameter(p, gl.ACTIVE_UNIFORMS) || 0;
  for (let i = 0; i < count; i++) {
    const { name } = gl.getActiveUniform(p, i);
    uniforms[name] = gl.getUniformLocation(p, name);
  }
  return { p, u: uniforms };
}
function staticBuffer(gl, target, data) {
  const buffer = gl.createBuffer();
  gl.bindBuffer(target, buffer);
  gl.bufferData(target, data, gl.STATIC_DRAW);
  return buffer;
}
/* A mesh's vertex attributes at locations 0 and 1 (position, normal) and its index buffer. */
function meshBuffers(gl, geometry) {
  return {
    position: staticBuffer(gl, gl.ARRAY_BUFFER, new Float32Array(geometry.position)),
    normal: staticBuffer(gl, gl.ARRAY_BUFFER, new Float32Array(geometry.normal)),
    index: staticBuffer(gl, gl.ELEMENT_ARRAY_BUFFER, new Uint16Array(geometry.index)),
    count: geometry.index.length,
  };
}
/* Bind interleaved float instance attributes: layout [[location, size], ...] from `start`. */
function instanceAttributes(gl, buffer, layout) {
  const stride = layout.reduce((sum, [, size]) => sum + size, 0) * 4;
  gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
  let offset = 0;
  for (const [location, size] of layout) {
    gl.enableVertexAttribArray(location);
    gl.vertexAttribPointer(location, size, gl.FLOAT, false, stride, offset);
    gl.vertexAttribDivisor(location, 1);
    offset += size * 4;
  }
}
/* The buffers and vertex array of an instanced draw: the mesh's attributes (or `quad`, a 2D corner
   buffer at location 0), float instance attributes `layout` and a per-instance state float. */
function instanced(scene, mesh, layout, stateLocation) {
  const gl = scene.gl;
  const out = { instances: gl.createBuffer(), state: gl.createBuffer(), vao: gl.createVertexArray() };
  gl.bindVertexArray(out.vao);
  if (mesh instanceof WebGLBuffer) {
    gl.bindBuffer(gl.ARRAY_BUFFER, mesh);
    gl.enableVertexAttribArray(0);
    gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);
  } else scene.bindMesh(mesh);
  instanceAttributes(gl, out.instances, layout);
  instanceAttributes(gl, out.state, [[stateLocation, 1]]);
  gl.bindVertexArray(null);
  return out;
}
/* Upload an owner's instance data and states if they changed (flags `dirty`, `stateDirty`). */
function sync(gl, owner, data, state) {
  for (const [flag, buffer, array] of [['dirty', owner.gl.instances, data], ['stateDirty', owner.gl.state, state]]) {
    if (!owner[flag]) continue;
    gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
    gl.bufferData(gl.ARRAY_BUFFER, array, gl.DYNAMIC_DRAW);
    owner[flag] = false;
  }
}

/* ------------------------------------------------------------------ colour specs */
/* Per-instance RGBA (Float32Array, 4 per instance) from one spec, one spec per instance, or numbers;
   with `tone`, a role that has that tone ('pair' and 'face') resolves to it. */
function resolveColors(scene, spec, count, alphas = 1, tone = null) {
  const out = new Float32Array(4 * count);
  const cache = new Map();
  const one = (s) => {
    const key = Array.isArray(s) ? s.join() : s;
    if (!cache.has(key)) cache.set(key, scene.color(tone ? toned(s, tone) : s));
    return cache.get(key);
  };
  const numeric = ArrayBuffer.isView(spec) && !(spec instanceof DataView);
  const perGroup = Array.isArray(spec) && typeof spec[0] !== 'number';
  const k = numeric ? (spec.length === 4 * count ? 4 : 3) : 0;
  const unit = spec instanceof Uint8Array || spec instanceof Uint8ClampedArray ? 1 / 255 : 1;
  for (let i = 0; i < count; i++) {
    let c;
    if (numeric) c = [spec[i * k] * unit, spec[i * k + 1] * unit, spec[i * k + 2] * unit, k > 3 ? spec[i * k + 3] * unit : 1];
    else c = one(perGroup ? spec[i] : spec);
    const alpha = typeof alphas === 'number' ? alphas : alphas[i] ?? 1;
    out.set([c[0], c[1], c[2], c[3] * alpha], 4 * i);
  }
  return out;
}
const flatPoints = (points) => (ArrayBuffer.isView(points) ? points : Float32Array.from(points.flat ? points.flat() : points));

/* ------------------------------------------------------------------ instanced sets */
/* What boxes, meshes and lines share: per-item hover, selection and dimming, flattened to one
   state per instance (1 hover, 2 selected, -1 dimmed, 0 normal) for the shaders. */
class InstancedSet {
  constructor(scene, name, kind, options) {
    this.scene = scene;
    this.name = name;
    this.kind = kind;
    this.options = options;
    this.visible = options.visible ?? true;
    this.pickable = options.pickable ?? true;
    this.ids = options.ids || null;
    this.hovered = null;
    this.selected = new Set();
    this.dimmed = new Set();
    this.gl = null;
  }

  /** Change any option given at creation (arrays are replaced, not merged). */
  update(partial = {}) {
    const recolor = 'colors' in partial || 'alphas' in partial;
    this.options = { ...this.options, ...partial };
    if ('ids' in partial) this.ids = partial.ids;
    if ('visible' in partial) this.visible = partial.visible;
    if ('pickable' in partial) this.pickable = partial.pickable;
    const count = this.count;
    this.build(recolor);
    if (this.count !== count) this.clearStates();
    this.dirty = true;
    this.scene.invalidate(true);
    return this;
  }

  clearStates() {
    this.hovered = null;
    this.selected.clear();
    this.dimmed.clear();
    this.refreshState();
  }

  setHovered(index) {
    if (index === this.hovered) return;
    this.hovered = index;
    this.refreshState();
  }

  /** Mark items as selected (an index, a list of indices, or null to clear). */
  setSelected(indices) {
    this.selected = new Set(indices === null || indices === undefined ? [] : [indices].flat());
    this.refreshState();
  }

  /** Dim items (an index, a list of indices, or null for none), to set others off. */
  setDimmed(indices) {
    this.dimmed = new Set(indices === null || indices === undefined ? [] : [indices].flat());
    this.refreshState();
  }

  refreshState() {
    if (!this.itemOf) return;
    const n = this.itemOf.length;
    if (!this.state || this.state.length !== n) this.state = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      const item = this.itemOf[i];
      this.state[i] = item === this.hovered ? 1 : this.selected.has(item) ? 2 : this.dimmed.has(item) ? -1 : 0;
    }
    this.stateDirty = true;
    this.scene.invalidate();
  }

  restyle() {
    this.build(true);
    this.dirty = true;
  }

  dispose(gl) {
    if (!this.gl || !gl) return;
    for (const value of Object.values(this.gl)) {
      if (value instanceof WebGLBuffer) gl.deleteBuffer(value);
      else if (value instanceof WebGLVertexArrayObject) gl.deleteVertexArray(value);
    }
    this.gl = null;
  }
}

class BoxSet extends InstancedSet {
  constructor(scene, name, options) {
    super(scene, name, 'boxes', options);
    this.build(true);
  }

  build(recolor) {
    const o = this.options;
    const positions = flatPoints(o.positions || []);
    const n = positions.length / 3;
    const sizes = o.sizes === undefined ? [1, 1, 1] : flatPoints(o.sizes);
    this.count = n;
    this.itemOf = Array.from({ length: n }, (_, i) => i);
    this.data = new Float32Array(10 * n);
    if (recolor || !this.rgba || this.rgba.length !== 4 * n) this.rgba = resolveColors(this.scene, o.colors ?? 'ink-faint', n, o.alphas ?? 1, 'face');
    this.box = emptyBounds();
    for (let i = 0; i < n; i++) {
      const s = sizes.length === 3 ? sizes : sizes.subarray ? sizes.subarray(3 * i, 3 * i + 3) : sizes.slice(3 * i, 3 * i + 3);
      this.data.set([positions[3 * i], positions[3 * i + 1], positions[3 * i + 2], s[0], s[1], s[2]], 10 * i);
      this.data.set(this.rgba.subarray(4 * i, 4 * i + 4), 10 * i + 6);
      grow(grow(this.box, [positions[3 * i] - s[0] / 2, positions[3 * i + 1] - s[1] / 2, positions[3 * i + 2] - s[2] / 2]), [
        positions[3 * i] + s[0] / 2, positions[3 * i + 1] + s[1] / 2, positions[3 * i + 2] + s[2] / 2,
      ]);
    }
    this.glass = Array.from({ length: n }, (_, i) => this.rgba[4 * i + 3] < 0.999).some(Boolean);
    const edges = o.edges === false ? { width: 0 } : { width: 1, mix: SHADING.edgeMix, alpha: 0.9, ...(o.edges || {}) };
    this.edges = edges;
    this.refreshState();
  }

  /** The box of item `i`: {center, size}. */
  item(i) {
    const d = this.data.subarray(10 * i, 10 * i + 6);
    return { center: [d[0], d[1], d[2]], size: [d[3], d[4], d[5]] };
  }

  bounds(indices = null) {
    if (indices === null) return this.box;
    const b = emptyBounds();
    for (const i of [indices].flat()) {
      const { center, size } = this.item(i);
      grow(grow(b, sub(center, scale(size, 0.5))), add(center, scale(size, 0.5)));
    }
    return b;
  }

  upload(gl, eye) {
    if (!this.gl) {
      this.gl = instanced(this.scene, this.scene.meshes.cube, [[2, 3], [3, 3], [4, 4]], 5);
      this.dirty = this.stateDirty = true;
    }
    if (this.glass && eye) {
      /* Glass is drawn back to front: sort by distance from the eye and upload in that order. */
      const n = this.count;
      const order = Array.from({ length: n }, (_, i) => i);
      const d = new Float32Array(n);
      for (let i = 0; i < n; i++) {
        const c = this.data.subarray(10 * i, 10 * i + 3);
        d[i] = (c[0] - eye[0]) ** 2 + (c[1] - eye[1]) ** 2 + (c[2] - eye[2]) ** 2;
      }
      order.sort((a, b) => d[b] - d[a]);
      const data = new Float32Array(10 * n), state = new Float32Array(n);
      order.forEach((i, k) => {
        data.set(this.data.subarray(10 * i, 10 * i + 10), 10 * k);
        state[k] = this.state[i];
      });
      this.dirty = this.stateDirty = true;
      sync(gl, this, data, state);
      return;
    }
    sync(gl, this, this.data, this.state);
  }

  draw(gl, pass) {
    if (!this.count || pass !== (this.glass ? 'glass' : 'opaque')) return;
    const { programs, meshes, dpr } = this.scene;
    const prog = programs.box;
    gl.useProgram(prog.p);
    this.scene.commonUniforms(prog);
    gl.uniform1f(prog.u.uEdgeWidth, this.edges.width * dpr);
    gl.uniform1f(prog.u.uEdgeMix, this.edges.mix ?? SHADING.edgeMix);
    gl.uniform1f(prog.u.uEdgeAlpha, this.edges.alpha ?? 0.9);
    gl.bindVertexArray(this.gl.vao);
    if (this.glass) {
      gl.cullFace(gl.FRONT);
      gl.drawElementsInstanced(gl.TRIANGLES, meshes.cube.count, gl.UNSIGNED_SHORT, 0, this.count);
      gl.cullFace(gl.BACK);
    }
    gl.drawElementsInstanced(gl.TRIANGLES, meshes.cube.count, gl.UNSIGNED_SHORT, 0, this.count);
    gl.bindVertexArray(null);
  }

  pick(origin, dir) {
    let best = null;
    for (let i = 0; i < this.count; i++) {
      if (this.rgba[4 * i + 3] <= 0.01) continue;
      const d = this.data.subarray(10 * i, 10 * i + 6);
      let t0 = -Infinity, t1 = Infinity;
      for (let k = 0; k < 3; k++) {
        const lo = d[k] - d[k + 3] / 2, hi = d[k] + d[k + 3] / 2;
        if (Math.abs(dir[k]) < 1e-12) {
          if (origin[k] < lo || origin[k] > hi) t0 = Infinity;
          continue;
        }
        let a = (lo - origin[k]) / dir[k], b = (hi - origin[k]) / dir[k];
        if (a > b) [a, b] = [b, a];
        t0 = Math.max(t0, a);
        t1 = Math.min(t1, b);
      }
      if (t0 <= t1 && t1 > 0 && (!best || t0 < best.t)) best = { index: i, t: Math.max(0, t0) };
    }
    return best;
  }
}

/* Spheres, and tubes built from cylinders, joint spheres and arrowhead cones. Items are spheres
   or whole paths; `groups` hold the instances of one geometry each. */
class MeshSet extends InstancedSet {
  constructor(scene, name, kind, options) {
    super(scene, name, kind, options);
    this.build(true);
  }

  /* Spheres are solids and take a role's face tone; tubes are lines, so on the dark stage they keep the base,
     which the deep face tones would sink into. */
  tone() {
    return this.kind === 'tubes' && currentTheme() === 'dark' ? null : 'face';
  }

  build(recolor) {
    const o = this.options;
    const groups = { sphere: [], cylinder: [], cone: [] };
    const push = (geometry, a, b, r, item, segment = -1) => groups[geometry].push({ a, b, r, item, segment });
    let items = 0;
    if (this.kind === 'spheres') {
      const p = flatPoints(o.positions || []);
      items = p.length / 3;
      for (let i = 0; i < items; i++) {
        const r = o.radii ? o.radii[i] : o.radius ?? 0.1;
        push('sphere', [p[3 * i], p[3 * i + 1], p[3 * i + 2]], [0, 0, 0], r, i);
      }
    } else {
      const paths = (o.paths || []).map(flatPoints);
      items = paths.length;
      paths.forEach((path, item) => {
        const r = o.radii ? o.radii[item] : o.radius ?? 0.05;
        const n = path.length / 3;
        const at = (k) => [path[3 * k], path[3 * k + 1], path[3 * k + 2]];
        const arrow = o.arrow ? { length: 5 * r, radius: 2.4 * r, ...(o.arrow === true ? {} : o.arrow) } : null;
        for (let k = 0; k + 1 < n; k++) {
          let a = at(k), b = at(k + 1);
          if (arrow && k === n - 2) {
            const dir = normalize(sub(b, a));
            const shaft = Math.max(0, length(sub(b, a)) - arrow.length);
            const base = add(a, scale(dir, shaft));
            push('cone', base, b, arrow.radius, item, k);
            b = base;
          }
          push('cylinder', a, b, r, item, k);
        }
        if (o.joints !== false) for (let k = 0; k < n - (arrow ? 1 : 0); k++) push('sphere', at(k), [0, 0, 0], r, item, k);
      });
    }
    this.count = items;
    if (recolor || !this.itemRgba || this.itemRgba.length !== 4 * items) this.itemRgba = resolveColors(this.scene, o.colors ?? 'ink-soft', items, o.alphas ?? 1, this.tone());
    this.box = emptyBounds();
    this.groups = Object.entries(groups)
      .filter(([, list]) => list.length)
      .map(([geometry, list]) => {
        const data = new Float32Array(11 * list.length);
        list.forEach(({ a, b, r, item }, i) => {
          data.set([...a, ...b, r], 11 * i);
          data.set(this.itemRgba.subarray(4 * item, 4 * item + 4), 11 * i + 7);
          grow(this.box, a, r);
          if (geometry !== 'sphere') grow(this.box, b, r);
        });
        return { geometry, list, data, state: new Float32Array(list.length), gl: null, dirty: true };
      });
    this.glass = Array.from({ length: items }, (_, i) => this.itemRgba[4 * i + 3] < 0.999).some(Boolean);
    this.itemOf = this.groups.flatMap((g) => g.list.map((entry) => entry.item));
    this.refreshState();
  }

  refreshState() {
    super.refreshState();
    if (!this.groups) return;
    let offset = 0;
    for (const group of this.groups) {
      group.state.set(this.state.subarray(offset, offset + group.list.length));
      offset += group.list.length;
      group.stateDirty = true;
    }
  }

  bounds() {
    return this.box;
  }

  upload(gl) {
    for (const group of this.groups) {
      if (!group.gl) {
        group.gl = instanced(this.scene, this.scene.meshes[group.geometry], [[2, 3], [3, 3], [4, 1], [5, 4]], 6);
        group.dirty = group.stateDirty = true;
      }
      group.dirty ||= this.dirty;
      sync(gl, group, group.data, group.state);
    }
    this.dirty = false;
  }

  draw(gl, pass) {
    if (pass !== (this.glass ? 'glass' : 'opaque')) return;
    const { programs, meshes } = this.scene;
    const prog = programs.mesh;
    gl.useProgram(prog.p);
    this.scene.commonUniforms(prog);
    for (const group of this.groups) {
      gl.uniform1i(prog.u.uKind, { sphere: 0, cylinder: 1, cone: 2 }[group.geometry]);
      gl.bindVertexArray(group.gl.vao);
      gl.drawElementsInstanced(gl.TRIANGLES, meshes[group.geometry].count, gl.UNSIGNED_SHORT, 0, group.list.length);
    }
    gl.bindVertexArray(null);
  }

  pick(origin, dir) {
    let best = null;
    for (const group of this.groups) {
      for (const { a, b, r, item, segment } of group.list) {
        if (this.itemRgba[4 * item + 3] <= 0.01) continue;
        const t = group.geometry === 'sphere' ? raySphere(origin, dir, a, r) : raySegment(origin, dir, a, b, group.geometry === 'cone' ? r * 0.6 : r);
        if (t !== null && (!best || t < best.t)) best = { index: item, t, segment };
      }
    }
    return best;
  }

  dispose(gl) {
    for (const group of this.groups || []) {
      if (!group.gl || !gl) continue;
      gl.deleteBuffer(group.gl.instances);
      gl.deleteBuffer(group.gl.state);
      gl.deleteVertexArray(group.gl.vao);
      group.gl = null;
    }
  }
}

function raySphere(o, d, c, r) {
  const oc = sub(o, c);
  const b = dot(oc, d), cc = dot(oc, oc) - r * r;
  const h = b * b - cc;
  if (h < 0) return null;
  const t = -b - Math.sqrt(h);
  return t > 0 ? t : null;
}
/* The ray parameter where the ray passes within `r` of segment ab (a capsule test), or null. */
function raySegment(o, d, a, b, r) {
  const u = sub(b, a), w = sub(o, a);
  const uu = dot(u, u), ud = dot(u, d), uw = dot(u, w), dw = dot(d, w);
  const denom = uu - ud * ud;
  let s = denom > 1e-12 ? (uw - ud * dw) / denom : 0;
  s = Math.min(1, Math.max(0, s));
  const p = add(a, scale(u, s));
  const t = dot(sub(p, o), d);
  if (t <= 0) return null;
  const q = add(o, scale(d, t));
  return length(sub(q, p)) <= r ? t : null;
}

class LineSet extends InstancedSet {
  constructor(scene, name, options) {
    super(scene, name, 'lines', { pickable: false, ...options });
    this.build(true);
  }

  build(recolor) {
    const o = this.options;
    const segments = [];
    if (o.segments) {
      const s = flatPoints(o.segments);
      for (let i = 0; i < s.length / 6; i++) segments.push({ a: [...s.subarray(6 * i, 6 * i + 3)], b: [...s.subarray(6 * i + 3, 6 * i + 6)], item: i, segment: i });
    } else {
      (o.paths || []).map(flatPoints).forEach((path, item) => {
        for (let k = 0; k + 1 < path.length / 3; k++) {
          segments.push({ a: [...path.subarray(3 * k, 3 * k + 3)], b: [...path.subarray(3 * k + 3, 3 * k + 6)], item, segment: k });
        }
      });
    }
    const items = o.segments ? segments.length : (o.paths || []).length;
    this.count = items;
    this.segments = segments;
    if (recolor || !this.itemRgba || this.itemRgba.length !== 4 * items) this.itemRgba = resolveColors(this.scene, o.colors ?? 'ink-soft', items, o.alphas ?? 1);
    this.data = new Float32Array(10 * segments.length);
    this.box = emptyBounds();
    segments.forEach(({ a, b, item }, i) => {
      this.data.set([...a, ...b], 10 * i);
      this.data.set(this.itemRgba.subarray(4 * item, 4 * item + 4), 10 * i + 6);
      grow(grow(this.box, a), b);
    });
    this.itemOf = segments.map((s) => s.item);
    this.refreshState();
  }

  bounds() {
    return this.box;
  }

  upload(gl) {
    if (!this.gl) {
      this.gl = instanced(this.scene, this.scene.meshes.quad, [[1, 3], [2, 3], [3, 4]], 4);
      this.dirty = this.stateDirty = true;
    }
    sync(gl, this, this.data, this.state);
  }

  draw(gl, pass) {
    if (!this.segments.length || pass !== (this.options.onTop ? 'top' : 'lines')) return;
    const { programs, dpr } = this.scene;
    const prog = programs.line;
    gl.useProgram(prog.p);
    this.scene.commonUniforms(prog);
    gl.uniform1f(prog.u.uWidth, (this.options.width ?? 1.5) * dpr);
    const dash = this.options.dash || [0, 0];
    gl.uniform2f(prog.u.uDash, dash[0] * dpr, dash[1] * dpr);
    gl.bindVertexArray(this.gl.vao);
    gl.drawArraysInstanced(gl.TRIANGLE_STRIP, 0, 4, this.segments.length);
    gl.bindVertexArray(null);
  }

  pick(origin, dir, screen) {
    let best = null;
    const tolerance = LINE_PICK_PX + (this.options.width ?? 1.5) / 2;
    for (const { a, b, item, segment } of this.segments) {
      const pa = this.scene.project(a), pb = this.scene.project(b);
      if (!pa.visible && !pb.visible) continue;
      const v = [pb.x - pa.x, pb.y - pa.y];
      const l2 = v[0] * v[0] + v[1] * v[1];
      const s = l2 ? Math.min(1, Math.max(0, ((screen.x - pa.x) * v[0] + (screen.y - pa.y) * v[1]) / l2)) : 0;
      const dist = Math.hypot(pa.x + s * v[0] - screen.x, pa.y + s * v[1] - screen.y);
      if (dist > tolerance) continue;
      const t = dot(sub(add(a, scale(sub(b, a), s)), origin), dir);
      if (t > 0 && (!best || t < best.t)) best = { index: item, t, segment };
    }
    return best;
  }
}

/* One rectangle coloured by an array of values through a ramp (see the header). */
class PlaneSet {
  constructor(scene, name, options) {
    this.scene = scene;
    this.name = name;
    this.kind = 'plane';
    this.hi = [-1, -1, -1, -1];
    this.sel = [-1, -1, -1, -1];
    this.options = {};
    this.gl = null;
    this.update(options);
  }

  update(partial = {}) {
    const o = (this.options = { ...this.options, ...partial });
    this.visible = o.visible ?? true;
    this.pickable = o.pickable ?? true;
    if (o.box && o.face) Object.assign(o, faceFrame(o.box, o.face), { box: undefined, face: undefined });
    this.origin = o.origin;
    this.u = o.u;
    this.v = o.v;
    if ('values' in partial || 'domain' in partial || 'width' in partial || this.codes === undefined) this.encode();
    if ('colormap' in partial || 'categories' in partial || 'empty' in partial || this.lut === undefined) this.restyle();
    this.box = grow(grow(grow(grow(emptyBounds(), this.origin), add(this.origin, this.u)), add(this.origin, this.v)), add(add(this.origin, this.u), this.v));
    this.scene.invalidate(true);
    return this;
  }

  /* Map values to codes 1..255 (0 = empty), or keep integer category codes as they are. */
  encode() {
    const o = this.options;
    const values = o.values || new Float32Array((o.width || 1) * (o.height || 1));
    const [w, h] = [o.width || values.length, o.height || 1];
    if (w * h !== values.length) throw new Error(`plane "${this.name}": ${values.length} values for ${w} x ${h}`);
    this.width = w;
    this.height = h;
    this.values = values;
    const codes = new Uint8Array(w * h);
    if (o.categories) {
      for (let i = 0; i < codes.length; i++) codes[i] = Number.isFinite(values[i]) ? Math.max(0, Math.min(255, Math.round(values[i]))) : 0;
      this.domain = [1, o.categories.length];
    } else {
      let [lo, hi] = o.domain || [Infinity, -Infinity];
      if (!o.domain) {
        for (const v of values) if (Number.isFinite(v)) [lo, hi] = [Math.min(lo, v), Math.max(hi, v)];
        if (o.colormap === 'diverging') [lo, hi] = [-Math.max(Math.abs(lo), Math.abs(hi)), Math.max(Math.abs(lo), Math.abs(hi))];
      }
      if (!(hi > lo)) [lo, hi] = [lo - 0.5, lo + 0.5];
      this.domain = [lo, hi];
      for (let i = 0; i < codes.length; i++) {
        const v = values[i];
        codes[i] = Number.isFinite(v) ? 1 + Math.round(254 * Math.min(1, Math.max(0, (v - lo) / (hi - lo)))) : 0;
      }
    }
    this.codes = codes;
    this.dirtyValues = true;
  }

  restyle() {
    const o = this.options;
    const { element, mode } = this.scene.colorContext();
    const lut = new Uint8Array(4 * 256);
    if (o.categories) {
      o.categories.forEach((spec, k) => {
        const c = this.scene.color(spec);
        lut.set(c.map((x) => Math.round(x * 255)), 4 * (k + 1));
      });
    } else {
      const ramp = rampTable(o.colormap || 'sequential', { element, mode, n: 255 });
      lut.set(ramp, 4);
    }
    this.lut = lut;
    this.empty = o.empty === null ? [0, 0, 0, 0] : this.scene.color(o.empty ?? 'surface-alt');
    this.dirtyLut = true;
  }

  /** The value at a cell (NaN outside). */
  valueAt(col, row) {
    if (col < 0 || row < 0 || col >= this.width || row >= this.height) return NaN;
    return this.values[row * this.width + col];
  }

  /**
   * Outline a block of cells: {col, row} for one cell, or {col0, row0, col1, row1} (inclusive);
   * a missing col or row spans the whole axis. `kind` 'hover' uses the hover colour, 'select'
   * the selection colour; null clears.
   */
  setHighlight(rect, kind = 'select') {
    let out = [-1, -1, -1, -1];
    if (rect) {
      const col0 = rect.col0 ?? rect.col ?? 0, col1 = rect.col1 ?? rect.col ?? this.width - 1;
      const row0 = rect.row0 ?? rect.row ?? 0, row1 = rect.row1 ?? rect.row ?? this.height - 1;
      out = [col0, row0, col1 + 1, row1 + 1];
    }
    if (kind === 'hover') this.hi = out;
    else this.sel = out;
    this.scene.invalidate();
  }

  bounds() {
    return this.box;
  }

  upload(gl) {
    if (!this.gl) {
      this.gl = { values: gl.createTexture(), lut: gl.createTexture(), vao: gl.createVertexArray() };
      gl.bindVertexArray(this.gl.vao);
      gl.bindBuffer(gl.ARRAY_BUFFER, this.scene.meshes.unitQuad);
      gl.enableVertexAttribArray(0);
      gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);
      gl.bindVertexArray(null);
      this.dirtyValues = this.dirtyLut = true;
    }
    const texture = (tex, w, h, format, internal, data) => {
      gl.bindTexture(gl.TEXTURE_2D, tex);
      gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
      gl.texImage2D(gl.TEXTURE_2D, 0, internal, w, h, 0, format, gl.UNSIGNED_BYTE, data);
      for (const p of [gl.TEXTURE_MIN_FILTER, gl.TEXTURE_MAG_FILTER]) gl.texParameteri(gl.TEXTURE_2D, p, gl.NEAREST);
      for (const p of [gl.TEXTURE_WRAP_S, gl.TEXTURE_WRAP_T]) gl.texParameteri(gl.TEXTURE_2D, p, gl.CLAMP_TO_EDGE);
    };
    if (this.dirtyValues) {
      const max = gl.getParameter(gl.MAX_TEXTURE_SIZE);
      if (this.width > max || this.height > max) throw new Error(`plane "${this.name}" is ${this.width} x ${this.height}, over MAX_TEXTURE_SIZE ${max}`);
      texture(this.gl.values, this.width, this.height, gl.RED, gl.R8, this.codes);
      this.dirtyValues = false;
    }
    if (this.dirtyLut) {
      texture(this.gl.lut, 256, 1, gl.RGBA, gl.RGBA8, this.lut);
      this.dirtyLut = false;
    }
  }

  draw(gl, pass) {
    const alpha = this.options.alpha ?? 1;
    if (pass !== (alpha < 0.999 ? 'glass' : 'opaque')) return;
    const prog = this.scene.programs.plane;
    gl.useProgram(prog.p);
    this.scene.commonUniforms(prog);
    gl.uniform3fv(prog.u.uOrigin, this.origin);
    gl.uniform3fv(prog.u.uU, this.u);
    gl.uniform3fv(prog.u.uV, this.v);
    gl.uniform2i(prog.u.uSize, this.width, this.height);
    gl.uniform4fv(prog.u.uEmpty, this.empty);
    gl.uniform1f(prog.u.uAlpha, alpha);
    gl.uniform1f(prog.u.uBorder, (this.options.border ?? 1) * this.scene.dpr);
    gl.uniform1i(prog.u.uGrid, this.options.grid === false ? 0 : 1);
    gl.uniform4fv(prog.u.uHi, this.hi);
    gl.uniform4fv(prog.u.uSel, this.sel);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.gl.values);
    gl.uniform1i(prog.u.uValues, 0);
    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, this.gl.lut);
    gl.uniform1i(prog.u.uLut, 1);
    gl.bindVertexArray(this.gl.vao);
    gl.enable(gl.POLYGON_OFFSET_FILL);
    gl.polygonOffset(-1, -2);
    gl.disable(gl.CULL_FACE);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
    gl.enable(gl.CULL_FACE);
    gl.disable(gl.POLYGON_OFFSET_FILL);
    gl.bindVertexArray(null);
  }

  pick(origin, dir) {
    const n = cross(this.u, this.v);
    const denom = dot(dir, n);
    if (Math.abs(denom) < 1e-12) return null;
    const t = dot(sub(this.origin, origin), n) / denom;
    if (t <= 0) return null;
    const rel = sub(add(origin, scale(dir, t)), this.origin);
    const a = dot(rel, this.u) / dot(this.u, this.u), b = dot(rel, this.v) / dot(this.v, this.v);
    if (a < 0 || a >= 1 || b < 0 || b >= 1) return null;
    const col = Math.floor(a * this.width), row = Math.floor(b * this.height);
    return { index: row * this.width + col, t, col, row, value: this.valueAt(col, row) };
  }

  dispose(gl) {
    if (!this.gl || !gl) return;
    gl.deleteTexture(this.gl.values);
    gl.deleteTexture(this.gl.lut);
    gl.deleteVertexArray(this.gl.vao);
    this.gl = null;
  }
}

const shortestAngle = (a, b) => a + ((((b - a) % 360) + 540) % 360) - 180;
function interpolateCamera(a, b, f) {
  const lerp = (x, y) => x + (y - x) * f;
  return {
    target: [0, 1, 2].map((k) => lerp(a.target[k], b.target[k])),
    distance: Math.exp(lerp(Math.log(a.distance), Math.log(b.distance))),
    yaw: lerp(a.yaw, shortestAngle(a.yaw, b.yaw)),
    pitch: lerp(a.pitch, b.pitch),
  };
}

/* ------------------------------------------------------------------ the scene */
export class Scene3D extends EventTarget {
  /**
   * @param {HTMLElement} host  element that receives the canvas and the label overlay; give it a size in CSS
   * @param {object} [options]
   * @param {string} [options.label]           accessible name of the canvas
   * @param {object} [options.camera]          home camera {target, distance, yaw, pitch}; default: fit the content
   * @param {'perspective'|'orthographic'} [options.projection='perspective']
   * @param {number} [options.fov=35]          vertical field of view in degrees
   * @param {number[]} [options.zoomRange]     [min, max] camera distance
   * @param {'ctrl'|'always'} [options.wheel='ctrl']  whether a bare wheel zooms
   * @param {string} [options.touchAction='pan-y']    CSS touch-action of the canvas ('none' orbits freely)
   * @param {string} [options.background='surface']  colour spec of the background
   * @param {boolean} [options.autoHover=true], [options.autoSelect=true], [options.declutter=true]
   * @param {number} [options.maxLabels]      labels shown at once (default 8, or 5 on a canvas under 600 px)
   * @param {number} [options.ambient=0.72]    share of face shading that does not depend on the light
   */
  constructor(host, options = {}) {
    super();
    this.host = host;
    this.options = { projection: 'perspective', fov: 35, wheel: 'ctrl', touchAction: 'pan-y', background: 'surface', autoHover: true, autoSelect: true, declutter: true, ambient: SHADING.ambient, ...options };
    this.canvas = document.createElement('canvas');
    this.canvas.className = 'viz-scene-canvas';
    this.canvas.tabIndex = 0;
    this.canvas.setAttribute('role', 'img');
    this.setLabel(this.options.label || '3D view');
    this.canvas.style.touchAction = this.options.touchAction;
    const gl = this.canvas.getContext('webgl2', { antialias: true, alpha: false, premultipliedAlpha: true, depth: true, stencil: false, preserveDrawingBuffer: false });
    if (!gl) {
      const error = new Error('WebGL2 is not available in this browser');
      error.code = 'webgl2-unavailable';
      error.userMessage = 'This view needs WebGL2, which this browser does not provide.';
      throw error;
    }
    this.gl = gl;
    if (getComputedStyle(host).position === 'static') host.style.position = 'relative';
    this.overlay = document.createElement('div');
    this.overlay.className = 'viz-scene-labels';
    this.overlay.setAttribute('aria-hidden', 'true');
    this.leaders = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
    this.leaders.setAttribute('class', 'viz-scene-leaders');
    this.leaderPath = document.createElementNS('http://www.w3.org/2000/svg', 'path');
    this.leaders.append(this.leaderPath);
    this.overlay.append(this.leaders);
    this.message = document.createElement('p');
    this.message.className = 'viz-scene-message';
    this.message.hidden = true;
    host.append(this.canvas, this.overlay, this.message);
    this.sets = new Map();
    this.labels = new Set();
    this.animators = new Set();
    this.flights = new Set();
    this.dpr = 1;
    this.frame = 0;
    this.lost = false;
    this.cam = { target: [0, 0, 0], distance: 10, yaw: 30, pitch: 20 };
    this.homeCamera = options.camera ? { ...this.cam, ...options.camera } : null;
    if (this.homeCamera) this.cam = { ...this.homeCamera };
    this.autoFit = !options.camera;
    this.initGL();
    this.restyle();
    this.bind();
    this.handleContextLoss();
    this.resizer = new ResizeObserver(() => this.resize());
    this.resizer.observe(host);
    this.watchPixelRatio();
    this.unTheme = onTheme(() => this.restyle());
    this.resize();
  }

  setLabel(text) {
    this.canvas.setAttribute('aria-label', `${text}. Drag or use the arrow keys to rotate, plus and minus to zoom, 0 to reset.`);
  }

  initGL() {
    const gl = this.gl;
    this.programs = {
      box: program(gl, BOX_VS, BOX_FS),
      mesh: program(gl, MESH_VS, MESH_FS),
      line: program(gl, LINE_VS, LINE_FS),
      plane: program(gl, PLANE_VS, PLANE_FS),
    };
    this.meshes = {
      cube: meshBuffers(gl, cube()),
      cylinder: meshBuffers(gl, cylinder()),
      cone: meshBuffers(gl, cone()),
      sphere: meshBuffers(gl, sphere()),
      quad: staticBuffer(gl, gl.ARRAY_BUFFER, new Float32Array([0, -1, 1, -1, 0, 1, 1, 1])),
      unitQuad: staticBuffer(gl, gl.ARRAY_BUFFER, new Float32Array([0, 0, 1, 0, 0, 1, 1, 1])),
    };
    gl.enable(gl.DEPTH_TEST);
    gl.enable(gl.CULL_FACE);
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA);
  }

  bindMesh(mesh) {
    const gl = this.gl;
    gl.bindBuffer(gl.ARRAY_BUFFER, mesh.position);
    gl.enableVertexAttribArray(0);
    gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 0, 0);
    gl.bindBuffer(gl.ARRAY_BUFFER, mesh.normal);
    gl.enableVertexAttribArray(1);
    gl.vertexAttribPointer(1, 3, gl.FLOAT, false, 0, 0);
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, mesh.index);
  }

  /* ---------------------------------------------------------------- colours */
  colorContext() {
    return { element: this.host, mode: currentTheme() };
  }

  /** Resolve a colour spec on the host element to [r, g, b, a] in 0..1. */
  color(spec) {
    return resolveColor(spec, this.host, currentTheme());
  }

  /** Re-read every colour from CSS (the scene calls it on theme changes). */
  restyle() {
    const pick = (spec) => this.color(spec).slice(0, 3);
    this.palette = {
      background: pick(this.options.background),
      ink: pick('ink'),
      hover: pick('hover'),
      select: pick('select'),
    };
    for (const set of this.sets.values()) set.restyle();
    this.invalidate();
  }

  commonUniforms(prog) {
    const gl = this.gl;
    const u = prog.u;
    const p = this.palette;
    if (u.uViewProj) gl.uniformMatrix4fv(u.uViewProj, false, this.matrices.viewProj);
    if (u.uView) gl.uniformMatrix4fv(u.uView, false, this.matrices.view);
    if (u.uProj) gl.uniformMatrix4fv(u.uProj, false, this.matrices.proj);
    if (u.uViewport) gl.uniform2f(u.uViewport, this.canvas.width, this.canvas.height);
    if (u.uNear) gl.uniform1f(u.uNear, this.matrices.near);
    if (u.uOrtho) gl.uniform1i(u.uOrtho, this.options.projection === 'orthographic' ? 1 : 0);
    if (u.uLight) gl.uniform3fv(u.uLight, normalize(SHADING.light));
    if (u.uAmbient) gl.uniform1f(u.uAmbient, this.options.ambient);
    if (u.uInk) gl.uniform3fv(u.uInk, p.ink);
    if (u.uBg) gl.uniform3fv(u.uBg, p.background);
    if (u.uHover) gl.uniform3fv(u.uHover, p.hover);
    if (u.uSelect) gl.uniform3fv(u.uSelect, p.select);
    if (u.uDim) gl.uniform1f(u.uDim, 0.25);
  }

  /* ---------------------------------------------------------------- content */
  addSet(set) {
    this.remove(set.name);
    this.sets.set(set.name, set);
    this.invalidate(true);
    return set;
  }

  /** Instanced boxes; see the header. Returns the set, whose update() changes any option. */
  addBoxes(name, options) {
    return this.addSet(new BoxSet(this, name, options));
  }

  /** Spheres at `positions` with `radius` or `radii`. */
  addSpheres(name, options) {
    return this.addSet(new MeshSet(this, name, 'spheres', options));
  }

  /** Tubes along `paths` (arrays of xyz), with joints and an optional arrowhead. */
  addTubes(name, options) {
    return this.addSet(new MeshSet(this, name, 'tubes', options));
  }

  /** Screen-space lines along `paths` or between `segments` (6 numbers per segment). */
  addLines(name, options) {
    return this.addSet(new LineSet(this, name, options));
  }

  /** A textured rectangle; see the header and faceFrame. */
  addPlane(name, options) {
    return this.addSet(new PlaneSet(this, name, options));
  }

  get(name) {
    return this.sets.get(name) || null;
  }

  remove(name) {
    const set = this.sets.get(name);
    if (!set) return;
    set.dispose(this.lost ? null : this.gl);
    this.sets.delete(name);
    if (this.hoverHit?.set === name) this.hoverHit = null;
    this.invalidate(true);
  }

  clear() {
    for (const name of [...this.sets.keys()]) this.remove(name);
    for (const label of [...this.labels]) label.remove();
  }

  setVisible(name, visible) {
    const set = this.get(name);
    if (set) set.visible = visible;
    this.invalidate(true);
  }

  /** Bounds {min, max} of the visible content, of one set, or of some of its items. */
  bounds(name = null, indices = null) {
    if (name) return this.get(name).bounds(indices);
    const b = emptyBounds();
    for (const set of this.sets.values()) if (set.visible && boundsValid(set.bounds())) unionBounds(b, set.bounds());
    return boundsValid(b) ? b : { min: [-1, -1, -1], max: [1, 1, 1] };
  }

  /* ---------------------------------------------------------------- labels */
  /**
   * Pin an HTML label to a 3D point.
   * @param {object} spec  position [x, y, z]; text, html or node; className; anchor ('center', 'top',
   *   'bottom', 'left', 'right' or a corner such as 'top-left': the side of the label at the point);
   *   offset [dx, dy] in CSS px; priority (higher wins under declutter); maxDistance / minDistance
   *   (camera distances at which it shows); interactive (a <button> that takes pointer events and
   *   focus, with onClick); leader (a 1 px ink-faint line from the point to the label, 4 px short of it).
   * @returns {{element: HTMLElement, set: Function, remove: Function}}
   */
  addLabel(spec) {
    const label = { spec: { anchor: 'center', offset: [0, 0], priority: 0, visible: true, ...spec } };
    const el = document.createElement(spec.interactive ? 'button' : 'div');
    if (spec.interactive) {
      el.type = 'button';
      if (spec.onClick) el.addEventListener('click', spec.onClick);
      this.overlay.removeAttribute('aria-hidden');
    }
    label.element = el;
    label.set = (partial) => {
      Object.assign(label.spec, partial);
      const s = label.spec;
      el.className = `viz-scene-label${s.interactive ? ' viz-scene-label-button' : ''}${s.className ? ` ${s.className}` : ''}`;
      if ('text' in partial || 'html' in partial || 'node' in partial || !label.measured) {
        if (s.node) el.replaceChildren(s.node);
        else if (s.html !== undefined) el.innerHTML = s.html;
        else el.textContent = s.text ?? '';
        label.measured = null;
      }
      this.invalidate();
      return label;
    };
    label.remove = () => {
      el.remove();
      this.labels.delete(label);
    };
    label.set({});
    this.overlay.append(el);
    this.labels.add(label);
    return label;
  }

  clearLabels() {
    for (const label of [...this.labels]) label.remove();
  }

  placeLabels() {
    const w = this.canvas.clientWidth, h = this.canvas.clientHeight;
    const placed = [];
    const leaders = [];
    const cap = this.options.maxLabels ?? (w < 600 ? 5 : 8);
    const items = [...this.labels].sort((a, b) => b.spec.priority - a.spec.priority);
    for (const label of items) {
      const s = label.spec;
      const el = label.element;
      const p = this.project(s.position);
      let show = s.visible && p.visible && this.cam.distance <= (s.maxDistance ?? Infinity) && this.cam.distance >= (s.minDistance ?? 0);
      if (show && !label.measured) {
        el.style.display = '';
        label.measured = [el.offsetWidth, el.offsetHeight];
      }
      if (show) {
        const [lw, lh] = label.measured;
        const anchor = s.anchor;
        let x = p.x - lw / 2 + s.offset[0], y = p.y - lh / 2 + s.offset[1];
        if (anchor.includes('left')) x = p.x + s.offset[0];
        if (anchor.includes('right')) x = p.x - lw + s.offset[0];
        if (anchor.includes('top')) y = p.y + s.offset[1];
        if (anchor.includes('bottom')) y = p.y - lh + s.offset[1];
        /* A label whose point is on the canvas stays wholly on it. */
        if (p.x < 0 || p.x > w || p.y < 0 || p.y > h) show = false;
        x = Math.round(Math.min(Math.max(2, x), w - lw - 2));
        y = Math.round(Math.min(Math.max(2, y), h - lh - 2));
        const rect = [x, y, x + lw, y + lh];
        if (show && this.options.declutter && !s.interactive) {
          if (placed.some((r) => r[0] < rect[2] && rect[0] < r[2] && r[1] < rect[3] && rect[1] < r[3])) show = false;
          if (placed.length >= cap) show = false;
        }
        if (show) {
          placed.push(rect);
          el.style.transform = `translate(${x}px, ${y}px)`;
          if (s.leader) {
            const ex = Math.min(Math.max(p.x, x), x + lw), ey = Math.min(Math.max(p.y, y), y + lh);
            const d = Math.hypot(ex - p.x, ey - p.y);
            if (d > 8) leaders.push(`M${p.x.toFixed(1)} ${p.y.toFixed(1)}L${(ex - ((ex - p.x) * 4) / d).toFixed(1)} ${(ey - ((ey - p.y) * 4) / d).toFixed(1)}`);
          }
        }
      }
      el.style.display = show ? '' : 'none';
    }
    this.leaderPath.setAttribute('d', leaders.join(''));
  }

  /* ---------------------------------------------------------------- camera */
  /** The current camera: {target, distance, yaw, pitch} (a copy; angles in degrees). */
  get camera() {
    return { target: [...this.cam.target], distance: this.cam.distance, yaw: this.cam.yaw, pitch: this.cam.pitch };
  }

  fullCamera(partial = {}) {
    return { ...this.camera, ...partial, target: [...(partial.target || this.cam.target)] };
  }

  clampCamera(c) {
    const [lo, hi] = this.options.zoomRange || [1e-4, 1e7];
    c.distance = Math.min(hi, Math.max(lo, c.distance));
    c.pitch = Math.min(89.5, Math.max(-89.5, c.pitch));
    return c;
  }

  /** Jump to a camera (any subset of {target, distance, yaw, pitch}). */
  setCamera(partial) {
    this.cam = this.clampCamera(this.fullCamera(partial));
    this.autoFit = false;
    this.fitPending = false;
    this.matrices = null;
    this.viewChanged = true;
    this.invalidate();
  }

  /** The home camera (the constructor's `camera`, or the first fit). */
  setHome(camera) {
    this.homeCamera = this.fullCamera(camera);
  }

  home() {
    return this.flyTo(this.homeCamera || this.cameraFor(this.bounds()));
  }

  /**
   * Ease to a camera; resolves true on arrival, false if interrupted. Jumps under reduced motion.
   * @param {object} partial  any of {target, distance, yaw, pitch}
   * @param {{duration?: number, easing?: Function}} [options]  duration in ms (default by travel, 500-1400)
   */
  flyTo(partial, { duration, easing = easeInOutCubic } = {}) {
    this.stopFlights();
    const from = this.camera;
    const to = this.clampCamera(this.fullCamera(partial));
    if (reducedMotion() || duration === 0) {
      this.setCamera(to);
      return Promise.resolve(true);
    }
    const travel = length(sub(to.target, from.target)) / Math.max(from.distance, to.distance) + Math.abs(Math.log(to.distance / from.distance)) + Math.abs(shortestAngle(from.yaw, to.yaw) - from.yaw) / 90 + Math.abs(to.pitch - from.pitch) / 90;
    const total = duration ?? Math.min(1400, 500 + 450 * travel);
    return new Promise((resolve) => {
      let time = 0;
      const flight = {
        resolve,
        tick: (dt) => {
          time += dt;
          const f = Math.min(1, time / total);
          this.setCamera(interpolateCamera(from, to, easing(f)));
          if (f >= 1) {
            this.flights.delete(flight);
            resolve(true);
            return true;
          }
          return false;
        },
      };
      this.flights.add(flight);
      this.animate(flight);
    });
  }

  stopFlights() {
    for (const flight of this.flights) {
      this.animators.delete(flight);
      flight.resolve(false);
    }
    this.flights.clear();
  }

  /**
   * A camera that frames bounds {min, max}.
   * @param {object} bounds
   * @param {{yaw?: number, pitch?: number, padding?: number}} [options]  padding multiplies the radius (default 1.1)
   */
  cameraFor(bounds, { yaw = this.cam.yaw, pitch = this.cam.pitch, padding = 1.1 } = {}) {
    const center = scale(add(bounds.min, bounds.max), 0.5);
    const radius = Math.max(1e-6, (length(sub(bounds.max, bounds.min)) / 2) * padding);
    const aspect = this.canvas.clientWidth / Math.max(1, this.canvas.clientHeight) || 1;
    const half = (this.options.fov * DEG) / 2;
    const fit = Math.min(Math.tan(half), Math.tan(half) * aspect);
    const distance = this.options.projection === 'orthographic' ? radius / fit : radius / Math.sin(Math.atan(fit));
    return { target: center, distance, yaw, pitch };
  }

  /** Frame bounds (default: all visible content); flies unless `fly` is false. */
  fit(bounds = null, options = {}) {
    const camera = this.cameraFor(bounds || this.bounds(), options);
    return options.fly === false ? (this.setCamera(camera), Promise.resolve(true)) : this.flyTo(camera);
  }

  eye() {
    const { target, distance, yaw, pitch } = this.cam;
    const [y, p] = [yaw * DEG, pitch * DEG];
    return add(target, scale([Math.cos(p) * Math.sin(y), Math.sin(p), Math.cos(p) * Math.cos(y)], distance));
  }

  updateMatrices() {
    const { canvas } = this;
    const aspect = canvas.width / Math.max(1, canvas.height);
    const eye = this.eye();
    const b = this.bounds();
    const radius = length(sub(b.max, b.min)) / 2 + length(sub(scale(add(b.min, b.max), 0.5), this.cam.target));
    const d = this.cam.distance;
    const view = lookAt(eye, this.cam.target, [0, 1, 0]);
    let proj, near, far;
    if (this.options.projection === 'orthographic') {
      const halfH = d * Math.tan((this.options.fov * DEG) / 2);
      near = -(radius * 2 + d);
      far = d + radius * 2;
      proj = orthographic(halfH * aspect, halfH, near, far);
    } else {
      far = d + radius * 2 + 1e-3;
      near = Math.max(far / 5000, Math.min(d * 0.05, Math.max(d - radius * 1.5, d * 0.001)));
      proj = perspective(this.options.fov * DEG, aspect, near, far);
    }
    const viewProj = multiply(proj, view);
    this.matrices = { view, proj, viewProj, inverse: invert(viewProj), near, eye };
  }

  /* The auto-fit a resize or new content asked for, applied before the next frame or projection. */
  applyPendingFit() {
    if (!this.fitPending || !this.canvas.width || !this.canvas.height) return;
    this.fitPending = false;
    this.cam = this.clampCamera(this.cameraFor(this.bounds()));
    this.homeCamera ||= this.camera;
    this.matrices = null;
  }

  /** A 3D point in CSS px over the canvas: {x, y, depth, visible}, valid before the next frame too. */
  project(point) {
    this.applyPendingFit();
    if (!this.matrices) this.updateMatrices();
    const [x, y, z, w] = transform(this.matrices.viewProj, point);
    const w2 = this.canvas.clientWidth, h2 = this.canvas.clientHeight;
    const visible = w > 0 && Math.abs(z / w) <= 1;
    return { x: ((x / w + 1) / 2) * w2, y: ((1 - y / w) / 2) * h2, depth: z / w, visible };
  }

  /** The world-space ray under a client position: {origin, dir}. */
  ray(clientX, clientY) {
    this.applyPendingFit();
    if (!this.matrices) this.updateMatrices();
    const box = this.canvas.getBoundingClientRect();
    const nx = ((clientX - box.left) / box.width) * 2 - 1, ny = 1 - ((clientY - box.top) / box.height) * 2;
    const unproject = (z) => {
      const [x, y, zz, w] = transform(this.matrices.inverse, [nx, ny, z]);
      return [x / w, y / w, zz / w];
    };
    const a = unproject(-1), b = unproject(1);
    return { origin: a, dir: normalize(sub(b, a)), screen: { x: clientX - box.left, y: clientY - box.top } };
  }

  /** The nearest pickable content under a client position, or null (see the header). */
  pick(clientX, clientY) {
    const { origin, dir, screen } = this.ray(clientX, clientY);
    let best = null;
    for (const set of this.sets.values()) {
      if (!set.visible || !set.pickable) continue;
      const hit = set.pick(origin, dir, screen);
      if (hit && (!best || hit.t < best.t)) best = { ...hit, set: set.name, kind: set.kind };
    }
    if (!best) return null;
    const set = this.sets.get(best.set);
    best.id = set.ids ? set.ids[best.index] : best.index;
    best.point = add(origin, scale(dir, best.t));
    return best;
  }

  /* ---------------------------------------------------------------- interaction */
  bind() {
    const canvas = this.canvas;
    const pointers = new Map();
    let gesture = null;
    const user = () => this.stopFlights();
    canvas.addEventListener('contextmenu', (event) => event.preventDefault());
    canvas.addEventListener('pointerdown', (event) => {
      try {
        canvas.setPointerCapture(event.pointerId);
      } catch {
        /* A pointer the browser no longer tracks cannot be captured; the gesture still works. */
      }
      pointers.set(event.pointerId, { x: event.clientX, y: event.clientY, x0: event.clientX, y0: event.clientY, t0: performance.now() });
      const pan = event.shiftKey || event.button === 1 || event.button === 2;
      gesture = pointers.size === 1 ? { mode: pan ? 'pan' : 'orbit', moved: 0 } : { mode: 'pinch', moved: 99, span: spread(), mid: middle() };
      user();
    });
    const spread = () => {
      const [a, b] = [...pointers.values()];
      return Math.hypot(a.x - b.x, a.y - b.y);
    };
    const middle = () => {
      const [a, b] = [...pointers.values()];
      return { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 };
    };
    canvas.addEventListener('pointermove', (event) => {
      const p = pointers.get(event.pointerId);
      if (!p) {
        if (event.pointerType !== 'touch') this.queueHover(event.clientX, event.clientY);
        return;
      }
      const dx = event.clientX - p.x, dy = event.clientY - p.y;
      p.x = event.clientX;
      p.y = event.clientY;
      gesture.moved = Math.max(gesture.moved, Math.hypot(p.x - p.x0, p.y - p.y0));
      if (gesture.mode === 'pinch' && pointers.size === 2) {
        const span = spread(), mid = middle();
        this.zoomAt(gesture.span / Math.max(1, span), mid.x, mid.y);
        this.panBy(mid.x - gesture.mid.x, mid.y - gesture.mid.y);
        gesture.span = span;
        gesture.mid = mid;
      } else if (gesture.mode === 'pan') this.panBy(dx, dy);
      else if (gesture.mode === 'orbit') this.orbitBy(dx, dy);
    });
    const release = (event) => {
      const p = pointers.get(event.pointerId);
      pointers.delete(event.pointerId);
      if (event.type === 'pointerup' && p && gesture && gesture.moved < CLICK_SLOP_PX && pointers.size === 0) this.click(event.clientX, event.clientY);
      if (pointers.size === 0) gesture = null;
    };
    canvas.addEventListener('pointerup', release);
    canvas.addEventListener('pointercancel', release);
    canvas.addEventListener('pointerleave', (event) => {
      if (event.pointerType !== 'touch') this.setHover(null);
    });
    canvas.addEventListener(
      'wheel',
      (event) => {
        if (this.options.wheel !== 'always' && !(event.ctrlKey || event.metaKey)) {
          this.hint('Hold Ctrl (⌘ on a Mac) and scroll to zoom');
          return;
        }
        event.preventDefault();
        user();
        const unit = event.deltaMode === 1 ? 16 : event.deltaMode === 2 ? 400 : 1;
        this.zoomAt(Math.exp(event.deltaY * unit * 0.0015), event.clientX, event.clientY);
      },
      { passive: false },
    );
    canvas.addEventListener('keydown', (event) => {
      const step = 10;
      const pan = 0.06 * canvas.clientHeight;
      const actions = {
        ArrowLeft: () => (event.shiftKey ? this.panBy(pan, 0) : this.orbitBy(0, 0, -step, 0)),
        ArrowRight: () => (event.shiftKey ? this.panBy(-pan, 0) : this.orbitBy(0, 0, step, 0)),
        ArrowUp: () => (event.shiftKey ? this.panBy(0, pan) : this.orbitBy(0, 0, 0, step)),
        ArrowDown: () => (event.shiftKey ? this.panBy(0, -pan) : this.orbitBy(0, 0, 0, -step)),
        '+': () => this.zoomBy(1 / 1.2),
        '=': () => this.zoomBy(1 / 1.2),
        '-': () => this.zoomBy(1.2),
        _: () => this.zoomBy(1.2),
        0: () => this.home(),
        Home: () => this.home(),
        Escape: () => {
          if (this.options.autoSelect) this.clearSelection();
          this.dispatchEvent(new CustomEvent('select', { detail: null }));
        },
      };
      const action = actions[event.key];
      if (!action || event.altKey || event.ctrlKey || event.metaKey) return;
      event.preventDefault();
      if (event.key !== 'Escape') user();
      action();
    });
  }

  /** Orbit by a pointer delta in CSS px, or by degrees (dyaw, dpitch). */
  orbitBy(dx, dy, dyaw = 0, dpitch = 0) {
    const perPx = 360 / Math.max(480, this.canvas.clientWidth);
    this.setCamera({ yaw: this.cam.yaw - dx * perPx + dyaw, pitch: this.cam.pitch + dy * perPx + dpitch });
  }

  /** World units per CSS px at the target's depth. */
  worldPerPixel() {
    const h = this.canvas.clientHeight || 1;
    return (2 * this.cam.distance * Math.tan((this.options.fov * DEG) / 2)) / h;
  }

  /** Move the scene with the pointer by a delta in CSS px. */
  panBy(dx, dy) {
    const eye = this.eye();
    const forward = normalize(sub(this.cam.target, eye));
    const right = normalize(cross(forward, [0, 1, 0]));
    const up = cross(right, forward);
    const k = this.worldPerPixel();
    this.setCamera({ target: add(this.cam.target, add(scale(right, -dx * k), scale(up, dy * k))) });
  }

  /** Multiply the distance by `factor` (< 1 zooms in). */
  zoomBy(factor) {
    this.setCamera({ distance: this.cam.distance * factor });
  }

  /* Zoom keeping the point under the pointer (on the target's plane) where it is. */
  zoomAt(factor, clientX, clientY) {
    const before = this.cam.distance;
    const after = this.clampCamera({ ...this.camera, distance: before * factor }).distance;
    const box = this.canvas.getBoundingClientRect();
    const dx = clientX - box.left - box.width / 2, dy = clientY - box.top - box.height / 2;
    const k = this.worldPerPixel();
    const eye = this.eye();
    const forward = normalize(sub(this.cam.target, eye));
    const right = normalize(cross(forward, [0, 1, 0]));
    const up = cross(right, forward);
    const point = add(this.cam.target, add(scale(right, dx * k), scale(up, -dy * k)));
    const target = add(point, scale(sub(this.cam.target, point), after / before));
    this.setCamera({ distance: after, target });
  }

  hint(text) {
    this.message.textContent = text;
    this.message.hidden = false;
    this.message.classList.add('viz-scene-hint');
    clearTimeout(this.hintTimer);
    this.hintTimer = setTimeout(() => {
      if (!this.lost) this.message.hidden = true;
    }, 1400);
  }

  queueHover(clientX, clientY) {
    this.pendingHover = [clientX, clientY];
    if (this.hoverFrame) return;
    this.hoverFrame = requestAnimationFrame(() => {
      this.hoverFrame = 0;
      if (!this.pendingHover) return;
      const [x, y] = this.pendingHover;
      this.setHover(this.pick(x, y));
    });
  }

  setHover(hit) {
    const same = (a, b) => a === b || (a && b && a.set === b.set && a.index === b.index && a.segment === b.segment);
    if (same(hit, this.hoverHit)) return;
    const previous = this.hoverHit;
    this.hoverHit = hit;
    this.canvas.style.cursor = hit ? 'pointer' : '';
    if (this.options.autoHover) {
      if (previous) {
        const set = this.get(previous.set);
        if (set?.kind === 'plane') set.setHighlight(null, 'hover');
        else set?.setHovered(null);
      }
      if (hit) {
        const set = this.get(hit.set);
        if (set.kind === 'plane') set.setHighlight({ col: hit.col, row: hit.row }, 'hover');
        else set.setHovered(hit.index);
      }
    }
    this.dispatchEvent(new CustomEvent('hover', { detail: hit }));
  }

  click(clientX, clientY) {
    const hit = this.pick(clientX, clientY);
    if (this.options.autoSelect) {
      this.clearSelection();
      if (hit) this.select(hit.set, hit.kind === 'plane' ? { col: hit.col, row: hit.row } : hit.index);
    }
    this.dispatchEvent(new CustomEvent('select', { detail: hit }));
  }

  /** Mark an item (an index, or a cell rect on a plane) as selected, without an event. */
  select(name, which) {
    const set = this.get(name);
    if (!set) return;
    if (set.kind === 'plane') set.setHighlight(which, 'select');
    else set.setSelected(which);
  }

  clearSelection() {
    for (const set of this.sets.values()) {
      if (set.kind === 'plane') set.setHighlight(null, 'select');
      else set.setSelected(null);
    }
  }

  /* ---------------------------------------------------------------- frames */
  animate(animator) {
    this.animators.add(animator);
    this.invalidate();
  }

  /** Schedule one frame; `boundsChanged` refits the camera while no one has moved it. */
  invalidate(boundsChanged = false) {
    if (boundsChanged) this.matrices = null;
    if (boundsChanged && this.autoFit && this.sets.size) this.fitPending = true;
    if (this.frame || this.lost) return;
    this.frame = requestAnimationFrame((now) => this.tick(now));
  }

  tick(now) {
    this.frame = 0;
    const dt = this.lastTick && this.animators.size ? Math.min(100, now - this.lastTick) : 16;
    this.lastTick = now;
    for (const animator of [...this.animators]) if (animator.tick(dt)) this.animators.delete(animator);
    this.render();
    if (this.animators.size) this.invalidate();
    else this.lastTick = 0;
  }

  /** Draw now (normally scheduled by invalidate). */
  render() {
    const gl = this.gl;
    if (this.lost || !this.canvas.width || !this.canvas.height) return;
    this.applyPendingFit();
    this.updateMatrices();
    gl.viewport(0, 0, this.canvas.width, this.canvas.height);
    const bg = this.palette.background;
    gl.clearColor(bg[0], bg[1], bg[2], 1);
    gl.depthMask(true);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
    const visible = [...this.sets.values()].filter((set) => set.visible);
    for (const set of visible) set.upload(gl, this.matrices.eye);
    for (const pass of ['opaque', 'lines', 'glass', 'top']) {
      gl.depthMask(pass === 'opaque');
      if (pass === 'top') gl.disable(gl.DEPTH_TEST);
      for (const set of visible) set.draw(gl, pass);
    }
    gl.enable(gl.DEPTH_TEST);
    gl.depthMask(true);
    this.placeLabels();
    if (this.viewChanged) {
      this.viewChanged = false;
      this.dispatchEvent(new CustomEvent('view', { detail: this.camera }));
    }
  }

  resize() {
    const ratio = Math.min(window.devicePixelRatio || 1, MAX_DPR);
    const w = Math.round(this.canvas.clientWidth * ratio), h = Math.round(this.canvas.clientHeight * ratio);
    this.dpr = ratio;
    if (!w || !h) return;
    if (w !== this.canvas.width || h !== this.canvas.height) {
      this.canvas.width = w;
      this.canvas.height = h;
      this.matrices = null;
      for (const label of this.labels) label.measured = null;
    }
    if (this.autoFit) this.fitPending = true;
    this.invalidate();
  }

  watchPixelRatio() {
    this.ratioQuery?.removeEventListener('change', this.onRatio);
    this.ratioQuery = matchMedia(`(resolution: ${window.devicePixelRatio || 1}dppx)`);
    this.onRatio = () => {
      this.resize();
      this.watchPixelRatio();
    };
    this.ratioQuery.addEventListener('change', this.onRatio);
  }

  /* A lost context keeps every array; restoring rebuilds programs, buffers and textures from them. */
  handleContextLoss() {
    this.canvas.addEventListener('webglcontextlost', (event) => {
      event.preventDefault();
      this.lost = true;
      cancelAnimationFrame(this.frame);
      this.frame = 0;
      for (const set of this.sets.values()) set.dispose(null);
      for (const set of this.sets.values()) {
        set.gl = null;
        for (const group of set.groups || []) group.gl = null;
      }
      this.message.classList.remove('viz-scene-hint');
      this.message.textContent = 'The 3D view lost its graphics context; it redraws when the browser restores it.';
      this.message.hidden = false;
      this.dispatchEvent(new CustomEvent('contextlost'));
    });
    this.canvas.addEventListener('webglcontextrestored', () => {
      this.lost = false;
      this.initGL();
      for (const set of this.sets.values()) {
        set.dirty = true;
        if (set.kind === 'plane') set.dirtyValues = set.dirtyLut = true;
        for (const group of set.groups || []) group.dirty = group.stateDirty = true;
      }
      this.message.hidden = true;
      this.invalidate();
      this.dispatchEvent(new CustomEvent('contextrestored'));
    });
  }

  destroy() {
    cancelAnimationFrame(this.frame);
    cancelAnimationFrame(this.hoverFrame);
    this.resizer.disconnect();
    this.ratioQuery?.removeEventListener('change', this.onRatio);
    this.unTheme();
    this.stopFlights();
    if (!this.lost) for (const set of this.sets.values()) set.dispose(this.gl);
    this.canvas.remove();
    this.overlay.remove();
    this.message.remove();
  }
}

