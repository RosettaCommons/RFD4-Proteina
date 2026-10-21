// Four interactive designs under the logo. A pre-rendered poster shows immediately and the live
// 3Dmol.js view fades in over it once the (gzipped) structure has loaded.
import { palette, applyStyle, fetchStructure } from "./structure-core.js";

const poster = new URLSearchParams(location.search).has("poster");
const motion = matchMedia("(prefers-reduced-motion: reduce)");
const views = [];

async function start(figure) {
  const canvas = figure.querySelector(".hero-canvas");
  const loading = fetchStructure(figure.dataset.file);
  if (!window.$3Dmol) await new Promise((resolve) => addEventListener("load", resolve));
  const viewer = $3Dmol.createViewer(canvas, { backgroundAlpha: 0, antialias: true });
  const view = { figure, viewer, visible: true };
  views.push(view);
  viewer.addModel(await loading, figure.dataset.file.endsWith(".cif") ? "cif" : "pdb");
  applyStyle(viewer, figure.dataset, figure.dataset.preset, { transparent: true });
  viewer.zoomTo();
  viewer.render();
  figure.dataset.ready = "true";
  canvas.addEventListener("pointerdown", () => viewer.spin(false));
  // Let the page scroll over the viewer; zoom only with Ctrl/Cmd held.
  canvas.addEventListener(
    "wheel",
    (event) => {
      if (!event.ctrlKey && !event.metaKey) event.stopImmediatePropagation();
    },
    true,
  );
  new IntersectionObserver(([entry]) => {
    view.visible = entry.isIntersecting;
    spin(view);
  }).observe(figure);
  spin(view);
}

function spin(view) {
  const run = view.visible && !document.hidden && !motion.matches && !poster;
  view.viewer.spin(run ? "y" : false, 0.6);
}

document.querySelectorAll(".hero-struct").forEach((figure) => start(figure).catch(console.error));
document.addEventListener("visibilitychange", () => views.forEach(spin));
document.addEventListener("themechange", () => {
  for (const { figure, viewer } of views) {
    applyStyle(viewer, figure.dataset, figure.dataset.preset, { transparent: true });
    viewer.render();
  }
});
