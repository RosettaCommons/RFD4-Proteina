// Interactive structure viewers for the experimental results.
// Coordinates load lazily with 3Dmol.js (vendor/3Dmol-min.js); every structure goes through the
// shared style presets. A pill may name an alternative state (data-alt-file), for example the
// refolded structure of a design, aligned onto it.
import { palette, applyStyle, fetchStructure } from "./structure-core.js";

const motion = matchMedia("(prefers-reduced-motion: reduce)");

class StructureView {
  constructor(root) {
    this.sv = root;
    this.stage = root.querySelector(".pdb-viewer-box");
    this.pills = [...root.querySelectorAll(".pdb-pill")];
    this.note = root.querySelector(".pdb-viewer-caption");
    this.preset = root.dataset.preset;
    this.visible = false;
    this.spinning = true;
    this.current = null;
    this.state = "main";
    this.buildChrome();
    this.pills.forEach((pill) =>
      pill.addEventListener("click", () => this.select(pill)),
    );
    new IntersectionObserver(
      ([entry]) => {
        this.visible = entry.isIntersecting;
        if (this.visible && !this.current) this.select(this.pills[0]);
        this.updateSpin();
      },
      { rootMargin: "300px 0px", threshold: 0.01 },
    ).observe(this.stage);
    document.addEventListener("visibilitychange", () => this.updateSpin());
    document.addEventListener("themechange", () => this.restyle());
    motion.addEventListener("change", () => this.updateSpin());
  }

  buildChrome() {
    this.canvas = document.createElement("div");
    this.canvas.className = "sv-canvas";
    this.status = document.createElement("p");
    this.status.className = "sv-status";
    this.status.setAttribute("role", "status");
    this.status.textContent = "Loading structure…";
    this.states = document.createElement("div");
    this.states.className = "sv-states";
    this.states.setAttribute("role", "group");
    this.states.hidden = true;
    this.stateButtons = {};
    for (const state of ["main", "alt"]) {
      const button = document.createElement("button");
      button.type = "button";
      button.addEventListener("click", () => this.select(this.current, state));
      this.stateButtons[state] = button;
      this.states.append(button);
    }
    this.stage.append(this.canvas, this.states, this.status);

  }

  ensureViewer() {
    if (this.viewer) return true;
    if (!window.$3Dmol) return false;
    this.viewer = $3Dmol.createViewer(this.canvas, {
      backgroundColor: palette().background,
      antialias: true,
    });
    this.canvas.addEventListener("pointerdown", () => this.viewer.spin(false));
    return true;
  }

  async select(pill, state = "main") {
    const hasAlt = Boolean(pill.dataset.altFile);
    if (!hasAlt) state = "main";
    const sameStructure = pill === this.current;
    this.pills.forEach((p) => {
      const active = p === pill;
      p.classList.toggle("active", active);
      p.setAttribute("aria-pressed", String(active));
    });
    this.current = pill;
    this.state = state;
    this.note.innerHTML = pill.dataset.note || "";
    this.states.hidden = !hasAlt;
    if (hasAlt) {
      this.stateButtons.main.innerHTML = pill.dataset.label || "Design";
      this.stateButtons.alt.innerHTML = pill.dataset.altLabel || "Refolded";
      for (const [name, button] of Object.entries(this.stateButtons)) {
        button.classList.toggle("active", name === state);
        button.setAttribute("aria-pressed", String(name === state));
      }
    }
    const token = (this.token = Symbol());
    this.status.hidden = false;
    this.status.textContent = "Loading structure…";
    const file = state === "alt" ? pill.dataset.altFile : pill.dataset.file;
    try {
      if (!this.ensureViewer()) throw new Error("3Dmol.js is unavailable");
      const text = await fetchStructure(file);
      if (token !== this.token) return;
      const view = sameStructure ? this.viewer.getView() : null;
      this.viewer.clear();
      this.viewer.addModel(text, file.endsWith(".cif") ? "cif" : "pdb");
      applyStyle(this.viewer, pill.dataset, this.preset);
      if (view) this.viewer.setView(view);
      else this.viewer.zoomTo();
      this.viewer.resize();
      this.viewer.render();
      this.updateSpin();
    } catch (error) {
      if (token !== this.token) return;
      console.error(error);
      this.status.textContent =
        "This structure could not be loaded. Download it with the button below.";
      return;
    }
    if (token === this.token) this.status.hidden = true;
  }

  restyle() {
    if (!this.viewer || !this.current) return;
    applyStyle(this.viewer, this.current.dataset, this.preset);
    this.viewer.render();
  }

  updateSpin() {
    const run =
      this.spinning && this.visible && !document.hidden && !motion.matches;
    this.viewer?.spin(run ? "y" : false);
  }
}

document
  .querySelectorAll(".pdb-viewer-inline")
  .forEach((root) => new StructureView(root));
