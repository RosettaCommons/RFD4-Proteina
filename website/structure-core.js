// Shared structure loading and styling for the hero and the wet-lab viewers.
const textCache = new Map();

const light = {
  background: "#f0f0f0",
  binder: "#186c70",
  target: "#8b969d",
  surface: "#ffffff",
  peptide: "#c4963b",
  glycanBlue: "#4b7bc8",
  glycanGreen: "#4fa86b",
  metal: "#7d8a92",
};
const dark = {
  background: "#1b1d27",
  binder: "#3fb6ad",
  target: "#8d99a6",
  surface: "#cfd8e0",
  peptide: "#d7b666",
  glycanBlue: "#6f9be0",
  glycanGreen: "#6fcf8a",
  metal: "#aab5bd",
};

export function palette() {
  return document.documentElement.dataset.theme === "dark" ? dark : light;
}

export const list = (value) => (value ? value.split(",").map((v) => v.trim()) : []);

export async function fetchStructure(file) {
  const url = `assets/structures/${file}.gz`;
  if (!textCache.has(url)) {
    textCache.set(
      url,
      fetch(url).then((response) => {
        if (!response.ok) throw new Error(`${url}: ${response.status}`);
        return new Response(
          response.body.pipeThrough(new DecompressionStream("gzip")),
        ).text();
      }),
    );
  }
  return textCache.get(url);
}

function residuesNear(viewer, protein, ligand, cutoff) {
  const ligandAtoms = viewer.selectedAtoms({ chain: ligand });
  const residues = new Set();
  for (const atom of viewer.selectedAtoms({ chain: protein })) {
    if (residues.has(atom.resi)) continue;
    for (const other of ligandAtoms) {
      const dx = atom.x - other.x,
        dy = atom.y - other.y,
        dz = atom.z - other.z;
      if (dx * dx + dy * dy + dz * dz < cutoff * cutoff) {
        residues.add(atom.resi);
        break;
      }
    }
  }
  return [...residues];
}

export function applyStyle(viewer, data, preset, options = {}) {
  const colors = palette();
  if (options.transparent) viewer.setBackgroundColor(0x000000, 0);
  else viewer.setBackgroundColor(colors.background);
  viewer.removeAllSurfaces();
  viewer.setStyle({}, {});
  const cartoon = (color) => ({ cartoon: { color } });
  const peptide = {
    cartoon: { color: colors.peptide },
    stick: { radius: 0.17, colorscheme: "orangeCarbon" },
  };
  const surface = (chains, opacity) =>
    data.surface === "none" ||
    viewer.addSurface(
      $3Dmol.SurfaceType.VDW,
      { color: colors.surface, opacity: colors === dark ? opacity * 0.45 : opacity },
      { chain: chains },
    );

  if (preset === "binder") {
    viewer.setStyle({ chain: list(data.target) }, cartoon(colors.target));
    viewer.setStyle({ chain: list(data.binder) }, cartoon(colors.binder));
    surface(list(data.target), 0.6);
  } else if (preset === "pmhc") {
    viewer.setStyle({ chain: list(data.target) }, cartoon(colors.target));
    viewer.setStyle({ chain: list(data.binder) }, cartoon(colors.binder));
    viewer.setStyle({ chain: list(data.peptide) }, peptide);
    surface(list(data.target), 0.6);
  } else if (preset === "idr") {
    viewer.setStyle({ chain: list(data.binder) }, cartoon(colors.binder));
    viewer.setStyle({ chain: list(data.peptide) }, peptide);
    surface(list(data.binder), 0.4);
  } else if (preset === "enzyme") {
    const protein = list(data.protein);
    const ligand = list(data.ligand);
    viewer.setStyle({ chain: protein }, cartoon(colors.binder));
    viewer.addStyle(
      { chain: ligand },
      { stick: { radius: 0.17, colorscheme: "orangeCarbon" } },
    );
    if (data.metal)
      viewer.setStyle(
        { chain: list(data.metal) },
        { sphere: { radius: 0.9, color: colors.metal } },
      );
    const near = residuesNear(viewer, protein, ligand, 4.5);
    if (near.length)
      viewer.addStyle(
        { chain: protein, resi: near },
        { stick: { radius: 0.13, colorscheme: "cyanCarbon" } },
      );
  } else if (preset === "glycan") {
    viewer.setStyle({ chain: list(data.protein) }, cartoon(colors.binder));
    const byResidue = {
      NAG: colors.glycanBlue,
      BMA: colors.glycanGreen,
      MAN: colors.glycanGreen,
    };
    for (const [resn, color] of Object.entries(byResidue))
      viewer.setStyle(
        { chain: list(data.glycan), resn },
        { stick: { radius: 0.2, color }, sphere: { radius: 0.35, color } },
      );
  }
}

