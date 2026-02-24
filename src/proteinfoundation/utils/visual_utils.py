import os
import random
import tempfile
from pathlib import Path
from typing import Literal

import ammolite
import biotite.structure as struc
import biotite.structure.io.pdb as pdb
import numpy as np


def convert_ss_annotations(ss_array: np.ndarray) -> np.ndarray:
    """Convert secondary structure annotations from Biotite to PyMOL format.

    Args:
        ss_array: Array of secondary structure annotations from Biotite.

    Returns:
        Array of secondary structure annotations in PyMOL format.
    """
    ss_mapping = {"c": "L", "a": "H", "b": "S"}
    return np.vectorize(ss_mapping.get)(ss_array)


def superimpose(vec1: np.ndarray, vec2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Calculate the rotation matrix and translation vector to superimpose vec2 onto vec1.

    Args:
        vec1: Reference vector of shape (n, 3).
        vec2: Vector to be superimposed of shape (n, 3).

    Returns:
        Tuple containing the rotation matrix and translation vector.
    """
    assert vec1.shape == vec2.shape, f"{vec1.shape} {vec2.shape}"
    n = vec1.shape[0]  # total points

    centroid1 = np.mean(vec1, axis=0)
    centroid2 = np.mean(vec2, axis=0)

    # Center the points
    vec1_centered = vec1 - centroid1
    vec2_centered = vec2 - centroid2

    # Singular Value Decomposition
    H = vec1_centered.T @ vec2_centered
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # Handle reflection case
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    t = centroid2 - centroid1 @ R
    return R, t


def apply_highres_settings() -> None:
    """Apply settings for high resolution rendering in PyMOL."""
    ammolite.cmd.set("cartoon_discrete_colors", 1)
    ammolite.cmd.set("cartoon_oval_length", 1.0)
    ammolite.cmd.set("ray_trace_mode", 1)
    ammolite.cmd.set("ambient", 0.1)


def visualise_structure(ca: struc.AtomArray, show_secondary_structure: bool = True) -> ammolite.PyMOLObject:
    """Visualize a protein structure with optional secondary structure coloring.

    Args:
        ca: C-alpha atom array of the protein structure.
        show_secondary_structure: Whether to color by secondary structure.

    Returns:
        PyMOL object representing the visualized structure.
    """
    pymol_object = ammolite.PyMOLObject.from_structure(ca)

    if show_secondary_structure:
        sse = convert_ss_annotations(struc.annotate_sse(ca))

        # Color each secondary structure element
        selections = {"L": [], "H": [], "S": []}
        for resi, ss in zip(ca.res_id, sse, strict=False):
            selections[ss].append(resi)

        colors = {"L": "white", "H": "salmon", "S": "lightblue"}
        for ss, color in colors.items():
            selection_string = f'resi {"+".join(map(str, selections[ss]))} and model {pymol_object.name}'
            ammolite.cmd.color(color, selection_string)

        # Show secondary structure as cartoon
        for resi, ss in zip(ca.res_id, sse, strict=False):
            ammolite.cmd.alter(f"resi {resi}", f'ss="{ss}"')

    return pymol_object


def add_spheres_at_residues(pymol_object: ammolite.PyMOLObject, res_ids: np.ndarray) -> str:
    """Add spheres at specified residue positions.

    Args:
        pymol_object: PyMOL object representing the protein structure.
        res_ids: Array of residue IDs to add spheres at.

    Returns:
        Selection string for the added spheres.
    """
    selection_string = f'resi {"+".join(map(str, res_ids))} and name CA and model {pymol_object.name}'
    ammolite.cmd.show("spheres", selection_string)
    ammolite.cmd.set("sphere_scale", 0.2, selection_string)
    return selection_string


def visualise_ca_coord_numpy(
    coords: np.ndarray,
    cond_res: np.ndarray = None,
    zoom_out: bool = True,
    save_path: str = None,
    save_type: Literal["png", "png-highres", "pse"] = "png",
) -> ammolite.PyMOLObject:
    """Visualize a protein structure with optional annotations and save the image.

    Args:
        coords: Coordinates of the backbone CA atoms of shape (n, 3).
        cond_res: Indices of the conditioned residues to annotate.
        zoom_out: Whether to zoom out to show the full structure.
        save_path: Path to save the image. If None, the image is not saved.
        save_type: How to save the image. Options are "png", "png-highres", and "pse".

    Returns:
        PyMOL object representing the visualized structure.
    """
    ammolite.reset()
    ca = struc.from_template(struc.AtomArray(coords))
    pymol_object = visualise_structure(ca)

    if cond_res is not None:
        selection = add_spheres_at_residues(pymol_object, cond_res + 1)  # +1 for 1-indexing in Biotite
        ammolite.cmd.color("purple", selection)
        ammolite.cmd.zoom(f"resi {max(min(cond_res) - 2, 0)}-{min(max(cond_res) + 2, len(ca))}")

    if zoom_out:
        ammolite.cmd.zoom("all")

    ammolite.cmd.set("cartoon_transparency", 0.6, pymol_object.name)

    if save_path:
        if save_type == "png":
            ammolite.cmd.png(f"{save_path!s}.png", width=500, height=330, dpi=100, ray=0)
        elif save_type == "png-highres":
            apply_highres_settings()
            ammolite.cmd.png(f"{save_path!s}.png", width=3000, height=2000, dpi=300, ray=1)
        elif save_type == "pse":
            ammolite.cmd.save(f"{save_path!s}.pse")
        return save_path
    else:
        return pymol_object


def visualise_ca_coord_pdb(
    pdb_file: str,
    cond_res: np.ndarray | None = None,
    motif_pdb_file: str | None = None,
    zoom_out: bool = True,
    save_path: str | None = None,
    save_type: Literal["png", "png-highres", "pse"] = "png",
) -> ammolite.PyMOLObject:
    """Visualize a protein structure from a PDB file containing only C-alpha coordinates.

    Args:
        pdb_file: Path to the PDB file containing only C-alpha coordinates.
        cond_res: Indices of the conditioned residues to annotate.
        motif_pdb_file: Path to a PDB file containing the motif to be visualized.
        zoom_out: Whether to zoom out to show the full structure.
        save_path: Path to save the image. If None, the image is not saved.
        save_type: How to save the image. Options are "png", "png-highres", and "pse".


    Returns:
        PyMOL object representing the visualized structure.
    """
    ammolite.reset()

    # Load the PDB file containing only C-alpha coordinates
    pdb_file = pdb.PDBFile.read(pdb_file)
    ca = pdb_file.get_structure(model=1)

    # Ensure that the structure contains only C-alpha atoms
    ca = ca[ca.atom_name == "CA"]

    pymol_object = visualise_structure(ca)

    if cond_res is not None:
        selection = add_spheres_at_residues(pymol_object, cond_res)  # +1 for 1-indexing in Biotite
        ammolite.cmd.color("purple", selection)
        ammolite.cmd.zoom(f"resi {max(min(cond_res) - 2, 0)}-{min(max(cond_res) + 2, len(ca))}")

    if motif_pdb_file is not None:
        # Load the PDB file containing the motif to be visualized
        motif_pdb_file = pdb.PDBFile.read(motif_pdb_file)

        # Ensure that the structure contains only C-alpha atoms
        ca_motif = motif_pdb_file.get_structure(model=1)
        ca_motif.bonds = struc.connect_via_distances(ca_motif)
        ca_motif = ca_motif[ca_motif.atom_name == "CA"]

        # align the motif to the corresponding conditioned residues in the main structure
        cond_res_0indexed = cond_res - ca[0].res_id
        ca_cond_res = ca[cond_res_0indexed]
        R, t = superimpose(ca_cond_res.coord, ca_motif.coord)
        ca_motif.coord = np.dot(ca_motif.coord, R) - t
        pymol_object_motif = visualise_structure(ca_motif)
        selection_string = f"name CA and model {pymol_object_motif.name}"
        ammolite.cmd.color("green", selection_string)

    if zoom_out:
        ammolite.cmd.zoom("all")

    ammolite.cmd.set("cartoon_trace_atoms", 1)
    ammolite.cmd.set("ribbon_trace_atoms", 1)
    ammolite.cmd.show("cartoon")
    ammolite.cmd.set("cartoon_transparency", 0.6)

    if save_path:
        if save_type == "png":
            ammolite.cmd.png(f"{save_path!s}.png", width=500, height=330, dpi=100, ray=0)
        elif save_type == "png-highres":
            apply_highres_settings()
            ammolite.cmd.png(f"{save_path!s}.png", width=3000, height=2000, dpi=300, ray=1)
        elif save_type == "pse":
            ammolite.cmd.save(f"{save_path!s}.pse")
        return save_path
    else:
        if motif_pdb_file is not None:
            return pymol_object, pymol_object_motif
        else:
            return pymol_object


def visualize_directory(
    pdb_folder: str,
    save_folder: str,
    save_for_publication: bool = True,
    limit: int = -1,
):
    """Visualize protein structures in the `pdb_folder`, and saves images in save_folder.

    Args:
        pdb_folder: directory containing pdb files.
        save_folder: directory where the figures will be saved.
        create_save_if_not_exists: whether to create the save directory if it does not exist.
        limit: if not -1, it plots that number of samples (in case we don't want to plot all samples in the directory)

    Returns:
        None.
    """
    if not os.path.exists(save_folder):
        print("Saving directory does not exist. Will not plot.")

    pdb_files = [str(f) for f in Path(pdb_folder).rglob("*.pdb")]
    if limit > 0:
        random.shuffle(pdb_files)
        pdb_files = pdb_files[:limit]
    filenames_without_extension = [Path(file).stem for file in pdb_files]
    save_type = "png-highres" if save_for_publication else "png"

    for pdb_file, pdb_name in zip(pdb_files, filenames_without_extension, strict=False):
        print(f"Visualising {pdb_name}")
        visualise_ca_coord_pdb(
            pdb_file=pdb_file,
            save_path=f"{save_folder}/{pdb_name}.png",
            save_type=save_type,
        )


def visualise_trajectory(trajectory_path: Path, save_dir: Path, sample_name: str) -> None:
    """Create a PyMOL session from a PDB trajectory file."""
    save_dir.mkdir(parents=True, exist_ok=True)

    # Initialize PyMOL
    ammolite.cmd.reinitialize()
    pymol_objects = []

    # Process each model in the PDB file
    current_model = []
    model_count = 0

    with open(trajectory_path) as f:
        for line in f:
            if line.startswith("MODEL"):
                current_model = []
                model_count += 1
            elif line.startswith("ENDMDL"):
                # Write current model to temporary file
                with tempfile.NamedTemporaryFile(mode="w", suffix=".pdb", delete=False) as temp_f:
                    temp_f.write("".join(current_model))
                    temp_path = Path(temp_f.name)

                # Process the model
                pdb_file = pdb.PDBFile.read(temp_path)
                ca = pdb_file.get_structure(model=1)
                ca = ca[ca.atom_name == "CA"]
                pymol_obj = visualise_structure(ca, show_secondary_structure=True)
                pymol_objects.append(pymol_obj)

                # Clean up temporary file
                temp_path.unlink()
            else:
                current_model.append(line)

    # Save the PyMOL session using the existing function
    save_pymol_session(pymol_objects, save_dir, sample_name)


def save_pymol_session(pymol_objects: list, save_dir: Path, sample_name: str) -> None:
    """Save PyMOL session."""
    for i, obj in enumerate(pymol_objects[1:], start=2):
        ammolite.cmd.create(pymol_objects[0].name, obj.name, 1, i)
        ammolite.cmd.delete(obj.name)
    session_path = save_dir / f"trajectory_{sample_name}.pse"
    ammolite.cmd.save(str(session_path))
    ammolite.cmd.delete("all")


def visualize_directory_all_atom(
    pdb_folder: str,
    save_folder: str,
    limit: int = -1,
):
    """Visualize protein structures in the `pdb_folder` (should be all atom), and saves images in save_folder.

    Args:
        pdb_folder: directory containing pdb files.
        save_folder: directory where the figures will be saved, should already exist.
        limit: if not -1, it plots that number of samples (in case we don't want to plot all samples in the directory)

    Returns:
        None.
    """
    if not os.path.exists(save_folder):
        print("Saving directory does not exist. Will not plot.")

    pdb_files = [str(f) for f in Path(pdb_folder).rglob("*.pdb")]
    if limit > 0:
        random.shuffle(pdb_files)
        pdb_files = pdb_files[:limit]
    filenames_without_extension = [Path(file).stem for file in pdb_files]

    for pdb_file, pdb_name in zip(pdb_files, filenames_without_extension, strict=False):
        print(f"Visualising {pdb_name}")
        visualize_pdb_all_atom(
            pdb_file=pdb_file,
            save_path=f"{save_folder}/{pdb_name}.png",
        )


def visualize_pdb_all_atom(
    pdb_file: str,
    save_path: str,
):
    """
    Visualize a PDB file with all atoms.

    Args:
        pdb_file: path to the PDB file.
        save_path: path to save the image.
    """
    ammolite.cmd.reinitialize()
    ammolite.cmd.set("ray_trace_frames", 0)  # Disable ray tracing
    ammolite.cmd.viewport(400, 300)  # Sets a smaller viewport size, reducing output image resolution
    # ammolite.cmd.viewport(800, 600)

    ammolite.cmd.load(pdb_file, "protein")  # Load the PDB file
    ammolite.cmd.show("cartoon")  # Show protein as a cartoon
    # ammolite.cmd.show("sticks", "protein and not (name C+N+O+CA)")  # Side chain as sticks
    ammolite.cmd.show("lines", "protein and not (name C+N+O+CA)")
    ammolite.cmd.color("blue", "protein")  # Color the protein blue
    ammolite.cmd.color("red", "protein and not (name C+N+O+CA)")  # Color the side chains red
    ammolite.cmd.color("yellow", "ss h")  # Color alpha helices yellow
    ammolite.cmd.color("blue", "ss s")  # Color beta sheets blue
    ammolite.cmd.color("violet", "ss l")  # Color loops green
    ammolite.cmd.png(save_path, dpi=72, quiet=1)  # Lower DPI and disable ray tracing directly in the png command
