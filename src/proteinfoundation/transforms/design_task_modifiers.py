import ast
import os
import shutil
import string
import subprocess
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from atomworks.io.utils.annotator import ensure_annotations
from atomworks.io.utils.atom_array_plus import as_atom_array_plus
from atomworks.io.utils.io_utils import to_pdb_string
from atomworks.io.utils.selection import get_residue_starts
from atomworks.ml.conditions import (
    C_CRD,
    C_DIS,
    C_SEQ,
)
from atomworks.ml.transforms.sasa import calculate_atomwise_rasa
from biotite.structure import AtomArray, spread_residue_wise

from proteinfoundation.conditions import (
    C_CATH,
    C_HB,
    C_INTCRD,
    C_RAS,
    C_SEC,
    HBondEnum,
    RasaEnum,
)
from proteinfoundation.metrics.sse_metrics import annotate_secondary_structure
from proteinfoundation.transforms.utils.conditioning_utils import sample_island_tokens


def add_rasa_condition(
    data: dict,
    probe_radius: float = 1.4,
    atom_radii: str | np.ndarray = "ProtOr",
    point_number: int = 100,
    p_include_non_ligand: bool = 0.1,
    rasa_max: float = 0.2,
) -> dict:
    """
    probe_radius (float, optional): Van-der-Waals radius of the probe in Angstrom. Defaults to
        1.4 (for water).
    atom_radii (str | np.ndarray, optional): Atom radii set to use for calculation. Defaults to
        "ProtOr". "ProtOr" will not get sasa's for hydrogen atoms and some other atoms, like ions
        or certain atoms with charges
    point_number (int, optional): Number of points in the Shrake-Rupley algorithm to sample for
        calculating SASA. Defaults to 100.
    p_include_non_ligand (float, optional): Probability of providing RASA conditioning for
        non-ligand atoms. Defaults to 0.1.
    rasa_max (float, optional): Maximum RASA value for binning. Defaults to 0.2.
    """
    atom_array = as_atom_array_plus(data["atom_array"])

    ensure_annotations(atom_array, "is_ligand", "atomize")

    # Determine valid residues for rasa conditioning
    if np.random.rand() < p_include_non_ligand:
        seq_mask = C_SEQ.mask(atom_array, default="generate")
        crd_mask = C_CRD.mask(atom_array, default="generate") | C_INTCRD.mask(atom_array, default="generate")
        dis_mask = C_DIS.mask(atom_array, default="generate").as_dense_array(False).any(axis=0)
        inclusion_mask = seq_mask | crd_mask | dis_mask
    else:
        inclusion_mask = atom_array.is_ligand | atom_array.atomize

    rasa = calculate_atomwise_rasa(atom_array, probe_radius, atom_radii, point_number, atom_filter=inclusion_mask)

    # NOTE: Bin count is hard-coded at 3 to match the RasaEnum
    bin_edges = np.linspace(0, rasa_max, 3)  # e.g., [0.0, 0.1, 0.2]
    bins = np.digitize(rasa, bin_edges, right=False)  # bins in [0, 3] -- however, bin 0 would mean negative rasa!
    if (bins == 0).any():
        raise ValueError("Negative RASA value was detected. This indicates an error.")

    # Convert to RasaEnum
    enum_rasa_values = C_RAS.default_annotation(atom_array)
    enum_rasa_values[bins == 1] = RasaEnum.BURIED
    enum_rasa_values[bins == 2] = RasaEnum.INTERMEDIATE
    enum_rasa_values[bins == 3] = RasaEnum.EXPOSED
    enum_rasa_values[~inclusion_mask] = RasaEnum.UNSPECIFIED

    C_RAS.set_annotation(atom_array, enum_rasa_values.astype(int))

    data["atom_array"] = atom_array
    return data


@contextmanager
def temporary_pdb_directory(atom_array: AtomArray, raise_on_failure: bool = True) -> Any:
    """Context manager that creates a temporary directory with a PDB file.

    Creates a temporary directory and writes the PDB file there. This ensures
    hbplus output files are created in a known location and can be easily cleaned up.

    Yields:
        tuple: (temp_dir, pdb_filename, atom_array, nan_mask, inverted_chain_map, failed)
               or (None, None, None, None, None, True) on failure
    """

    def _handle_nan_coords(atom_array: AtomArray, noise_level: float = 1e-3) -> tuple:
        coords = atom_array.coord
        nan_mask = np.isnan(coords)
        coords[nan_mask] = np.random.uniform(-noise_level, noise_level, size=nan_mask.sum())
        atom_array.coord = coords
        return atom_array, nan_mask

    # To avoid modifying the original atom_array with this preprocessing
    atom_array = atom_array.copy()
    atom_array, nan_mask = _handle_nan_coords(atom_array)

    chain_iids = np.unique(atom_array.chain_iid)
    if len(chain_iids) > 52:
        if raise_on_failure:
            raise ValueError("Too many chain_iids, cannot convert to PDB")
        yield None, None, None, None, None, True
        return

    # Build chain ID mapping (PDB format only supports single-character chain IDs)
    all_possible_chain_ids = string.ascii_letters
    chain_map = {}
    for chain_iid in chain_iids:
        if len(chain_iid) == 1:
            chain_map[chain_iid] = chain_iid
            all_possible_chain_ids = all_possible_chain_ids.replace(chain_iid, "")
    for chain_iid in chain_iids:
        if len(chain_iid) > 1:
            chain_map[chain_iid] = all_possible_chain_ids[0]
            all_possible_chain_ids = all_possible_chain_ids.replace(chain_map[chain_iid], "")

    atom_array.chain_id = [chain_map[iid] for iid in atom_array.chain_iid]
    inverted_chain_map = {v: k for k, v in chain_map.items()}
    atom_array.b_factor = np.zeros(len(atom_array))

    # Create temporary directory and PDB file
    temp_dir = tempfile.mkdtemp(prefix="hbplus_")
    pdb_filename = "structure.pdb"
    pdb_path = os.path.join(temp_dir, pdb_filename)

    try:
        with open(pdb_path, "w") as f:
            f.write(to_pdb_string(atom_array))

        yield temp_dir, pdb_filename, atom_array, nan_mask, inverted_chain_map, False
    finally:
        # Clean up entire temporary directory
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)


def check_atom_array_has_hydrogen(data: dict[str, Any]) -> None:
    if not np.any(data["atom_array"].element == "H"):
        raise ValueError("Key `atom_array` in data has no hydrogens.")


def _parse_hbplus_line(line: str, chain_map: dict[str, str]) -> dict[str, Any]:
    """Parse a single line from hbplus output (.hb2 file).

    HBplus output format (fixed-width columns):
    - Columns 0: donor chain
    - Columns 1-5: donor residue ID (with insertion code at position 5)
    - Columns 6-9: donor residue name
    - Columns 9-13: donor atom name
    - Columns 14: acceptor chain
    - Columns 15-19: acceptor residue ID (with insertion code at position 19)
    - Columns 20-23: acceptor residue name
    - Columns 23-27: acceptor atom name
    - Columns 27-32: distance
    """
    return {
        "d_chain": chain_map[line[0]],
        "d_resi": str(int(line[1:5].strip())),
        "d_resn": line[6:9].strip(),
        "d_ins": line[5].replace("-", " "),
        "d_atom": line[9:13].strip(),
        "a_chain": chain_map[line[14]],
        "a_resi": str(int(line[15:19].strip())),
        "a_resn": line[20:23].strip(),
        "a_ins": line[19].replace("-", " "),
        "a_atom": line[23:27].strip(),
        "dist": float(line[27:32].strip()),
    }


def _parse_hbplus_output(hb2_path: str, chain_map: dict[str, str]) -> list[dict[str, Any]]:
    """Parse hbplus output file and return list of hydrogen bonds.

    Args:
        hb2_path: Path to the .hb2 output file
        chain_map: Mapping from PDB chain IDs to original chain IIDs

    Returns:
        List of hydrogen bond dictionaries
    """
    if not os.path.exists(hb2_path):
        raise FileNotFoundError(f"HBplus output file not found: {hb2_path}")

    hbonds = []
    with open(hb2_path) as f:
        lines = f.readlines()
        # Skip header (first 8 lines)
        for line in lines[8:]:
            # Skip empty lines or lines that are too short
            if len(line.strip()) < 32:
                continue
            try:
                hbonds.append(_parse_hbplus_line(line, chain_map))
            except (ValueError, KeyError, IndexError):
                # Skip malformed lines but continue processing
                continue
    return hbonds


def _identify_atom(atom_array: AtomArray, chain_iid: str, res_id: str, atom_name: str) -> np.ndarray:
    """Create a mask identifying a specific atom in the array."""
    return (
        (atom_array.chain_iid == chain_iid) & (atom_array.res_id == float(res_id)) & (atom_array.atom_name == atom_name)
    )


@dataclass
class HBondResult:
    atom_array: AtomArray | None
    motif_hbonds: list | None
    all_hbonds: list | None
    num_motif_hbonds: int | None
    failed: bool


def calculate_hbonds(
    atom_array: AtomArray,
    motif_mask: np.ndarray,
    cutoff_HA_dist: float = 2.2,
    cutoff_DA_distance: float = 3.2,
    raise_on_failure: bool = True,
) -> HBondResult:
    """Calculate hydrogen bonds using the hbplus external tool.

    Args:
        atom_array: The atomic structure
        motif_mask: Boolean mask indicating which atoms are part of the motif
        cutoff_HA_dist: Hydrogen-acceptor distance cutoff in Angstroms
        cutoff_DA_distance: Donor-acceptor distance cutoff in Angstroms
        raise_on_failure: Whether to raise exceptions on failure

    Returns:
        HBondResult with atom_array, motif_hbonds, all_hbonds, num_motif_hbonds, and
        failed flag. If failed is True, all other fields are None.
    """
    with temporary_pdb_directory(atom_array, raise_on_failure=raise_on_failure) as result:
        if result[0] is None:
            return HBondResult(None, None, None, None, True)

        temp_dir, pdb_filename, atom_array, _nan_mask, inverted_chain_map, _failed = result
        _pdb_path = os.path.join(temp_dir, pdb_filename)

        # HBplus writes output to the current working directory
        # We need to run it from the temp directory
        original_dir = os.getcwd()
        try:
            os.chdir(temp_dir)

            hbplus_exe = os.environ.get("HBPLUS_PATH")
            if not hbplus_exe:
                if raise_on_failure:
                    raise OSError(
                        "Environment variable HBPLUS_PATH is not set! Please update your `.env` "
                        "file. On the DIGS, this should be: `/path/to/data`"
                    )
                return HBondResult(None, None, None, None, True)

            # Run hbplus (output will be created in temp_dir)
            subprocess.run(
                [
                    hbplus_exe,
                    "-h",
                    str(cutoff_HA_dist),
                    "-d",
                    str(cutoff_DA_distance),
                    pdb_filename,  # Use just the filename since we're in the temp dir
                    pdb_filename,
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,  # hbplus may return non-zero even on success
            )

            # HBplus creates output with .hb2 extension in current directory
            hb2_filename = pdb_filename.replace(".pdb", ".hb2")
            hb2_path = os.path.join(temp_dir, hb2_filename)

            # Parse hbplus output
            try:
                hbonds = _parse_hbplus_output(hb2_path, inverted_chain_map)
            except FileNotFoundError:
                if raise_on_failure:
                    raise
                return HBondResult(None, None, None, None, True)

        finally:
            # Restore original working directory
            os.chdir(original_dir)

    # Initialize arrays for tracking hydrogen bonds using enum
    hbond_all = np.full(len(atom_array), HBondEnum.NONE, dtype=int)
    hbond_between_motif_and_diffused = np.full(len(atom_array), HBondEnum.NONE, dtype=int)

    motif_hbonds = []
    for hbond in hbonds:
        # Identify donor and acceptor atoms
        current_donor_mask = _identify_atom(atom_array, hbond["d_chain"], hbond["d_resi"], hbond["d_atom"])
        current_acceptor_mask = _identify_atom(atom_array, hbond["a_chain"], hbond["a_resi"], hbond["a_atom"])

        # Ensure that we can uniquely identify the donor and acceptor atoms
        if current_donor_mask.sum() != 1:
            if raise_on_failure:
                raise ValueError(
                    f"Unable to uniquely identify a donor atom with chain_iid={hbond['d_chain']}, "
                    f"res_id={hbond['d_resi']}, atom_name={hbond['d_atom']}."
                )
            return HBondResult(None, None, None, None, True)
        if current_acceptor_mask.sum() != 1:
            if raise_on_failure:
                raise ValueError(
                    f"Unable to uniquely identify an acceptor atom with chain_iid={hbond['a_chain']}, "
                    f"res_id={hbond['a_resi']}, atom_name={hbond['a_atom']}."
                )
            return HBondResult(None, None, None, None, True)

        current_donor_is_motif = motif_mask[current_donor_mask][0]
        current_acceptor_is_motif = motif_mask[current_acceptor_mask][0]

        # Check for conflicts (atom marked as both donor and acceptor)
        # This shouldn't happen in valid data, but check for robustness
        donor_idx = np.where(current_donor_mask)[0][0]
        acceptor_idx = np.where(current_acceptor_mask)[0][0]

        if hbond_all[donor_idx] not in (HBondEnum.NONE, HBondEnum.DONOR):
            if raise_on_failure:
                raise ValueError(
                    f"Atom at index {donor_idx} is marked as both donor and acceptor. "
                    f"This indicates conflicting hydrogen bonds in the structure."
                )
            # Skip this hydrogen bond but continue processing
            continue

        if hbond_all[acceptor_idx] not in (HBondEnum.NONE, HBondEnum.ACCEPTOR):
            if raise_on_failure:
                raise ValueError(
                    f"Atom at index {acceptor_idx} is marked as both donor and acceptor. "
                    f"This indicates conflicting hydrogen bonds in the structure."
                )
            # Skip this hydrogen bond but continue processing
            continue

        # Record all ground-truth hbonds
        hbond_all[current_donor_mask] = HBondEnum.DONOR
        hbond_all[current_acceptor_mask] = HBondEnum.ACCEPTOR

        # Separately annotate hbonds between motif and diffused atoms
        if current_donor_is_motif != current_acceptor_is_motif:
            motif_hbonds.append(hbond)
            hbond_between_motif_and_diffused[current_donor_mask] = HBondEnum.DONOR
            hbond_between_motif_and_diffused[current_acceptor_mask] = HBondEnum.ACCEPTOR

    # Set annotations on the atom array
    atom_array.set_annotation("hbond_gt_between_motif_and_diffused", hbond_between_motif_and_diffused)
    atom_array.set_annotation("hbond_gt", hbond_all)

    return HBondResult(atom_array, motif_hbonds, hbonds, len(motif_hbonds), False)


def add_hbond_condition(
    data: dict,
    cutoff_HA_dist: float = 2.2,
    cutoff_DA_distance: float = 3.2,
    p_hbond_subsample: float = 0.5,
) -> dict:
    atom_array: AtomArray = as_atom_array_plus(data["atom_array"])

    # Get mask for valid atoms to receive hbond conditioning
    seq_mask = C_SEQ.mask(atom_array, default="generate")
    crd_mask = C_CRD.mask(atom_array, default="generate")
    intcrd_mask = C_INTCRD.mask(atom_array, default="generate")
    dis_mask = C_DIS.mask(atom_array, default="generate").as_dense_array(False).any(axis=0)
    motif_mask = seq_mask | crd_mask | intcrd_mask | dis_mask

    hbond_result = calculate_hbonds(
        atom_array,
        motif_mask=motif_mask,
        cutoff_HA_dist=cutoff_HA_dist,
        cutoff_DA_distance=cutoff_DA_distance,
        raise_on_failure=False,
    )

    data.setdefault("log_dict", {})
    log_dict = data["log_dict"]

    # NOTE: If the calculation failed, log it but then behave as if we had skipped this modifier.
    # AddIsXFeats will add the default feature (zeros).
    if hbond_result.failed:
        log_dict["hbond_calculation_failed"] = True
        return data

    log_dict["hbond_calculation_failed"] = False
    atom_array = hbond_result.atom_array

    # Get hydrogen bond annotations from calculate_hbonds
    hbond_gt = atom_array.hbond_gt_between_motif_and_diffused.copy()

    # Mask to only include motif atoms
    hbond_gt[~motif_mask] = HBondEnum.NONE

    # Count total hydrogen bonds
    n_hbonds = np.sum(hbond_gt != HBondEnum.NONE)
    log_dict["hbond_total_count"] = n_hbonds

    # Store the motif-masked ground truth for reference
    atom_array.set_annotation("motif_hbond_gt_between_motif_and_diffused", hbond_gt)

    # Subsample hydrogen bonds if needed
    should_subsample = np.random.rand() < p_hbond_subsample
    if should_subsample and n_hbonds > 3:
        base_fraction = 0.1
        max_fraction = 0.9
        max_hbonds = 50

        fraction = max_fraction - (max_fraction - base_fraction) * min(n_hbonds / max_hbonds, 1.0)
        hbond_gt = subsample_hbond_enum(hbond_gt, fraction)

    # Set the final hydrogen bond condition
    C_HB.set_annotation(atom_array, hbond_gt)
    log_dict["hbond_subsample_atoms"] = np.sum(hbond_gt != HBondEnum.NONE)

    data["log_dict"] = log_dict
    data["atom_array"] = atom_array

    return data


def subsample_hbond_enum(hbond_array: np.ndarray, fraction: float) -> np.ndarray:
    """Subsample hydrogen bond annotations by randomly keeping a fraction of them.

    Args:
        hbond_array: Array with HBondEnum values
        fraction: Fraction of hydrogen bonds to keep (0 < fraction <= 1)

    Returns:
        New array with subsampled hydrogen bonds
    """
    if not (0 < fraction <= 1):
        raise ValueError("Fraction must be in the range (0, 1].")

    hbond_array = hbond_array.copy()

    # Find all atoms that have hydrogen bonds (donor or acceptor)
    hbond_indices = np.where(hbond_array != HBondEnum.NONE)[0]
    num_hbonds = len(hbond_indices)
    keep_count = int(num_hbonds * fraction)

    # Randomly select which hydrogen bonds to keep
    np.random.shuffle(hbond_indices)
    remove_indices = hbond_indices[keep_count:]

    # Set removed hydrogen bonds to NONE
    hbond_array[remove_indices] = HBondEnum.NONE

    return hbond_array


def add_secondary_structure_condition(
    data: dict[str, Any],
    max_secondary_structure_frac_to_provide: float = 0.4,
    min_ss_island_len: int = 1,
    max_ss_island_len: int = 10,
    n_islands_max: int = 3,
) -> dict[str, Any]:
    """Add secondary structure conditioning.

    Uses P-SEA to annotate secondary structure, samples islands from residues
    with valid SSE annotations, and sets C_SEC condition. Sampling is done at
    residue granularity (not token granularity) so that residues spanning
    multiple tokens — e.g., atomized PTM residues — receive a uniform value, satisfying the
    ``Level.RESIDUE`` invariant on the secondary-structure condition.

    Args:
      data: Data dict containing atom_array.
      max_secondary_structure_frac_to_provide: Max fraction of valid SS residues
        to condition. Defaults to 0.4.
      min_ss_island_len: Minimum island length (in residues). Defaults to 1.
      max_ss_island_len: Maximum island length (in residues). Defaults to 10.
      n_islands_max: Maximum number of islands. Defaults to 3.

    Returns:
      Updated data dict with C_SEC conditioning.
    """
    atom_array = as_atom_array_plus(data["atom_array"])

    # Step 1: Annotate with P-SEA (residue-uniform sse / sse_is_valid)
    atom_array = annotate_secondary_structure(
        atom_array, annotation_name="sse", is_valid_annotation_name="sse_is_valid"
    )

    # Step 2: Sample islands from eligible residues
    residue_starts = get_residue_starts(atom_array)
    residue_level_array = atom_array[residue_starts]

    # Eligible: residues with valid SSE annotations
    eligible_for_ss_info_mask = residue_level_array.sse_is_valid

    n_residues_with_valid_ss = eligible_for_ss_info_mask.sum()
    max_residues_with_ss_conditioning = int(np.ceil(n_residues_with_valid_ss * max_secondary_structure_frac_to_provide))

    where_to_show_ss = sample_island_tokens(
        n_residues_with_valid_ss,
        island_len_min=min_ss_island_len,
        island_len_max=max_ss_island_len,
        n_islands_min=1,
        n_islands_max=n_islands_max,
        max_length=max_residues_with_ss_conditioning,
    )

    # Convert to residue-level mask and spread to atoms
    residue_level_ss_mask = np.zeros(len(residue_starts), dtype=bool)
    residue_level_ss_mask[eligible_for_ss_info_mask] = where_to_show_ss
    ss_mask = spread_residue_wise(atom_array, residue_level_ss_mask)

    # Step 3: Set C_SEC annotation (SSEnum.NONE for unspecified, sse values where sampled)
    ss_annotation = np.full(atom_array.array_length(), 0, dtype=int)  # 0 is SSEnum.NONE
    ss_annotation[ss_mask] = atom_array.sse[ss_mask]
    C_SEC.set_annotation(atom_array, ss_annotation)

    data["atom_array"] = atom_array
    return data


def add_cath_condition(
    data: dict[str, Any],
    query_pn_unit_iid: str | None = None,
    p_add_cath_c_condition: float = 0.0,
    p_add_cath_ca_condition: float = 0.0,
    p_add_cath_cat_condition: float = 0.0,
) -> dict[str, Any]:
    """Add CATH annotation to the AtomArray.

    Args:
      data: Data dict containing atom_array.
      query_pn_unit_iid: Chain ID to use for CATH annotation. Defaults to None, which will skip annotation
        unless this is a single-chain structure.
      p_add_cath_c_condition: Probability of adding CATH conditioning at C level only.
      p_add_cath_ca_condition: Probability of adding CATH conditioning at C and A levels only.
      p_add_cath_cat_condition: Probability of adding CATH conditioning at C, A, and T levels.
    """

    sampled_value = np.random.rand()

    p_skip = 1 - (p_add_cath_c_condition + p_add_cath_ca_condition + p_add_cath_cat_condition)
    if sampled_value < p_skip:
        return data

    if "extra_info" not in data:
        return data

    extra_info = data["extra_info"]
    atom_array = data["atom_array"]

    if query_pn_unit_iid is None:
        # Only monomers allowed in this case
        if len(np.unique(atom_array.chain_iid)) != 1:
            return data

        query_pn_unit_iid = atom_array.chain_iid[0]
        cath_code_key = "cath_code"
    else:
        if query_pn_unit_iid not in np.unique(atom_array.chain_iid):
            return data

        if extra_info.get("pn_unit_1_iid") == query_pn_unit_iid:
            cath_code_key = "pn_unit_1_cath_code"
        elif extra_info.get("pn_unit_2_iid") == query_pn_unit_iid:
            cath_code_key = "pn_unit_2_cath_code"
        else:
            return data

    if cath_code_key not in extra_info:
        return data

    if pd.isna(extra_info[cath_code_key]):
        return data

    cath_code = ast.literal_eval(extra_info[cath_code_key])

    # Sample only one cath annotation
    if isinstance(cath_code, list):
        cath_code = np.random.choice(cath_code)

    cath_code_split = cath_code.split(".")

    if len(cath_code_split) != 4:
        raise ValueError(f"Incomplete or invalid CATH code: {cath_code}")

    if sampled_value < p_skip + p_add_cath_c_condition:
        cath_code_split[1] = "x"
        cath_code_split[2] = "x"
        cath_code_split[3] = "x"
        data["task"]["added_cath_c_condition"] = True
    elif sampled_value < p_skip + p_add_cath_c_condition + p_add_cath_ca_condition:
        cath_code_split[2] = "x"
        cath_code_split[3] = "x"
        data["task"]["added_cath_ca_condition"] = True
    else:
        cath_code_split[3] = "x"
        data["task"]["added_cath_cat_condition"] = True

    cath_code_output = ".".join(cath_code_split)

    # Apply only to the specified chain
    cath_annot = C_CATH.default_annotation(atom_array)
    cath_annot[atom_array.chain_iid == query_pn_unit_iid] = cath_code_output
    C_CATH.set_annotation(atom_array, cath_annot)

    return data
