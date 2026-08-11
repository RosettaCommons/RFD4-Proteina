import os
import shutil
import string
import subprocess
import tempfile
from contextlib import contextmanager
from typing import Any

import numpy as np
from atomworks.io.utils.io_utils import to_pdb_string
from biotite.structure import AtomArray, BadStructureError


@contextmanager
def temporary_pdb_directory(atom_array: AtomArray, raise_on_failure: bool = True) -> Any:
    """Create a temporary directory containing a PDB version of an atom array.

    Args:
        atom_array: Structure to serialize for HBPLUS.
        raise_on_failure: Whether to raise if the structure cannot be represented
            in single-character PDB chain IDs.

    Yields:
        Tuple ``(temp_dir, pdb_filename, atom_array, nan_mask, inverted_chain_map, failed)``.
        If setup fails and ``raise_on_failure`` is ``False``, yields
        ``(None, None, None, None, None, True)``.
    """

    def _handle_nan_coords(atom_array: AtomArray, noise_level: float = 1e-3) -> tuple:
        coords = atom_array.coord
        nan_mask = np.isnan(coords)
        coords[nan_mask] = np.random.uniform(-noise_level, noise_level, size=nan_mask.sum())
        atom_array.coord = coords
        return atom_array, nan_mask

    atom_array = atom_array.copy()
    atom_array, nan_mask = _handle_nan_coords(atom_array)

    chain_iids = np.unique(atom_array.chain_iid)
    if len(chain_iids) > 52:
        if raise_on_failure:
            raise ValueError("Too many chain_iids, cannot convert to PDB")
        yield None, None, None, None, None, True
        return

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

    temp_dir = tempfile.mkdtemp(prefix="hbplus_")
    pdb_filename = "structure.pdb"
    pdb_path = os.path.join(temp_dir, pdb_filename)

    try:
        try:
            pdb_string = to_pdb_string(atom_array)
        except BadStructureError:
            # PDB cannot hold e.g. 5-character CCD codes; drop the condition, not the example.
            if raise_on_failure:
                raise
            yield None, None, None, None, None, True
            return

        with open(pdb_path, "w") as f:
            f.write(pdb_string)

        yield temp_dir, pdb_filename, atom_array, nan_mask, inverted_chain_map, False
    finally:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)


def parse_hbplus_line(line: str, chain_map: dict[str, str]) -> dict[str, Any]:
    """Parse a single fixed-width HBPLUS record.

    Args:
        line: Raw line from an ``.hb2`` file.
        chain_map: Mapping from temporary PDB chain IDs back to original
            ``chain_iid`` values.

    Returns:
        Dictionary describing one donor/acceptor pair.
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


def parse_hbplus_output(hb2_path: str, chain_map: dict[str, str]) -> list[dict[str, Any]]:
    """Parse an HBPLUS ``.hb2`` file.

    Args:
        hb2_path: Path to the output file.
        chain_map: Mapping from temporary PDB chain IDs back to original
            ``chain_iid`` values.

    Returns:
        List of parsed H-bond records.
    """
    if not os.path.exists(hb2_path):
        raise FileNotFoundError(f"HBplus output file not found: {hb2_path}")

    hbonds = []
    with open(hb2_path) as f:
        for line in f.readlines()[8:]:
            if len(line.strip()) < 32:
                continue
            try:
                hbonds.append(parse_hbplus_line(line, chain_map))
            except (ValueError, KeyError, IndexError):
                continue
    return hbonds


def run_hbplus(
    atom_array: AtomArray,
    *,
    cutoff_HA_dist: float,
    cutoff_DA_dist: float,
    raise_on_failure: bool,
    default_hbplus_path: str | None = None,
) -> tuple[AtomArray | None, list[dict[str, Any]] | None, bool]:
    """Run HBPLUS and return parsed bond records.

    This is the shared low-level execution layer used by higher-level H-bond
    consumers. It handles temporary PDB creation, chain-ID remapping, HBPLUS
    invocation, and ``.hb2`` parsing, but does not impose any interpretation of
    the resulting bond list.

    Args:
        atom_array: Structure to analyse.
        cutoff_HA_dist: Hydrogen-acceptor distance cutoff in Angstroms.
        cutoff_DA_dist: Donor-acceptor distance cutoff in Angstroms.
        raise_on_failure: Whether missing configuration or missing output should
            raise rather than returning ``failed=True``.
        default_hbplus_path: Optional fallback HBPLUS executable path when
            ``HBPLUS_PATH`` is unset.

    Returns:
        Tuple ``(prepared_atom_array, hbonds, failed)``.
    """
    with temporary_pdb_directory(atom_array, raise_on_failure=raise_on_failure) as result:
        if result[0] is None:
            return None, None, True

        temp_dir, pdb_filename, prepared_atom_array, _nan_mask, inverted_chain_map, _failed = result
        original_dir = os.getcwd()
        try:
            os.chdir(temp_dir)

            hbplus_exe = os.environ.get("HBPLUS_PATH", default_hbplus_path)
            if not hbplus_exe:
                if raise_on_failure:
                    raise OSError(
                        "Environment variable HBPLUS_PATH is not set! Please update your `.env` "
                        "file. On the DIGS, this should be: `/path/to/data`"
                    )
                return None, None, True

            subprocess.run(
                [
                    hbplus_exe,
                    "-h",
                    str(cutoff_HA_dist),
                    "-d",
                    str(cutoff_DA_dist),
                    pdb_filename,
                    pdb_filename,
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )

            hb2_path = os.path.join(temp_dir, pdb_filename.replace(".pdb", ".hb2"))
            try:
                hbonds = parse_hbplus_output(hb2_path, inverted_chain_map)
            except FileNotFoundError:
                if raise_on_failure:
                    raise
                return None, None, True
        finally:
            os.chdir(original_dir)

    return prepared_atom_array, hbonds, False
