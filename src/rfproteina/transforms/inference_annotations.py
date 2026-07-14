"""
Annotate source-component provenance for inference inputs, mimicing behavior for
json-based input parsing pipeline. Traditional json-based input parsing code adds
annotations to the output atom array to indicate which parts of the input each part
of the output came from. These are contig labels like "A3-10", "B10", or "15-20".
They are meant for readability aids for users when trying to determine how the
output structure was constructed.
"""

import biotite.structure as struc
import numpy as np
from atomworks.io.utils.standard_annotations import S_SEGMAX, S_SEGMIN
from atomworks.ml.transforms.base import Transform
from biotite.structure import AtomArray

from rfproteina.conditions import C_IDXGRP
from rfproteina.utils.standard_annotations import SA_SRC


def build_src_component_annotation(atom_array: AtomArray) -> np.ndarray:
    """Compute per-atom :py:data:`~rfproteina.utils.standard_annotations.SA_SRC` contigs.

    Non-sentinel residues that share :py:data:`~rfproteina.conditions.C_IDXGRP`
    and have contiguous ``res_id`` on the same chain are merged into a single
    contig label (e.g. ``"A3-10"`` or ``"B10"`` for a singleton). Expandable-segment
    sentinels are labeled ``"{seg_min}-{seg_max}"``.

    Returns:
      String array of shape ``(n_atoms,)`` with dtype matching ``SA_SRC``.
    """
    res_starts = struc.get_residue_starts(atom_array, add_exclusive_stop=True)
    starts, stops = res_starts[:-1], res_starts[1:]
    n_res = len(starts)

    index_group = C_IDXGRP.annotation(atom_array, default="generate")
    is_expandable_segment_atom = S_SEGMIN.mask(atom_array)

    chain_id = atom_array.chain_id[starts]
    res_id = atom_array.res_id[starts].astype(int)
    res_idx_group = index_group[starts]
    is_expandable_segment = is_expandable_segment_atom[starts]
    seg_min = S_SEGMIN.annotation(atom_array)[starts].astype(int)
    seg_max = S_SEGMAX.annotation(atom_array)[starts].astype(int)

    src = np.empty(atom_array.array_length(), dtype=SA_SRC.dtype)

    i = 0
    while i < n_res:
        if is_expandable_segment[i]:
            src[starts[i] : stops[i]] = f"{seg_min[i]}-{seg_max[i]}"
            i += 1
            continue

        # Extend a run of residues that share (chain, index_group) with contiguous res_id.
        j = i
        while (
            j + 1 < n_res
            and not is_expandable_segment[j + 1]
            and chain_id[j + 1] == chain_id[i]
            and res_idx_group[j + 1] == res_idx_group[i]
            and res_id[j + 1] == res_id[j] + 1
        ):
            j += 1

        start_rid, end_rid = res_id[i], res_id[j]
        label = f"{chain_id[i]}{start_rid}" if start_rid == end_rid else f"{chain_id[i]}{start_rid}-{end_rid}"
        src[starts[i] : stops[j]] = label
        i = j + 1

    return src


class AddSrcComponentAnnotation(Transform):
    """Annotate :py:data:`~rfproteina.utils.standard_annotations.SA_SRC` contig strings.

    Must run **before** :py:class:`~atomworks.ml.transforms.realize_expandable_segments.RealizeExpandableSegments`
    so that original ``res_id`` values are preserved on non-sentinel residues.
    """

    def forward(self, data: dict) -> dict:
        atom_array = data["atom_array"]
        SA_SRC.set_annotation(atom_array, build_src_component_annotation(atom_array))
        data["atom_array"] = atom_array
        return data
