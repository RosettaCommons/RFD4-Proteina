"""Atom mask helpers shared by distance and TMol block-pair metrics."""

from __future__ import annotations

import ast
from typing import Any

import numpy as np
from atomworks.io.utils.query import QueryExpression
from atomworks.ml.conditions.annotator import ANNOTATOR_REGISTRY, ensure_annotations
from biotite.structure import AtomArray

from foundry.utils.standard_annotations import SA_SRC


def _as_array(values: Any, dtype: Any) -> np.ndarray:
    arr = np.asarray(values, dtype=dtype)
    return arr.reshape(-1)


def extract_annotation_names(query: str) -> list[str]:
    """Annotation names in a QueryExpression that have registered annotators."""
    qe = QueryExpression(query)
    names: list[str] = []
    for node in ast.walk(qe.tree):
        if isinstance(node, ast.Name) and node.id in ANNOTATOR_REGISTRY:
            names.append(node.id)
    return names


def atom_selection_from_spec(atom_array: AtomArray, spec: dict[str, Any]) -> np.ndarray:
    """Build a boolean atom mask from a region specification dict.

    Supports an optional ``query`` (AtomWorks QueryExpression). When present, the query
    mask is AND-ed with any field filters. Each field filter (``chain_id``, ``res_name``,
    ``res_id``, ``atom_name``) accepts a scalar or a list of allowed values.

    Example::

        {"query": "is_protein & (chain_id == 'A')"}
        {"res_name": "lpr"}
        {"query": "mask_coordinate_1_atom", "res_name": "HIS", "atom_name": "ND1"}
    """
    spec = dict(spec)
    query = spec.pop("query", None)

    if query is not None:
        ensure_annotations(atom_array, *extract_annotation_names(query))
        sel = atom_array.mask(query)
    else:
        sel = np.ones(atom_array.array_length(), dtype=bool)

    if "src_component" in spec:
        if not SA_SRC.has_annotation(atom_array):
            raise ValueError(f"Atom spec requires {SA_SRC.full_name} but structure lacks it: {spec}")
        sel &= SA_SRC.annotation(atom_array) == spec["src_component"]

    if "chain_id" in spec:
        sel &= np.isin(atom_array.chain_id, _as_array(spec["chain_id"], atom_array.chain_id.dtype))
    if "res_name" in spec:
        sel &= np.isin(atom_array.res_name, _as_array(spec["res_name"], atom_array.res_name.dtype))
    if "res_id" in spec and spec["res_id"] is not None:
        sel &= np.isin(atom_array.res_id, _as_array(spec["res_id"], np.int32))
    if "atom_name" in spec:
        sel &= np.isin(atom_array.atom_name, _as_array(spec["atom_name"], atom_array.atom_name.dtype))

    return sel
