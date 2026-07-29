from typing import Any

import numpy as np
from atomworks.ml.utils.token import get_token_starts
from biotite.structure import AtomArray
from pydantic import BaseModel, ConfigDict, model_validator

from rfproteina.inference_utils.components import ComponentValidationError, extract_pn_unit_info, split_contig


class NucleicSSInputSpec(BaseModel):
    """User-facing nucleic secondary-structure input syntax.

    This schema captures the JSON/YAML forms accepted at inference time before
    they are resolved onto the constructed token layout.

    Supported keys (all optional):

    * ``ss_dbn``: global dot-bracket string applied to the first ``L`` tokens.
    * ``ss_dbn_dict``: mapping of ``"<chain><start>-<end>"`` to a dot-bracket string.
    * ``paired_region_list``: list like ``["A5-15,B1-11", ...]``.
    * ``paired_position_list``: list like ``["A19,A61,A20", ...]``.
    * ``loop_region_list``: list like ``["A5-10", ...]`` for explicitly unpaired regions.
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    ss_dbn: str | None = None
    ss_dbn_dict: dict[str, str] | None = None
    paired_region_list: list[str] | None = None
    paired_position_list: list[str] | None = None
    loop_region_list: list[str] | None = None

    @model_validator(mode="before")
    @classmethod
    def canonicalize_lists(cls, data: Any) -> Any:
        """Normalize flexible user-facing list fields before validation.

        Args:
            data: Raw value provided for ``nucleic_ss``.

        Returns:
            Normalized mapping suitable for model validation.
        """
        if data is None or isinstance(data, cls):
            return data
        if not isinstance(data, dict):
            raise TypeError(f"nucleic_ss must be a mapping, got {type(data)}")

        normalized = dict(data)
        for key in ("paired_region_list", "paired_position_list", "loop_region_list"):
            value = normalized.get(key)
            if value is None:
                continue
            if isinstance(value, str):
                normalized[key] = [value]
            elif isinstance(value, list):
                normalized[key] = [str(item).strip() for item in value if str(item).strip()]
            else:
                raise TypeError(f"nucleic_ss.{key} must be a string or list[str], got {type(value)}")

        ss_dbn_dict = normalized.get("ss_dbn_dict")
        if ss_dbn_dict is not None:
            if not isinstance(ss_dbn_dict, dict):
                raise TypeError(f"nucleic_ss.ss_dbn_dict must be a mapping, got {type(ss_dbn_dict)}")
            normalized["ss_dbn_dict"] = {
                str(k).strip(): str(v).strip() for k, v in ss_dbn_dict.items() if str(k).strip() and str(v).strip()
            }
        return normalized


def _parse_na_ss_region(region_str: str) -> tuple[str, int, int] | None:
    """Parse a ``<chain><start>-<end>`` region string.

    Args:
        region_str: Region string such as ``"A5-10"``.

    Returns:
        Tuple ``(chain_id, start_res, end_res)`` if parsing succeeds, otherwise
        ``None``.
    """
    region_str = str(region_str).strip()
    if not region_str:
        return None
    try:
        chain_id, start_res, end_res = extract_pn_unit_info(region_str)
    except ComponentValidationError:
        return None
    if start_res > end_res:
        start_res, end_res = end_res, start_res
    return chain_id, start_res, end_res


def _parse_na_ss_position(pos_str: str) -> tuple[str, int] | None:
    """Parse a ``<chain><res_id>`` position string.

    Args:
        pos_str: Position string such as ``"A19"``.

    Returns:
        Tuple ``(chain_id, res_id)`` if parsing succeeds, otherwise ``None``.
    """
    pos_str = str(pos_str).strip()
    if not pos_str:
        return None
    try:
        chain_id, res_id = split_contig(pos_str)
    except ComponentValidationError:
        return None
    return str(chain_id), int(res_id)


#: Bracket families whose open/close roles are fixed by shape.
_SYMBOL_PAIRS = {"(": ")", "[": "]", "{": "}", "<": ">"}


def _opener_for(dot_bracket: str) -> dict[str, str]:
    """Map each closing character to its opener, resolving letter-case per string.

    Letter families (``Aa``, ``Bb``, ...) encode higher-order pseudoknots, but the
    case convention is not universal: Vienna/bpRNA write ``A...a`` (uppercase opens)
    while Eterna/OpenKnot write ``a...A`` (lowercase opens). Both are well-formed, so
    the opener is taken to be whichever case appears first for that letter.

    Args:
        dot_bracket: Dot-bracket notation string.

    Returns:
        Mapping of closing character to its opening character.
    """
    opener_for = {close: open_ for open_, close in _SYMBOL_PAIRS.items()}
    for lower in {c.lower() for c in dot_bracket if c.isalpha()}:
        upper = lower.upper()
        i_lower, i_upper = dot_bracket.find(lower), dot_bracket.find(upper)
        # A missing case leaves an unbalanced string; either assignment surfaces it below.
        if i_upper >= 0 and (i_lower < 0 or i_upper < i_lower):
            opener_for[lower] = upper
        else:
            opener_for[upper] = lower
    return opener_for


def parse_dot_bracket(dot_bracket: str) -> tuple[list[tuple[int, int]], list[int]]:
    """Parse a dot-bracket string into base pairs and unpaired positions.

    Supports standard ``()``, ``[]``, ``{}``, ``<>`` plus letter families
    (``Aa``, ``Bb``, ...) for higher-order pseudoknots, in either case convention
    (see :py:func:`_opener_for`).

    Args:
        dot_bracket: Dot-bracket notation string.

    Returns:
        Tuple of ``(pairs, unpaired)`` where *pairs* is a list of 0-based
        ``(i, j)`` index tuples and *unpaired* is a list of 0-based indices
        corresponding to ``.`` characters.

    Raises:
        ValueError: If a bracket is unmatched or an unrecognized character appears.
            Malformed conditioning is never dropped silently — an ignored bracket
            would quietly weaken the requested secondary structure.

    Examples:
        >>> parse_dot_bracket("((..))")[0]
        [(1, 4), (0, 5)]
        >>> parse_dot_bracket("a.A")[0]  # lowercase opens (Eterna/OpenKnot)
        [(0, 2)]
        >>> parse_dot_bracket("A.a")[0]  # uppercase opens (Vienna/bpRNA)
        [(0, 2)]
    """
    dot_bracket = str(dot_bracket)
    opener_for = _opener_for(dot_bracket)
    openers = set(opener_for.values())

    stack: dict[str, list[int]] = {}
    pairs: list[tuple[int, int]] = []
    unpaired: list[int] = []

    for i, ch in enumerate(dot_bracket):
        if ch == ".":
            unpaired.append(i)
        elif ch in openers:
            stack.setdefault(ch, []).append(i)
        elif ch in opener_for:
            open_ = opener_for[ch]
            if not stack.get(open_):
                raise ValueError(f"unmatched closing bracket {ch!r} at position {i} in dot-bracket string")
            pairs.append((stack[open_].pop(), i))
        else:
            raise ValueError(f"unrecognized dot-bracket character {ch!r} at position {i}")

    dangling = {open_: idxs for open_, idxs in stack.items() if idxs}
    if dangling:
        detail = ", ".join(f"{open_!r} at {idxs}" for open_, idxs in sorted(dangling.items()))
        raise ValueError(f"unmatched opening bracket(s) in dot-bracket string: {detail}")
    return pairs, unpaired


def resolve_nucleic_ss_to_token_payload(
    atom_array: AtomArray,
    nucleic_ss: NucleicSSInputSpec | None,
) -> dict[str, list[Any]] | None:
    """Resolve user-facing NA-SS syntax against the built token layout.

    Args:
        atom_array: Fully assembled inference AtomArray whose token layout defines
            the meaning of user-provided chain/residue references.
        nucleic_ss: Validated NA-SS input specification.

    Returns:
        Syntax-agnostic token-level payload with keys:
        ``pair_token_indices``, ``loop_token_indices``, and
        ``specified_token_indices``. Returns ``None`` when no NA-SS
        specification is present.
    """
    if nucleic_ss is None:
        return None

    token_starts = get_token_starts(atom_array)
    n_tokens = len(token_starts)
    if n_tokens == 0:
        return {
            "pair_token_indices": [],
            "loop_token_indices": [],
            "specified_token_indices": [],
        }

    if hasattr(atom_array, "chain_iid"):
        chain_iid_list = [str(v) for v in np.asarray(atom_array.chain_iid)[token_starts].tolist()]
    else:
        chain_iid_list = [str(v) for v in np.asarray(atom_array.chain_id)[token_starts].tolist()]
    chain_id_list = [str(v) for v in np.asarray(atom_array.chain_id)[token_starts].tolist()]
    resi_list = [int(v) for v in np.asarray(atom_array.res_id)[token_starts].tolist()]
    chain_res_to_tok: dict[tuple[str, int], int] = {}
    for i, (chain_iid, chain_id, res_id) in enumerate(zip(chain_iid_list, chain_id_list, resi_list, strict=False)):
        chain_res_to_tok.setdefault((chain_iid, int(res_id)), int(i))
        chain_res_to_tok.setdefault((chain_id, int(res_id)), int(i))
        chain_res_to_tok.setdefault((chain_iid.split("_", 1)[0], int(res_id)), int(i))

    def _region_to_token_indices(region_str: str) -> list[int]:
        parsed = _parse_na_ss_region(region_str)
        if parsed is None:
            raise ValueError(f"Invalid nucleic_ss region selection: {region_str!r}")
        chain_id, start_res, end_res = parsed
        token_indices: list[int] = []
        for res_id in range(start_res, end_res + 1):
            idx = chain_res_to_tok.get((chain_id, int(res_id)))
            if idx is None:
                raise ValueError(f"nucleic_ss region {region_str!r} references missing residue {chain_id}{int(res_id)}")
            token_indices.append(int(idx))
        return token_indices

    def _pos_to_token_index(pos_str: str) -> int | None:
        parsed = _parse_na_ss_position(pos_str)
        if parsed is None:
            raise ValueError(f"Invalid nucleic_ss position selection: {pos_str!r}")
        chain_id, res_id = parsed
        tok = chain_res_to_tok.get((chain_id, int(res_id)))
        if tok is None:
            raise ValueError(f"nucleic_ss position {pos_str!r} does not exist in the assembled token layout")
        return tok

    partners: list[set[int]] = [set() for _ in range(n_tokens)]
    loop_token_idxs: set[int] = set()
    specified_token_idxs: set[int] = set()

    def _add_pair(i: int, j: int) -> None:
        if not (0 <= i < n_tokens and 0 <= j < n_tokens):
            raise ValueError(f"nucleic_ss pair index out of range: ({i}, {j}) for {n_tokens} tokens")
        if i == j:
            raise ValueError(f"nucleic_ss cannot pair a token with itself: index {i}")
        if i in loop_token_idxs or j in loop_token_idxs:
            raise ValueError(f"nucleic_ss token conflict: cannot pair loop token(s) {i}, {j}")
        partners[i].add(j)
        partners[j].add(i)

    if isinstance(nucleic_ss.ss_dbn, str) and nucleic_ss.ss_dbn.strip():
        ss_dbn = nucleic_ss.ss_dbn.strip()
        if len(ss_dbn) > n_tokens:
            raise ValueError(
                f"nucleic_ss.ss_dbn has length {len(ss_dbn)} but the assembled input has only {n_tokens} tokens"
            )
        pairs, unpaired = parse_dot_bracket(ss_dbn)
        L = min(len(ss_dbn), n_tokens)
        specified_token_idxs.update(range(L))
        for i_local, j_local in pairs:
            if 0 <= i_local < L and 0 <= j_local < L:
                _add_pair(int(i_local), int(j_local))
        for i_local in unpaired:
            if 0 <= int(i_local) < L:
                loop_token_idxs.add(int(i_local))

    if nucleic_ss.ss_dbn_dict:
        for region_str, dbn_str in nucleic_ss.ss_dbn_dict.items():
            toks = _region_to_token_indices(region_str)
            if len(toks) != len(dbn_str):
                raise ValueError(
                    f"nucleic_ss.ss_dbn_dict entry {region_str!r} spans {len(toks)} tokens but DBN has length "
                    f"{len(dbn_str)}"
                )
            specified_token_idxs.update(int(tok) for tok in toks)
            pairs, unpaired = parse_dot_bracket(dbn_str)
            for i_local, j_local in pairs:
                if 0 <= i_local < len(toks) and 0 <= j_local < len(toks):
                    _add_pair(int(toks[int(i_local)]), int(toks[int(j_local)]))
            for i_local in unpaired:
                if 0 <= i_local < len(toks):
                    loop_token_idxs.add(int(toks[int(i_local)]))

    for region_entry in nucleic_ss.paired_region_list or []:
        region_parts = [p.strip() for p in region_entry.split(",") if p.strip()]
        if len(region_parts) != 2:
            raise ValueError(
                "Entries in nucleic_ss.paired_region_list must contain exactly two comma-separated regions, "
                f"got {region_entry!r}"
            )
        toks1 = _region_to_token_indices(region_parts[0])
        toks2 = _region_to_token_indices(region_parts[1])
        specified_token_idxs.update(int(tok) for tok in toks1)
        specified_token_idxs.update(int(tok) for tok in toks2)
        for ti in toks1:
            for tj in toks2:
                _add_pair(int(ti), int(tj))

    for group_str in nucleic_ss.paired_position_list or []:
        pos_parts = [p.strip() for p in group_str.split(",") if p.strip()]
        if len(pos_parts) < 2:
            raise ValueError(
                "Entries in nucleic_ss.paired_position_list must contain at least two comma-separated positions, "
                f"got {group_str!r}"
            )
        tok_indices: list[int] = []
        for pos_str in pos_parts:
            tok = _pos_to_token_index(pos_str)
            tok_indices.append(int(tok))
        specified_token_idxs.update(tok_indices)
        for i in range(len(tok_indices)):
            for j in range(i + 1, len(tok_indices)):
                _add_pair(tok_indices[i], tok_indices[j])

    for region_str in nucleic_ss.loop_region_list or []:
        toks = _region_to_token_indices(region_str)
        specified_token_idxs.update(int(tok) for tok in toks)
        for tok in toks:
            loop_token_idxs.add(int(tok))

    for i in list(loop_token_idxs):
        if not (0 <= i < n_tokens):
            raise ValueError(f"nucleic_ss loop index out of range: {i} for {n_tokens} tokens")
        for j in list(partners[i]):
            partners[j].discard(i)
        partners[i].clear()

    pair_token_indices = sorted((int(i), int(j)) for i in range(n_tokens) for j in sorted(partners[i]))
    return {
        "pair_token_indices": pair_token_indices,
        "loop_token_indices": sorted(int(i) for i in loop_token_idxs),
        "specified_token_indices": sorted(int(i) for i in specified_token_idxs),
    }
