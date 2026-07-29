"""Dot-bracket parsing across letter-bracket case conventions."""

import pytest

from rfproteina.inference_utils.nucleic_ss_input import parse_dot_bracket


@pytest.mark.parametrize(
    "dbn,expected_pairs",
    [
        ("((..))", [(0, 5), (1, 4)]),  # nested
        ("(([[))]]", [(0, 5), (1, 4), (2, 7), (3, 6)]),  # crossing symbol families
        ("A..a", [(0, 3)]),  # uppercase opens (Vienna/bpRNA)
        ("a..A", [(0, 3)]),  # lowercase opens (Eterna/OpenKnot)
        ("(a.).A", [(0, 3), (1, 5)]),  # letter family crossing a symbol family
        ("((.)", None),  # unmatched opener
        ("(.))", None),  # unmatched closer
        ("a..a", None),  # same case twice cannot pair
        ("((.-.))", None),  # unrecognized character
    ],
)
def test_parse_dot_bracket(dbn: str, expected_pairs: list[tuple[int, int]] | None):
    """Both case conventions parse; malformed input raises rather than dropping pairs."""
    if expected_pairs is None:
        with pytest.raises(ValueError):
            parse_dot_bracket(dbn)
        return

    pairs, unpaired = parse_dot_bracket(dbn)
    assert sorted(pairs) == expected_pairs
    # Nothing silently dropped: every position is either paired or marked unpaired.
    assert set(unpaired) | {i for pair in pairs for i in pair} == set(range(len(dbn)))
