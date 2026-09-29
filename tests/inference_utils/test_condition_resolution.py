"""Tests for resolving an arbitrary Condition/StandardAnnotation name and value from a spec.

Nothing in the codebase resolved a condition/annotation from a plain string before
this (the PyMOL registry only ever imports the Python objects directly), and the
shorthand-alias scan spans three different modules (most `C_*` aliases actually
live in `atomworks.ml.conditions`, not `rfproteina`) -- both are easy to silently
regress, so covered explicitly here rather than just exercised incidentally by the
end-to-end condition_spec test. Enum-value coercion is covered thoroughly because a
condition spec is often round-tripped through JSON (whose dict keys are always
strings) or hand-written YAML, both of which can produce inputs Python's own
`IntEnum` machinery doesn't handle the way a spec author would expect.
"""

import numpy as np
import pytest
from atomworks.ml.conditions import C_CRD, C_IDX

from rfproteina.conditions import C_HOT, RasaEnum
from rfproteina.utils.condition_resolution import coerce_enum_value, resolve_standard_annotation
from rfproteina.utils.standard_annotations import SA_SRC


@pytest.mark.parametrize(
    "alias,systematic_name,expected_cls",
    [
        ("C_HOT", "hotspot", C_HOT),  # alias defined in rfproteina.conditions
        ("C_CRD", "coordinate", C_CRD),  # alias defined in atomworks.ml.conditions
        ("SA_SRC", "src-component", SA_SRC),  # alias defined in rfproteina.utils.standard_annotations
    ],
)
def test_alias_and_systematic_name_resolve_to_the_same_class(alias, systematic_name, expected_cls):
    assert resolve_standard_annotation(alias) is expected_cls
    assert resolve_standard_annotation(systematic_name) is expected_cls


@pytest.mark.parametrize(
    "value,expected",
    [
        ("BURIED", RasaEnum.BURIED.value),
        (RasaEnum.BURIED.value, RasaEnum.BURIED.value),
        ("3", RasaEnum.EXPOSED.value),  # JSON dict keys are always strings, even for an int value
        (3, RasaEnum.EXPOSED.value),
    ],
)
def test_enum_value_coercion_accepts_equivalent_forms(value, expected):
    assert coerce_enum_value(RasaEnum, value) == expected


def test_presence_authored_conditions_write_unified_annotations(lysozyme):
    from rfproteina.inference_utils.condition_spec import _apply_condition_value

    for cls in (C_CRD, C_IDX):
        arr = lysozyme.copy()
        indices = np.arange(5)
        _apply_condition_value(cls, arr, indices, True)

        assert cls.has_annotation(arr)
        assert cls.mask(arr, default="generate")[indices].all()


def test_presence_authoring_only_updates_selected_coordinate_values(lysozyme):
    from rfproteina.inference_utils.condition_spec import _apply_condition_value

    arr = lysozyme.copy()
    persisted = C_CRD.default_annotation(arr)
    persisted[5:10] = C_CRD.ground_truth_value(arr)[5:10] + 100
    C_CRD.set_annotation(arr, persisted)

    _apply_condition_value(C_CRD, arr, np.arange(5), True)
    np.testing.assert_array_equal(C_CRD.annotation(arr)[:5], C_CRD.ground_truth_value(arr)[:5])
    np.testing.assert_array_equal(C_CRD.annotation(arr)[5:10], persisted[5:10])

    _apply_condition_value(C_CRD, arr, np.arange(2), False)
    assert np.isnan(C_CRD.annotation(arr)[:2]).all()
    np.testing.assert_array_equal(C_CRD.annotation(arr)[2:5], C_CRD.ground_truth_value(arr)[2:5])
    np.testing.assert_array_equal(C_CRD.annotation(arr)[5:10], persisted[5:10])
