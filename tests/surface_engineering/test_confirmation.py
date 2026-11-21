"""Confirmation cannot promote an unsafe or single-scaffold-only chemistry hit."""

import pytest

from scripts.run_surface_confirmation import select_finalists, validate_zero_controls


class TestFinalistSelection:
    def test_requires_every_target_mode_and_prefers_distinct_mechanisms(
        self, confirmation_rows: list[dict[str, str | float | int]]
    ) -> None:
        rows = confirmation_rows
        rows.append(dict(rows[-1], arm="unsafe", surface_fraction_DETA=1.0, passes_geometry_gate=0))
        assert select_finalists(rows) == ["notion_moderate", "xtr_moderate"]
        for row in rows:
            if row["target"] == "tll" and row["arm"] == "notion_moderate":
                row["passes_geometry_gate"] = 0
        assert select_finalists(rows) == ["xtr_moderate"]

    def test_no_qualified_hit_is_not_replaced_with_a_failed_candidate(self) -> None:
        with pytest.raises(ValueError, match="baseline"):
            select_finalists([])


class TestZeroControlCompleteness:
    def test_missing_or_duplicate_stratum_cannot_pass(self) -> None:
        complete = [
            {"target": target, "mode": mode, "same_identities_and_atoms": True, "max_coordinate_delta_A": 0.0}
            for target in ["tll", "bacillus"]
            for mode in ["inverse", "partial035"]
        ]
        validate_zero_controls(complete)
        with pytest.raises(RuntimeError, match="four|strata"):
            validate_zero_controls(complete[:3])
        with pytest.raises(RuntimeError, match="four|strata"):
            validate_zero_controls(complete[:3] + complete[:1])
