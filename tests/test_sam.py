"""Tests for lightweight SAM placement planning."""

import pytest

from sammd.config import SAMComponentConfig, SAMConfig
from sammd.sam import plan_sam_placements
from sammd.surfaces import generate_binding_sites, plan_pd111_slab


def _binding_sites():
    """Return deterministic binding sites for a small Pd(111) slab."""

    return generate_binding_sites(plan_pd111_slab((1.0, 1.0), 3))


def test_grafting_density_selects_expected_site_count() -> None:
    """Convert area and grafting density to per-side selected site count."""

    plan = plan_sam_placements(SAMConfig(), _binding_sites(), lateral_area_nm2=1.0, seed=7)

    assert plan.selected_sites_per_side == 4
    assert len(plan.placements) == 8


def test_fraction_mode_assignment_sums_exactly_and_is_seeded() -> None:
    """Assign fraction-mode mixed SAMs with exact counts on each face."""

    sam_config = SAMConfig(
        components=[
            SAMComponentConfig(name="a", smiles="CCCS", fraction=0.25),
            SAMComponentConfig(name="b", smiles="CCCCS", fraction=0.75),
        ]
    )

    first = plan_sam_placements(sam_config, _binding_sites(), lateral_area_nm2=1.0, seed=11)
    second = plan_sam_placements(sam_config, _binding_sites(), lateral_area_nm2=1.0, seed=11)

    assert first == second
    for side in ("bottom", "top"):
        side_names = [
            placement.component_name for placement in first.placements if placement.side == side
        ]
        assert side_names.count("a") == 1
        assert side_names.count("b") == 3
        assert len(side_names) == 4


def test_count_mode_assignment_validates_total_counts() -> None:
    """Validate count-mode component totals against selected sites per face."""

    valid_config = SAMConfig(
        components=[
            SAMComponentConfig(name="a", smiles="CCCS", count=2),
            SAMComponentConfig(name="b", smiles="CCCCS", count=2),
        ]
    )
    invalid_config = SAMConfig(
        components=[
            SAMComponentConfig(name="a", smiles="CCCS", count=2),
            SAMComponentConfig(name="b", smiles="CCCCS", count=1),
        ]
    )

    plan = plan_sam_placements(valid_config, _binding_sites(), lateral_area_nm2=1.0, seed=13)

    assert len(plan.placements) == 8
    with pytest.raises(ValueError, match="counts sum to 3, but total_sites is 4"):
        plan_sam_placements(invalid_config, _binding_sites(), lateral_area_nm2=1.0, seed=13)


def test_top_and_bottom_surfaces_are_both_planned() -> None:
    """Decorate both exposed faces with independent selected sites."""

    plan = plan_sam_placements(SAMConfig(), _binding_sites(), lateral_area_nm2=1.0, seed=17)

    sides = [placement.side for placement in plan.placements]

    assert sides.count("bottom") == 4
    assert sides.count("top") == 4
    assert {placement.normal for placement in plan.placements if placement.side == "top"} == {
        (0.0, 0.0, 1.0)
    }
    assert {placement.normal for placement in plan.placements if placement.side == "bottom"} == {
        (0.0, 0.0, -1.0)
    }
