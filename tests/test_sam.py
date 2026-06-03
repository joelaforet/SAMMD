"""Tests for lightweight SAM placement planning."""

import pytest

from sammd.config import AnchorConfig, SAMComponentConfig, SAMConfig
from sammd.sam import plan_sam_placements
from sammd.surfaces import generate_binding_sites, plan_pd111_slab


def _binding_sites():
    """Return deterministic binding sites for a small Pd(111) slab."""

    return generate_binding_sites(plan_pd111_slab((1.0, 1.0), 3))


def _mixed_binding_sites():
    """Return fcc and hcp binding sites for one deterministic slab."""

    slab = plan_pd111_slab((1.0, 1.0), 3)
    return generate_binding_sites(slab, "fcc_hollow") + generate_binding_sites(slab, "hcp_hollow")


def test_grafting_density_selects_expected_site_count() -> None:
    """Convert area and grafting density to per-side selected site count."""

    plan = plan_sam_placements(SAMConfig(), _binding_sites(), lateral_area_nm2=1.0, seed=7)

    assert plan.selected_sites_per_side == 4
    assert len(plan.placements) == 8


def test_grafting_density_uses_half_up_rounding() -> None:
    """Round .5 target grafting counts up instead of using banker rounding."""

    plan = plan_sam_placements(SAMConfig(), _binding_sites(), lateral_area_nm2=1.125, seed=7)

    assert plan.selected_sites_per_side == 5
    assert len(plan.placements) == 10


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


def test_anchor_site_mismatch_fails_clearly() -> None:
    """Reject SAM configs that request a site kind absent from supplied sites."""

    sam_config = SAMConfig(anchor=AnchorConfig(site="bridge"))

    with pytest.raises(ValueError, match=r"bridge.*supplied binding sites contain: fcc_hollow"):
        plan_sam_placements(sam_config, _binding_sites(), lateral_area_nm2=1.0, seed=19)


def test_mixed_supplied_sites_filter_to_requested_anchor_site() -> None:
    """Prevent placement onto a supplied site kind that was not requested."""

    sam_config = SAMConfig(anchor=AnchorConfig(site="hcp_hollow"))

    plan = plan_sam_placements(sam_config, _mixed_binding_sites(), lateral_area_nm2=1.0, seed=23)

    assert {placement.site_kind for placement in plan.placements} == {"hcp_hollow"}
    assert {placement.anchor_metadata["site"] for placement in plan.placements} == {"hcp_hollow"}


def test_component_specific_mixed_anchor_sites_fail_before_assignment() -> None:
    """Reject mixed anchor site requests until per-site-kind placement is implemented."""

    sam_config = SAMConfig(
        anchor=AnchorConfig(site="fcc_hollow"),
        components=[
            SAMComponentConfig(
                name="a", smiles="CCCS", fraction=1.0, anchor=AnchorConfig(site="hcp_hollow")
            )
        ],
    )

    with pytest.raises(ValueError, match="mixed SAM anchor site kinds are not supported"):
        plan_sam_placements(sam_config, _mixed_binding_sites(), lateral_area_nm2=1.0, seed=29)
