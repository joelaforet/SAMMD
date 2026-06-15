"""Tests for deterministic SAM placement planning."""

from math import dist, pi, sqrt

import pytest

from sammd.core.config import SAMComponentConfig, SAMConfig
from sammd.model.metal_sulfur import (
    METAL_SULFUR_EPSILON_KCAL_MOL,
    METAL_SULFUR_EPSILON_KJ_MOL,
    METAL_SULFUR_INTERACTION_MODE,
    METAL_SULFUR_PAIRS_PER_ANCHOR,
    METAL_SULFUR_SIGMA_NM,
)
from sammd.model.sam import (
    DEFAULT_SULFUR_HEIGHT_NM,
    plan_anchor_pose,
    plan_sam_placements,
    sam_azimuth_rad,
)
from sammd.model.surfaces import generate_binding_sites, plan_fcc111_slab


def _binding_sites():
    """Return deterministic binding sites for a small Fcc(111) slab."""

    return generate_binding_sites(plan_fcc111_slab("Pd", (1.0, 1.0), 3))


def _mixed_binding_sites():
    """Return fcc and hcp binding sites for one deterministic slab."""

    slab = plan_fcc111_slab("Pd", (1.0, 1.0), 3)
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
            SAMComponentConfig(name="a", residue_name="AAA", smiles="CCCS", fraction=0.25),
            SAMComponentConfig(name="b", residue_name="BBB", smiles="CCCCS", fraction=0.75),
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


def test_site_selection_avoids_nearest_neighbor_clustering() -> None:
    """Dense smoke systems should not place SAM anchors on adjacent hollow sites."""

    slab = plan_fcc111_slab("Pd", (2.0, 2.0), 5)
    lateral_area_nm2 = slab.lateral_size_nm[0] * slab.lateral_size_nm[1]
    plan = plan_sam_placements(
        SAMConfig(), generate_binding_sites(slab), lateral_area_nm2=lateral_area_nm2, seed=2026
    )

    for side in ("bottom", "top"):
        positions = [
            placement.position_nm for placement in plan.placements if placement.side == side
        ]
        min_distance = min(
            dist(left, right)
            for index, left in enumerate(positions)
            for right in positions[index + 1 :]
        )

        assert min_distance > 0.45


def test_count_mode_assignment_validates_total_counts() -> None:
    """Validate count-mode component totals against selected sites per face."""

    valid_config = SAMConfig(
        components=[
            SAMComponentConfig(name="a", residue_name="AAA", smiles="CCCS", count=2),
            SAMComponentConfig(name="b", residue_name="BBB", smiles="CCCCS", count=2),
        ]
    )
    invalid_config = SAMConfig(
        components=[
            SAMComponentConfig(name="a", residue_name="AAA", smiles="CCCS", count=2),
            SAMComponentConfig(name="b", residue_name="BBB", smiles="CCCCS", count=1),
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


def test_placement_uses_internal_fcc_hollow_anchor_strategy() -> None:
    """Keep anchor details out of the student config while preserving metadata."""

    plan = plan_sam_placements(SAMConfig(), _mixed_binding_sites(), lateral_area_nm2=1.0, seed=23)

    assert {placement.site_kind for placement in plan.placements} == {"fcc_hollow"}
    assert {placement.anchor_metadata["site"] for placement in plan.placements} == {"fcc_hollow"}
    assert {placement.anchor_metadata["mode"] for placement in plan.placements} == {
        METAL_SULFUR_INTERACTION_MODE
    }


def test_placements_record_canonical_metal_sulfur_strategy() -> None:
    """Record export-ready metal-S LJ override metadata for every placement."""

    plan = plan_sam_placements(SAMConfig(), _binding_sites(), lateral_area_nm2=1.0, seed=23)

    for placement in plan.placements:
        interaction = placement.anchor_metadata["metal_sulfur_interaction"]

        assert placement.anchor_pose.attachment_mode == METAL_SULFUR_INTERACTION_MODE
        assert len(placement.anchor_pose.nearest_metal_atom_indices) == (
            METAL_SULFUR_PAIRS_PER_ANCHOR
        )
        assert len(placement.anchor_metadata["nearest_metal_atom_indices"]) == (
            METAL_SULFUR_PAIRS_PER_ANCHOR
        )
        assert interaction["mode"] == METAL_SULFUR_INTERACTION_MODE
        assert interaction["site_kind"] == "fcc_hollow"
        assert interaction["pairs_per_anchor"] == METAL_SULFUR_PAIRS_PER_ANCHOR
        assert interaction["sigma_nm"] == METAL_SULFUR_SIGMA_NM
        assert interaction["epsilon_kcal_mol"] == METAL_SULFUR_EPSILON_KCAL_MOL
        assert interaction["epsilon_kj_mol"] == METAL_SULFUR_EPSILON_KJ_MOL


def test_anchor_pose_offsets_sulfur_from_site_plane_and_preserves_metadata() -> None:
    """Represent sulfur anchor placeholders above or below the binding-site plane."""

    plan = plan_sam_placements(SAMConfig(), _binding_sites(), lateral_area_nm2=1.0, seed=29)

    for placement in plan.placements:
        pose = placement.anchor_pose
        assert placement.position_nm == pose.site_position_nm
        assert pose.site_kind == placement.site_kind == "fcc_hollow"
        assert pose.normal == placement.normal
        assert pose.axis_direction == placement.normal
        assert pose.attachment_mode == METAL_SULFUR_INTERACTION_MODE
        assert pose.nearest_metal_atom_indices == placement.anchor_metadata[
            "nearest_metal_atom_indices"
        ]
        assert pose.sulfur_height_nm == DEFAULT_SULFUR_HEIGHT_NM
        assert pose.sulfur_position_nm[:2] == pose.site_position_nm[:2]
        expected_z = pose.site_position_nm[2] + pose.normal[2] * DEFAULT_SULFUR_HEIGHT_NM
        assert pose.sulfur_position_nm[2] == expected_z


def test_plan_anchor_pose_supports_explicit_height_and_azimuth() -> None:
    """Allow callers to build an anchor pose for a single site without export deps."""

    site = _binding_sites()[0]
    pose = plan_anchor_pose(site, sulfur_height_nm=0.25, azimuth_rad=1.5)

    assert pose.site_position_nm == site.position_nm
    assert pose.sulfur_position_nm[2] == site.position_nm[2] + site.normal[2] * 0.25
    assert pose.azimuth_rad == 1.5
    assert pose.nearest_metal_atom_indices == site.nearest_metal_atom_indices


def test_sam_azimuth_rad_uses_deterministic_golden_angle_sequence() -> None:
    """Distribute per-placement twist deterministically without random state."""

    golden_angle_rad = pi * (3.0 - sqrt(5.0))

    assert sam_azimuth_rad(0) == 0.0
    assert sam_azimuth_rad(1) == golden_angle_rad
    assert sam_azimuth_rad(2) == (2 * golden_angle_rad) % (2.0 * pi)

    plan = plan_sam_placements(SAMConfig(), _binding_sites(), lateral_area_nm2=1.0, seed=31)
    for side in ("bottom", "top"):
        side_placements = [placement for placement in plan.placements if placement.side == side]
        assert [placement.anchor_pose.azimuth_rad for placement in side_placements] == [
            sam_azimuth_rad(index) for index in range(len(side_placements))
        ]
