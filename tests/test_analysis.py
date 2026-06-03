"""Tests for lightweight orientation analysis primitives."""

import math

import pytest

from sammd.analysis import (
    analyze_orientation,
    angle_degrees,
    center_of_mass,
    orientation_vector,
    surface_normal,
    target_point,
)


def test_angle_degrees_known_vector_normal_cases() -> None:
    """Calculate known 0, 90, and 180 degree angles."""

    assert angle_degrees((0.0, 0.0, 1.0), (0.0, 0.0, 1.0)) == pytest.approx(0.0)
    assert angle_degrees((1.0, 0.0, 0.0), (0.0, 0.0, 1.0)) == pytest.approx(90.0)
    assert angle_degrees((0.0, 0.0, -1.0), (0.0, 0.0, 1.0)) == pytest.approx(180.0)


def test_center_of_mass_without_masses_uses_equal_weights() -> None:
    """Average coordinates when masses are omitted."""

    coordinates = [(0.0, 0.0, 0.0), (2.0, 4.0, 6.0)]

    assert center_of_mass(coordinates) == pytest.approx((1.0, 2.0, 3.0))


def test_center_of_mass_with_masses_uses_weighted_average() -> None:
    """Weight coordinates by positive masses when supplied."""

    coordinates = [(0.0, 0.0, 0.0), (2.0, 0.0, 0.0)]
    masses = [1.0, 3.0]

    assert center_of_mass(coordinates, masses) == pytest.approx((1.5, 0.0, 0.0))


def test_target_point_supports_atom_midpoint_and_centroid() -> None:
    """Select targets from one atom or centroid-style atom groups."""

    coordinates = [(0.0, 0.0, 0.0), (2.0, 0.0, 0.0), (2.0, 3.0, 1.0)]

    assert target_point(coordinates, atom_index=2) == pytest.approx((2.0, 3.0, 1.0))
    assert target_point(coordinates, atom_indices=[0, 1]) == pytest.approx((1.0, 0.0, 0.0))
    assert target_point(coordinates, atom_indices=[0, 1, 2]) == pytest.approx(
        (4.0 / 3.0, 1.0, 1.0 / 3.0)
    )


def test_top_and_bottom_side_normals_are_z_aligned() -> None:
    """Map SAMMD surface sides to deterministic z-axis normals."""

    assert surface_normal("top") == (0.0, 0.0, 1.0)
    assert surface_normal("bottom") == (0.0, 0.0, -1.0)


def test_analyze_orientation_uses_side_specific_normals() -> None:
    """Report opposite angles for top and bottom surface sides."""

    coordinates = [(0.0, 0.0, 0.0), (0.0, 0.0, 2.0)]

    top = analyze_orientation(coordinates, atom_index=1, side="top")
    bottom = analyze_orientation(coordinates, atom_index=1, side="bottom")

    assert top.angle_degrees == pytest.approx(0.0)
    assert top.vector == pytest.approx((0.0, 0.0, 1.0))
    assert top.normal == (0.0, 0.0, 1.0)
    assert top.side == "top"
    assert bottom.angle_degrees == pytest.approx(180.0)
    assert bottom.normal == (0.0, 0.0, -1.0)
    assert bottom.side == "bottom"


def test_analyze_orientation_accepts_explicit_normal() -> None:
    """Normalize explicit normals and suppress side in the result."""

    coordinates = [(0.0, 0.0, 0.0), (0.0, 2.0, 0.0)]

    result = analyze_orientation(coordinates, atom_index=1, normal=(0.0, 4.0, 0.0))

    assert result.angle_degrees == pytest.approx(0.0)
    assert result.normal == pytest.approx((0.0, 1.0, 0.0))
    assert result.side is None


@pytest.mark.parametrize("indices", [[-1], [2]])
def test_invalid_indices_fail_clearly(indices: list[int]) -> None:
    """Reject atom selections outside the 0-based coordinate range."""

    with pytest.raises(IndexError, match="out of range"):
        target_point([(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)], atom_indices=indices)


def test_zero_length_orientation_vector_fails_clearly() -> None:
    """Reject target points identical to the center of mass."""

    with pytest.raises(ValueError, match="nonzero"):
        orientation_vector((1.0, 1.0, 1.0), (1.0, 1.0, 1.0))


def test_mismatched_masses_fail_clearly() -> None:
    """Require one mass per coordinate."""

    with pytest.raises(ValueError, match="masses length"):
        center_of_mass([(0.0, 0.0, 0.0)], masses=[1.0, 2.0])


def test_nonpositive_masses_fail_clearly() -> None:
    """Require positive mass values."""

    with pytest.raises(ValueError, match="positive"):
        center_of_mass([(0.0, 0.0, 0.0)], masses=[0.0])


def test_non_finite_coordinates_fail_clearly() -> None:
    """Reject NaN and infinite coordinates before analysis."""

    with pytest.raises(ValueError, match="finite numbers"):
        center_of_mass([(math.nan, 0.0, 0.0)])


def test_invalid_side_fails_clearly() -> None:
    """Reject unknown surface-side names."""

    with pytest.raises(ValueError, match="side must"):
        surface_normal("middle")  # type: ignore[arg-type]


def test_cinnamaldehyde_like_toy_coordinates_are_deterministic() -> None:
    """Analyze a small RDKit-free reactant geometry deterministically."""

    coordinates = [
        (-2.0, 0.0, 0.0),
        (-1.0, 0.8, 0.0),
        (0.0, 0.7, 0.0),
        (1.0, 0.4, 0.0),
        (2.0, 0.2, 0.0),
        (3.0, 0.1, 0.0),
    ]
    masses = [12.0, 12.0, 12.0, 12.0, 12.0, 16.0]

    result = analyze_orientation(coordinates, atom_indices=[4, 5], masses=masses, side="top")

    assert result.angle_degrees == pytest.approx(90.0)
    assert result.com == pytest.approx((0.631578947368421, 0.3526315789473684, 0.0))
    assert result.target_point == pytest.approx((2.5, 0.15000000000000002, 0.0))
    assert result.vector[2] == pytest.approx(0.0)
