"""Tests for shared geometry helpers."""

import math

import pytest

from sammd.geometry import (
    angle_between_points,
    centroid,
    indexed_displacements,
    matvec,
    norm,
    normalize,
    rotation_matrix,
)


def test_rotation_matrix_maps_source_vector_to_target_vector() -> None:
    """Orientation helpers should rotate one vector onto another."""

    matrix = rotation_matrix((1.0, 0.0, 0.0), (0.0, 0.0, -1.0))
    rotated = matvec(matrix, (1.0, 0.0, 0.0))

    assert rotated[0] == pytest.approx(0.0, abs=1.0e-12)
    assert rotated[1] == pytest.approx(0.0, abs=1.0e-12)
    assert rotated[2] == pytest.approx(-1.0, abs=1.0e-12)
    assert math.isclose(norm(rotated), 1.0)


def test_geometry_validates_empty_or_zero_length_inputs() -> None:
    """Reject undefined centroid and unit-vector operations clearly."""

    with pytest.raises(ValueError, match="positions must contain"):
        centroid(())
    with pytest.raises(ValueError, match="zero vector"):
        normalize((0.0, 0.0, 0.0))


def test_angle_and_indexed_displacements() -> None:
    """Measure angles and indexed coordinate displacements in nanometer tuples."""

    angle = angle_between_points((1.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 1.0, 0.0))
    displacements = indexed_displacements(
        ((0.0, 0.0, 0.0), (1.0, 1.0, 0.0)),
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
        (1,),
    )

    assert angle == pytest.approx(math.pi / 2.0)
    assert displacements == pytest.approx((1.0,))
