"""Lightweight orientation analysis primitives for SAMMD trajectories."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

Coordinate = tuple[float, float, float]
SurfaceSide = Literal["top", "bottom"]


@dataclass(frozen=True)
class OrientationResult:
    """Result of a reactant orientation analysis calculation.

    Attributes
    ----------
    angle_degrees
        Angle between the reactant orientation vector and surface normal in degrees.
    vector
        Unit vector pointing from the reactant center of mass to the target point.
    normal
        Unit surface normal used for the angle calculation.
    com
        Reactant center of mass in the same units as input coordinates.
    target_point
        Selected target atom, midpoint, or centroid in the same units as input coordinates.
    side
        Surface side used to choose the normal, or ``None`` when an explicit normal is supplied.
    """

    angle_degrees: float
    vector: Coordinate
    normal: Coordinate
    com: Coordinate
    target_point: Coordinate
    side: SurfaceSide | None


def dot(a: Coordinate, b: Coordinate) -> float:
    """Calculate the dot product of two 3D vectors.

    Parameters
    ----------
    a, b
        3D vectors.

    Returns
    -------
    float
        Dot product of the input vectors.
    """

    vector_a = _validate_vector(a, "a")
    vector_b = _validate_vector(b, "b")
    return sum(
        component_a * component_b
        for component_a, component_b in zip(vector_a, vector_b, strict=True)
    )


def norm(vector: Coordinate) -> float:
    """Calculate the Euclidean norm of a 3D vector.

    Parameters
    ----------
    vector
        3D vector.

    Returns
    -------
    float
        Euclidean vector norm.
    """

    validated = _validate_vector(vector, "vector")
    return math.sqrt(dot(validated, validated))


def normalize(vector: Coordinate) -> Coordinate:
    """Return a unit vector in the same direction as a 3D vector.

    Parameters
    ----------
    vector
        3D vector.

    Returns
    -------
    Coordinate
        Normalized 3D vector.

    Raises
    ------
    ValueError
        If the vector has zero length.
    """

    validated = _validate_vector(vector, "vector")
    vector_norm = norm(validated)
    if vector_norm == 0.0:
        msg = "vector must be nonzero"
        raise ValueError(msg)
    return tuple(component / vector_norm for component in validated)


def angle_degrees(vector: Coordinate, reference: Coordinate) -> float:
    """Calculate the angle between two 3D vectors in degrees.

    Parameters
    ----------
    vector, reference
        3D vectors to compare.

    Returns
    -------
    float
        Angle between the input vectors in degrees.
    """

    unit_vector = normalize(vector)
    unit_reference = normalize(reference)
    cosine = max(-1.0, min(1.0, dot(unit_vector, unit_reference)))
    return math.degrees(math.acos(cosine))


def center_of_mass(
    coordinates: Sequence[Coordinate], masses: Sequence[float] | None = None
) -> Coordinate:
    """Calculate the center of mass or geometric center of coordinates.

    Parameters
    ----------
    coordinates
        0-based coordinate sequence for the reactant atoms.
    masses
        Optional positive masses aligned with ``coordinates``. When omitted, equal masses are used.

    Returns
    -------
    Coordinate
        Center of mass in the same units as input coordinates.
    """

    validated_coordinates = _validate_coordinates(coordinates)
    if masses is None:
        weight = float(len(validated_coordinates))
        return tuple(
            sum(coordinate[axis] for coordinate in validated_coordinates) / weight
            for axis in range(3)
        )

    validated_masses = _validate_masses(masses, len(validated_coordinates))
    total_mass = sum(validated_masses)
    return tuple(
        sum(
            coordinate[axis] * mass
            for coordinate, mass in zip(validated_coordinates, validated_masses, strict=True)
        )
        / total_mass
        for axis in range(3)
    )


def target_point(
    coordinates: Sequence[Coordinate],
    *,
    atom_index: int | None = None,
    atom_indices: Sequence[int] | None = None,
) -> Coordinate:
    """Calculate a target point from 0-based atom selection indices.

    Parameters
    ----------
    coordinates
        Coordinate sequence for the reactant atoms.
    atom_index
        Single 0-based atom index for the target atom.
    atom_indices
        0-based atom indices whose centroid defines the target point. Two indices give a midpoint.

    Returns
    -------
    Coordinate
        Target atom coordinate, midpoint, or centroid.
    """

    validated_coordinates = _validate_coordinates(coordinates)
    if (atom_index is None) == (atom_indices is None):
        msg = "define exactly one of atom_index or atom_indices"
        raise ValueError(msg)

    if atom_index is not None:
        index = _validate_index(atom_index, len(validated_coordinates), "atom_index")
        return validated_coordinates[index]

    if atom_indices is None or len(atom_indices) == 0:
        msg = "atom_indices must contain at least one index"
        raise ValueError(msg)
    indices = [
        _validate_index(index, len(validated_coordinates), "atom_indices")
        for index in atom_indices
    ]
    return tuple(
        sum(validated_coordinates[index][axis] for index in indices) / len(indices)
        for axis in range(3)
    )


def orientation_vector(com: Coordinate, target: Coordinate) -> Coordinate:
    """Calculate a unit orientation vector from COM to target point.

    Parameters
    ----------
    com
        Center of mass coordinate.
    target
        Target point coordinate.

    Returns
    -------
    Coordinate
        Unit vector pointing from ``com`` to ``target``.
    """

    validated_com = _validate_vector(com, "com")
    validated_target = _validate_vector(target, "target")
    return normalize(
        tuple(
            target_component - com_component
            for com_component, target_component in zip(
                validated_com, validated_target, strict=True
            )
        )
    )


def surface_normal(side: SurfaceSide) -> Coordinate:
    """Return the unit surface normal for a named SAMMD surface side.

    Parameters
    ----------
    side
        ``"top"`` for +z or ``"bottom"`` for -z.

    Returns
    -------
    Coordinate
        Unit surface normal.
    """

    if side == "top":
        return (0.0, 0.0, 1.0)
    if side == "bottom":
        return (0.0, 0.0, -1.0)
    msg = "side must be 'top' or 'bottom'"
    raise ValueError(msg)


def analyze_orientation(
    coordinates: Sequence[Coordinate],
    *,
    atom_index: int | None = None,
    atom_indices: Sequence[int] | None = None,
    masses: Sequence[float] | None = None,
    side: SurfaceSide | None = "top",
    normal: Coordinate | None = None,
) -> OrientationResult:
    """Analyze reactant orientation relative to a surface normal.

    Parameters
    ----------
    coordinates
        Reactant atom coordinates using 0-based Python atom-index conventions.
    atom_index
        Single target atom index.
    atom_indices
        Target atom indices whose centroid defines the target point.
    masses
        Optional positive atom masses aligned with ``coordinates`` for COM calculation.
    side
        Surface side used when ``normal`` is not supplied.
    normal
        Optional explicit surface normal vector.

    Returns
    -------
    OrientationResult
        Orientation angle and intermediate geometry values.
    """

    com = center_of_mass(coordinates, masses)
    target = target_point(coordinates, atom_index=atom_index, atom_indices=atom_indices)
    vector = orientation_vector(com, target)
    result_side = side
    if normal is None:
        if side is None:
            msg = "side is required when normal is not supplied"
            raise ValueError(msg)
        result_normal = surface_normal(side)
    else:
        result_normal = normalize(normal)
        result_side = None
    angle = angle_degrees(vector, result_normal)
    return OrientationResult(
        angle_degrees=angle,
        vector=vector,
        normal=result_normal,
        com=com,
        target_point=target,
        side=result_side,
    )


def _validate_coordinates(coordinates: Sequence[Coordinate]) -> tuple[Coordinate, ...]:
    """Validate a coordinate sequence for analysis calculations."""

    if len(coordinates) == 0:
        msg = "coordinates must contain at least one coordinate"
        raise ValueError(msg)
    return tuple(_validate_vector(coordinate, "coordinate") for coordinate in coordinates)


def _validate_vector(vector: Sequence[float], name: str) -> Coordinate:
    """Validate one finite 3D numeric vector."""

    if len(vector) != 3:
        msg = f"{name} must contain exactly three coordinates"
        raise ValueError(msg)
    validated = tuple(_validate_finite_number(component, name) for component in vector)
    return (validated[0], validated[1], validated[2])


def _validate_finite_number(value: float, name: str) -> float:
    """Validate a finite scalar coordinate or mass value."""

    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
        msg = f"{name} values must be finite numbers"
        raise ValueError(msg)
    return float(value)


def _validate_masses(masses: Sequence[float], coordinate_count: int) -> tuple[float, ...]:
    """Validate positive finite masses aligned with coordinates."""

    if len(masses) != coordinate_count:
        msg = "masses length must match coordinates length"
        raise ValueError(msg)
    validated = tuple(_validate_finite_number(mass, "masses") for mass in masses)
    if any(mass <= 0.0 for mass in validated):
        msg = "masses must be positive"
        raise ValueError(msg)
    return validated


def _validate_index(index: int, coordinate_count: int, name: str) -> int:
    """Validate a 0-based atom index against coordinate length."""

    if isinstance(index, bool) or not isinstance(index, int):
        msg = f"{name} must contain integer 0-based indices"
        raise ValueError(msg)
    if index < 0 or index >= coordinate_count:
        msg = f"{name} index {index} is out of range for {coordinate_count} coordinates"
        raise IndexError(msg)
    return index
