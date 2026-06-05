"""Small 3D geometry helpers used by SAMMD placement and validation workflows."""

from __future__ import annotations

import math

Vector3 = tuple[float, float, float]
Matrix3 = tuple[Vector3, Vector3, Vector3]


def add_vectors(left: Vector3, right: Vector3) -> Vector3:
    """Add two 3D vectors."""

    return (left[0] + right[0], left[1] + right[1], left[2] + right[2])


def subtract_vectors(left: Vector3, right: Vector3) -> Vector3:
    """Subtract two 3D vectors."""

    return (left[0] - right[0], left[1] - right[1], left[2] - right[2])


def scale_vector(vector: Vector3, scale: float) -> Vector3:
    """Scale a 3D vector."""

    return (vector[0] * scale, vector[1] * scale, vector[2] * scale)


def dot_product(left: Vector3, right: Vector3) -> float:
    """Return the dot product of two 3D vectors."""

    return left[0] * right[0] + left[1] * right[1] + left[2] * right[2]


def cross_product(left: Vector3, right: Vector3) -> Vector3:
    """Return the cross product of two 3D vectors."""

    return (
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    )


def norm(vector: Vector3) -> float:
    """Return the Euclidean norm of a 3D vector."""

    return math.sqrt(dot_product(vector, vector))


def normalize(vector: Vector3) -> Vector3:
    """Return a unit vector in the same direction as ``vector``."""

    vector_norm = norm(vector)
    if vector_norm < 1.0e-15:
        msg = "cannot normalize a zero vector"
        raise ValueError(msg)
    return (vector[0] / vector_norm, vector[1] / vector_norm, vector[2] / vector_norm)


def distance(left: Vector3, right: Vector3) -> float:
    """Return Euclidean distance between two 3D points."""

    return norm(subtract_vectors(left, right))


def centroid(positions: tuple[Vector3, ...]) -> Vector3:
    """Return the coordinate centroid of one or more 3D positions."""

    if not positions:
        msg = "positions must contain at least one coordinate"
        raise ValueError(msg)
    count = len(positions)
    return (
        sum(position[0] for position in positions) / count,
        sum(position[1] for position in positions) / count,
        sum(position[2] for position in positions) / count,
    )


def matvec(matrix: Matrix3, vector: Vector3) -> Vector3:
    """Multiply a 3x3 matrix by a 3D vector."""

    return tuple(dot_product(row, vector) for row in matrix)  # type: ignore[return-value]


def rotate_about_axis(vector: Vector3, axis: Vector3, angle_rad: float) -> Vector3:
    """Rotate a vector around an arbitrary axis through the origin."""

    unit_axis = normalize(axis)
    cosine = math.cos(angle_rad)
    sine = math.sin(angle_rad)
    parallel = scale_vector(unit_axis, dot_product(unit_axis, vector) * (1.0 - cosine))
    cross = cross_product(unit_axis, vector)
    return add_vectors(
        add_vectors(scale_vector(vector, cosine), scale_vector(cross, sine)),
        parallel,
    )


def rotation_matrix(source: Vector3, target: Vector3) -> Matrix3:
    """Return a 3x3 matrix rotating ``source`` onto ``target``."""

    source_unit = normalize(source)
    target_unit = normalize(target)
    cross = cross_product(source_unit, target_unit)
    dot = max(-1.0, min(1.0, dot_product(source_unit, target_unit)))
    sine = norm(cross)
    if sine < 1.0e-12:
        if dot > 0.0:
            return ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
        axis_candidate = cross_product(source_unit, (1.0, 0.0, 0.0))
        if norm(axis_candidate) < 1.0e-12:
            axis_candidate = cross_product(source_unit, (0.0, 1.0, 0.0))
        axis = normalize(axis_candidate)
        return axis_angle_matrix(axis, math.pi)
    k_matrix = skew_matrix(cross)
    k_squared = matrix_multiply(k_matrix, k_matrix)
    scale = (1.0 - dot) / (sine * sine)
    identity = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    return matrix_add(matrix_add(identity, k_matrix), matrix_scale(k_squared, scale))


def axis_angle_matrix(axis: Vector3, angle_rad: float) -> Matrix3:
    """Return a 3x3 rotation matrix for an axis-angle rotation."""

    x, y, z = axis
    cosine = math.cos(angle_rad)
    sine = math.sin(angle_rad)
    one_minus_cosine = 1.0 - cosine
    return (
        (
            cosine + x * x * one_minus_cosine,
            x * y * one_minus_cosine - z * sine,
            x * z * one_minus_cosine + y * sine,
        ),
        (
            y * x * one_minus_cosine + z * sine,
            cosine + y * y * one_minus_cosine,
            y * z * one_minus_cosine - x * sine,
        ),
        (
            z * x * one_minus_cosine - y * sine,
            z * y * one_minus_cosine + x * sine,
            cosine + z * z * one_minus_cosine,
        ),
    )


def indexed_displacements(
    final_positions_nm: tuple[Vector3, ...],
    reference_positions_nm: tuple[Vector3, ...],
    indices: tuple[int, ...],
) -> tuple[float, ...]:
    """Return displacement magnitudes for atom indices against same-index references."""

    return tuple(
        distance(final_positions_nm[index], reference_positions_nm[index]) for index in indices
    )


def indexed_reference_displacements(
    final_positions_nm: tuple[Vector3, ...],
    indices: tuple[int, ...],
    reference_positions_nm: tuple[Vector3, ...],
) -> tuple[float, ...]:
    """Return displacement magnitudes against a compact reference list."""

    return tuple(
        distance(final_positions_nm[index], reference)
        for index, reference in zip(indices, reference_positions_nm, strict=True)
    )


def angle_between_points(first: Vector3, center: Vector3, third: Vector3) -> float:
    """Return the first-center-third angle in radians."""

    left = normalize(subtract_vectors(first, center))
    right = normalize(subtract_vectors(third, center))
    cosine = max(-1.0, min(1.0, dot_product(left, right)))
    return math.acos(cosine)


def skew_matrix(vector: Vector3) -> Matrix3:
    """Return the skew-symmetric matrix for a 3D vector."""

    x, y, z = vector
    return ((0.0, -z, y), (z, 0.0, -x), (-y, x, 0.0))


def matrix_multiply(left: Matrix3, right: Matrix3) -> Matrix3:
    """Multiply two 3x3 matrices."""

    columns = tuple((right[0][index], right[1][index], right[2][index]) for index in range(3))
    return tuple(tuple(dot_product(row, column) for column in columns) for row in left)  # type: ignore[return-value]


def matrix_add(left: Matrix3, right: Matrix3) -> Matrix3:
    """Add two 3x3 matrices."""

    return tuple(
        tuple(left[row][column] + right[row][column] for column in range(3))
        for row in range(3)
    )  # type: ignore[return-value]


def matrix_scale(matrix: Matrix3, scale: float) -> Matrix3:
    """Scale a 3x3 matrix."""

    return tuple(tuple(value * scale for value in row) for row in matrix)  # type: ignore[return-value]
