"""Shared dependency-free 3D geometry helpers."""

from __future__ import annotations

import math

Vector3 = tuple[float, float, float]
Matrix3 = tuple[Vector3, Vector3, Vector3]


def add_vectors(left: Vector3, right: Vector3) -> Vector3:
    """Add two vectors."""

    return (left[0] + right[0], left[1] + right[1], left[2] + right[2])


def subtract_vectors(left: Vector3, right: Vector3) -> Vector3:
    """Subtract ``right`` from ``left``."""

    return (left[0] - right[0], left[1] - right[1], left[2] - right[2])


def scale_vector(vector: Vector3, scale: float) -> Vector3:
    """Scale a vector by a scalar."""

    return (vector[0] * scale, vector[1] * scale, vector[2] * scale)


def distance(left: Vector3, right: Vector3) -> float:
    """Return Euclidean distance between two vectors."""

    return norm(subtract_vectors(left, right))


def norm(vector: Vector3) -> float:
    """Return Euclidean norm."""

    return math.sqrt(dot_product(vector, vector))


def normalize(vector: Vector3) -> Vector3:
    """Return a unit vector."""

    vector_norm = norm(vector)
    if vector_norm < 1.0e-15:
        msg = "cannot normalize a zero vector"
        raise ValueError(msg)
    return (vector[0] / vector_norm, vector[1] / vector_norm, vector[2] / vector_norm)


def dot_product(left: Vector3, right: Vector3) -> float:
    """Return vector dot product."""

    return left[0] * right[0] + left[1] * right[1] + left[2] * right[2]


def cross_product(left: Vector3, right: Vector3) -> Vector3:
    """Return vector cross product."""

    return (
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    )


def centroid(positions: tuple[Vector3, ...]) -> Vector3:
    """Return coordinate centroid."""

    count = len(positions)
    return (
        sum(position[0] for position in positions) / count,
        sum(position[1] for position in positions) / count,
        sum(position[2] for position in positions) / count,
    )


def matvec(matrix: Matrix3, vector: Vector3) -> Vector3:
    """Multiply a 3x3 matrix by a vector."""

    return tuple(dot_product(row, vector) for row in matrix)  # type: ignore[return-value]


def rotation_matrix(source: Vector3, target: Vector3) -> Matrix3:
    """Return a 3x3 matrix rotating ``source`` onto ``target``."""

    source_unit = normalize(source)
    target_unit = normalize(target)
    cross = cross_product(source_unit, target_unit)
    sine = norm(cross)
    cosine = dot_product(source_unit, target_unit)
    if sine < 1.0e-12:
        if cosine > 0.0:
            return ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
        orthogonal = normalize((1.0, 0.0, 0.0) if abs(source_unit[0]) < 0.9 else (0.0, 1.0, 0.0))
        cross = normalize(cross_product(source_unit, orthogonal))
        return axis_angle_matrix(cross, math.pi)
    skew = skew_matrix(cross)
    skew_squared = matrix_multiply(skew, skew)
    return matrix_add(
        matrix_add(((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)), skew),
        matrix_scale(skew_squared, (1.0 - cosine) / (sine * sine)),
    )


def axis_angle_matrix(axis: Vector3, angle_rad: float) -> Matrix3:
    """Return a rotation matrix for rotation around ``axis``."""

    skew = skew_matrix(axis)
    return matrix_add(
        matrix_add(
            matrix_scale(((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)), math.cos(angle_rad)),
            matrix_scale(skew, math.sin(angle_rad)),
        ),
        matrix_scale(outer_product(axis), 1.0 - math.cos(angle_rad)),
    )


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


def skew_matrix(vector: Vector3) -> Matrix3:
    """Return skew-symmetric matrix for a vector."""

    x, y, z = vector
    return ((0.0, -z, y), (z, 0.0, -x), (-y, x, 0.0))


def outer_product(vector: Vector3) -> Matrix3:
    """Return vector outer product with itself."""

    return tuple(
        tuple(vector[row] * vector[column] for column in range(3)) for row in range(3)
    )  # type: ignore[return-value]


def matrix_multiply(left: Matrix3, right: Matrix3) -> Matrix3:
    """Multiply two 3x3 matrices."""

    columns = tuple((right[0][i], right[1][i], right[2][i]) for i in range(3))
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
