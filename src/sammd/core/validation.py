"""Dependency-free validation gates for lightweight SAMMD build plans."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite
from pathlib import Path
from typing import Any

from sammd.model.metal_sulfur import default_metal_sulfur_interaction

APPROX_TOLERANCE = 1.0e-6


@dataclass(frozen=True)
class ValidationGateResult:
    """Outcome for one dependency-free validation gate."""

    gate_id: str
    passed: bool
    severity: str
    message: str
    details: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class ValidationReport:
    """Collection of validation gate outcomes for one subject."""

    subject: str
    gates: tuple[ValidationGateResult, ...]

    @property
    def passed(self) -> bool:
        """Return whether all gates passed."""

        return all(gate.passed for gate in self.gates)

    def to_summary(self) -> dict[str, object]:
        """Return a JSON-serializable validation summary."""

        return {
            "subject": self.subject,
            "passed": self.passed,
            "gates": [
                {
                    "gate_id": gate.gate_id,
                    "passed": gate.passed,
                    "severity": gate.severity,
                    "message": gate.message,
                    "details": gate.details,
                }
                for gate in self.gates
            ],
        }


def validate_build_plan(plan: Any) -> ValidationReport:
    """Validate the current lightweight build-plan contract without backend imports."""

    gates = [
        _validate_surface_atom_counts(plan),
        _validate_binding_sites(plan),
        _validate_sam_count(plan),
        _validate_solution_volume(plan),
        _validate_box_consistency(plan),
        _validate_box_lateral_size(plan),
        _validate_sam_anchor_metadata(plan),
        _validate_metal_s_pair_counts(plan),
        _validate_metal_s_strategy_metadata(plan),
    ]
    return ValidationReport(subject="build_plan", gates=tuple(gates))


def validate_output_paths(paths: Any) -> ValidationReport:
    """Validate configured current and reserved output path suffixes.

    Missing optional paths are not failures because future backend artifacts are
    reserved but not required by the current lightweight builder.
    """

    expected_suffixes = {
        "sam_grafting_density": ".cif",
        "solvated_system": ".cif",
        "openff_interchange": ".json",
        "anchor_metadata": ".json",
        "build_summary": ".json",
        "resolved_config": ".yaml",
        "trajectory": ".dcd",
        "thermodynamics": ".csv",
    }
    failures: dict[str, str] = {}
    checked: dict[str, str | None] = {}
    for field_name, expected_suffix in expected_suffixes.items():
        value = getattr(paths, field_name, None)
        checked[field_name] = None if value is None else str(value)
        if value is None:
            continue
        suffix = Path(value).suffix.lower()
        if suffix != expected_suffix:
            failures[field_name] = f"expected {expected_suffix}, found {suffix or 'no suffix'}"
    gate = _gate(
        "output_path_suffixes",
        not failures,
        "output paths use expected current/reserved suffixes",
        "output paths must use expected current/reserved suffixes",
        {"checked": checked, "failures": failures},
    )
    return ValidationReport(subject="output_paths", gates=(gate,))


def validate_topology_cif_text(
    text: str,
    expected_atom_count: int | None = None,
    expected_box_nm: tuple[float, float, float] | None = None,
) -> ValidationReport:
    """Validate lightweight topology CIF text emitted by the current builder."""

    atom_count = sum(1 for line in text.splitlines() if line.startswith("HETATM "))
    gates = [
        _gate(
            "topology_cif_atom_count",
            expected_atom_count is None or atom_count == expected_atom_count,
            "topology CIF atom count matches expectation",
            "topology CIF atom count does not match expectation",
            {"actual": atom_count, "expected": expected_atom_count},
        )
    ]

    cell_lengths_nm = _parse_cell_lengths_nm(text)
    cell_passed = True
    if expected_box_nm is not None:
        cell_passed = cell_lengths_nm is not None and all(
            _approx_equal(actual, expected)
            for actual, expected in zip(cell_lengths_nm, expected_box_nm, strict=True)
        )
    gates.append(
        _gate(
            "topology_cif_cell_lengths",
            cell_passed,
            "topology CIF cell lengths match expectation",
            "topology CIF cell lengths do not match expectation",
            {"actual_nm": cell_lengths_nm, "expected_nm": expected_box_nm},
        )
    )
    return ValidationReport(subject="topology_cif_text", gates=tuple(gates))


def _validate_surface_atom_counts(plan: Any) -> ValidationGateResult:
    slab = plan.slab
    lengths = {
        "positions_nm": len(slab.positions_nm),
        "labels": len(slab.labels),
        "layer_indices": len(slab.layer_indices),
    }
    passed = len(set(lengths.values())) == 1 and lengths["positions_nm"] > 0
    return _gate(
        "surface_atom_count_consistency",
        passed,
        "surface atom positions, labels, and layer indices are consistent",
        "surface atom positions, labels, and layer indices must have equal non-zero lengths",
        lengths,
    )


def _validate_binding_sites(plan: Any) -> ValidationGateResult:
    counts = {"top": 0, "bottom": 0, "invalid": 0}
    for site in plan.binding_sites:
        if site.side in {"top", "bottom"}:
            counts[site.side] += 1
        else:
            counts["invalid"] += 1
    passed = counts["top"] > 0 and counts["bottom"] > 0 and counts["invalid"] == 0
    return _gate(
        "binding_sites_nonempty_sides",
        passed,
        "binding sites include non-empty top and bottom faces",
        "binding sites must include non-empty top and bottom faces with valid side labels",
        counts,
    )


def _validate_sam_count(plan: Any) -> ValidationGateResult:
    expected = 2 * plan.sam_placements.selected_sites_per_side
    actual = len(plan.sam_placements.placements)
    return _gate(
        "sam_count_selected_sites",
        actual == expected,
        "SAM placement count equals two faces times selected sites per side",
        "SAM placement count must equal two faces times selected sites per side",
        {"actual": actual, "expected": expected},
    )


def _validate_solution_volume(plan: Any) -> ValidationGateResult:
    actual = plan.solution.box_volume_nm3
    expected = plan.box_plan.volume_nm3
    return _gate(
        "solution_volume_matches_box",
        _approx_equal(actual, expected),
        "solution count volume matches box-plan volume",
        "solution count volume must match box-plan volume",
        {"solution_volume_nm3": actual, "box_volume_nm3": expected},
    )


def _validate_box_consistency(plan: Any) -> ValidationGateResult:
    box = plan.box_plan
    dimensions = tuple(box.dimensions_nm)
    bounds = tuple(tuple(axis_bounds) for axis_bounds in box.bounds_nm)
    finite_positive_dimensions = len(dimensions) == 3 and all(
        _finite_positive(dimension) for dimension in dimensions
    )
    valid_bounds = len(bounds) == 3 and all(
        len(axis_bounds) == 2
        and _finite(axis_bounds[0])
        and _finite(axis_bounds[1])
        and axis_bounds[1] > axis_bounds[0]
        for axis_bounds in bounds
    )
    bounds_dimensions_match = valid_bounds and len(dimensions) == 3 and all(
        _approx_equal(axis_bounds[1] - axis_bounds[0], dimension)
        for axis_bounds, dimension in zip(bounds, dimensions, strict=True)
    )
    expected_volume = (
        dimensions[0] * dimensions[1] * dimensions[2] if len(dimensions) == 3 else None
    )
    volume_passed = (
        expected_volume is not None
        and _finite_positive(box.volume_nm3)
        and _approx_equal(box.volume_nm3, expected_volume)
    )
    passed = (
        finite_positive_dimensions and valid_bounds and bounds_dimensions_match and volume_passed
    )
    return _gate(
        "box_dimensions_bounds_volume",
        passed,
        "box dimensions, bounds, and volume are finite, positive, and consistent",
        "box dimensions, bounds, and volume must be finite, positive, and consistent",
        {
            "dimensions_nm": dimensions,
            "bounds_nm": bounds,
            "volume_nm3": box.volume_nm3,
            "expected_volume_nm3": expected_volume,
        },
    )


def _validate_box_lateral_size(plan: Any) -> ValidationGateResult:
    return _gate(
        "box_lateral_matches_slab",
        all(
            _approx_equal(actual, expected)
            for actual, expected in zip(
                plan.box_plan.lateral_size_nm, plan.slab.lateral_size_nm, strict=True
            )
        ),
        "box lateral size matches slab lateral size",
        "box lateral size must match slab lateral size",
        {
            "box_lateral_size_nm": tuple(plan.box_plan.lateral_size_nm),
            "slab_lateral_size_nm": tuple(plan.slab.lateral_size_nm),
        },
    )


def _validate_sam_anchor_metadata(plan: Any) -> ValidationGateResult:
    failures: list[dict[str, object]] = []
    for index, placement in enumerate(plan.sam_placements.placements):
        pose = placement.anchor_pose
        metadata = placement.anchor_metadata
        if not _finite_positive(pose.sulfur_height_nm):
            failures.append({"index": index, "field": "sulfur_height_nm"})
        if not placement.site_kind or not pose.site_kind or not metadata.get("site"):
            failures.append({"index": index, "field": "site_kind"})
        if not pose.attachment_mode or not metadata.get("mode"):
            failures.append({"index": index, "field": "attachment_mode"})
        if not all(_finite(coordinate) for coordinate in pose.sulfur_position_nm):
            failures.append({"index": index, "field": "sulfur_position_nm"})
    return _gate(
        "sam_anchor_metadata",
        not failures,
        "SAM anchor metadata contains finite sulfur positions and required labels",
        "SAM anchor metadata must contain finite sulfur positions and required labels",
        {"failures": failures},
    )


def _validate_metal_s_pair_counts(plan: Any) -> ValidationGateResult:
    expected_pairs = default_metal_sulfur_interaction().pairs_per_anchor
    slab_atom_count = len(plan.slab.positions_nm)
    failures: list[dict[str, object]] = []
    for index, placement in enumerate(plan.sam_placements.placements):
        pairs = tuple(placement.anchor_pose.nearest_metal_atom_indices)
        if len(pairs) != expected_pairs:
            failures.append({"index": index, "actual_pair_count": len(pairs)})
        invalid_indices = [
            metal_index
            for metal_index in pairs
            if not isinstance(metal_index, int) or metal_index < 0 or metal_index >= slab_atom_count
        ]
        if invalid_indices:
            failures.append({"index": index, "invalid_indices": invalid_indices})
    return _gate(
        "metal_s_pair_count_and_indices",
        not failures,
        "metal-S pair metadata has expected pair counts and slab atom indices",
        "metal-S pair metadata must have expected pair counts and slab atom indices",
        {
            "expected_pairs_per_anchor": expected_pairs,
            "slab_atom_count": slab_atom_count,
            "failures": failures,
        },
    )


def _validate_metal_s_strategy_metadata(plan: Any) -> ValidationGateResult:
    expected = default_metal_sulfur_interaction().to_summary()
    failures: list[dict[str, object]] = []
    for index, placement in enumerate(plan.sam_placements.placements):
        pose_summary = placement.anchor_pose.metal_sulfur_interaction.to_summary()
        metadata_summary = placement.anchor_metadata.get("metal_sulfur_interaction")
        if pose_summary != expected:
            failures.append({"index": index, "field": "anchor_pose"})
        if metadata_summary != expected:
            failures.append({"index": index, "field": "anchor_metadata"})
    return _gate(
        "metal_s_default_strategy_metadata",
        not failures,
        "metal-S metadata matches the default strategy summary",
        "metal-S metadata must match the default strategy summary",
        {"expected": expected, "failures": failures},
    )


def _parse_cell_lengths_nm(text: str) -> tuple[float, float, float] | None:
    values: dict[str, float] = {}
    keys = {
        "_cell.length_a": "a",
        "_cell.length_b": "b",
        "_cell.length_c": "c",
    }
    for line in text.splitlines():
        parts = line.split()
        if len(parts) != 2 or parts[0] not in keys:
            continue
        try:
            values[keys[parts[0]]] = float(parts[1]) / 10.0
        except ValueError:
            return None
    if set(values) != {"a", "b", "c"}:
        return None
    return (values["a"], values["b"], values["c"])


def _gate(
    gate_id: str,
    passed: bool,
    passed_message: str,
    failed_message: str,
    details: dict[str, object],
) -> ValidationGateResult:
    return ValidationGateResult(
        gate_id=gate_id,
        passed=passed,
        severity="error",
        message=passed_message if passed else failed_message,
        details=details,
    )


def _finite(value: object) -> bool:
    return isinstance(value, int | float) and isfinite(value)


def _finite_positive(value: object) -> bool:
    return _finite(value) and value > 0


def _approx_equal(left: float, right: float) -> bool:
    return abs(left - right) <= APPROX_TOLERANCE
