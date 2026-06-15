"""Smoke-tool parity tests for runtime packing helpers."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest


def load_smoke_module() -> ModuleType:
    """Load the excluded temporary smoke tool from its source path."""

    module_path = Path(__file__).parents[1] / "temporary" / "openmm_smoke.py"
    spec = importlib.util.spec_from_file_location("openmm_smoke_test_module", module_path)
    if spec is None or spec.loader is None:
        msg = f"could not load smoke module from {module_path}"
        raise ImportError(msg)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_smoke_auto_solvent_count_allows_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    """Return the recomputed auto count exactly, including zero."""

    smoke = load_smoke_module()

    def plan_solution_composition(config: object, volume_nm3: float) -> SimpleNamespace:
        """Return a zero-count solvent component for auto-count parity."""

        assert volume_nm3 == pytest.approx(0.0)
        return SimpleNamespace(
            solvent_components=(SimpleNamespace(name="ethanol", count=0),),
        )

    monkeypatch.setattr(smoke, "plan_solution_composition", plan_solution_composition)

    count = smoke.resolve_solvent_count(
        "auto",
        SimpleNamespace(config=SimpleNamespace()),
        "ethanol",
        planning_volume_nm3=0.0,
    )

    assert count == 0


def test_smoke_explicit_zero_solvent_count_is_valid() -> None:
    """Allow explicit zero solvent count to reach the zero-solvent path."""

    smoke = load_smoke_module()

    args = SimpleNamespace(
        lateral_size_nm=2.0,
        solvent_padding_nm=3.0,
        timestep_fs=2.0,
        friction_per_ps=1.0,
        pd_s_sigma_angstrom=2.0,
        pd_s_epsilon_kcal_mol=1.0,
        duration_ns=1.0,
        sulfur_height_nm=0.18,
        seed=1,
        steps=1,
        minimize_iterations=0,
        frames=1,
        report_interval=1,
        reactant_count=None,
        solvent_count="0",
    )

    smoke.validate_args(args)


def test_smoke_fixed_solute_containment_accepts_tolerance() -> None:
    """Allow tiny numerical drift at runtime box boundaries."""

    smoke = load_smoke_module()

    smoke.ensure_positions_inside_box(
        ((-1.0e-10, 0.5, 1.0 + 1.0e-10),),
        (1.0, 1.0, 1.0),
        context="test fixed solute",
    )


def test_smoke_fixed_solute_containment_rejects_outside_box() -> None:
    """Reject shifted fixed-solute coordinates outside runtime dimensions."""

    smoke = load_smoke_module()

    with pytest.raises(
        ValueError,
        match=r"test fixed solute atom 2 z-coordinate 1\.1 nm lies outside runtime box",
    ):
        smoke.ensure_positions_inside_box(
            ((0.5, 0.5, 0.5), (0.5, 0.5, 1.1)),
            (1.0, 1.0, 1.0),
            context="test fixed solute",
        )


def test_smoke_solvent_clearance_uses_configured_packmol_tolerance() -> None:
    """Use non-default Packmol tolerance when sizing runtime solvent clearance."""

    smoke = load_smoke_module()
    plan = SimpleNamespace(
        config=SimpleNamespace(
            packing=SimpleNamespace(packmol=SimpleNamespace(tolerance=2.5)),
        ),
        box_plan=SimpleNamespace(dimensions_nm=(2.0, 2.0, 1.0)),
    )

    box, z_shift, regions = smoke.actual_solvent_packing_geometry(
        plan,
        ((0.5, 0.5, 0.0), (0.5, 0.5, 1.0)),
        2.0,
    )

    assert box == pytest.approx((2.0, 2.0, 3.0))
    assert z_shift == pytest.approx(1.0)
    assert regions[0][2] == pytest.approx((0.0, 0.75))
    assert regions[1][2] == pytest.approx((2.25, 3.0))


def test_smoke_zero_solvent_count_skips_packmol_after_containment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Return no solvent positions for zero counts without writing Packmol files."""

    smoke = load_smoke_module()

    def fail_if_packmol_runs(*args: object, **kwargs: object) -> None:
        """Fail if zero-count placement reaches Packmol execution."""

        raise AssertionError("Packmol should not run for zero solvent count")

    monkeypatch.setattr(smoke, "run_packmol", fail_if_packmol_runs)

    solution = smoke.pack_solution_with_packmol(
        topology=SimpleNamespace(),
        solute_positions_nm=((0.5, 0.5, 0.5),),
        solvent_template=SimpleNamespace(),
        solvent_count=0,
        box_dimensions_nm=(1.0, 1.0, 1.0),
        solvent_regions_nm=(),
        working_dir=tmp_path / "packmol",
    )

    assert solution.solvent_positions_nm == ()
    assert not (tmp_path / "packmol").exists()


def test_smoke_zero_solvent_count_still_validates_fixed_solute_containment(
    tmp_path: Path,
) -> None:
    """Reject invalid fixed solute coordinates before zero-count short-circuiting."""

    smoke = load_smoke_module()

    with pytest.raises(ValueError, match="fixed solute Packmol atom 1"):
        smoke.pack_solution_with_packmol(
            topology=SimpleNamespace(),
            solute_positions_nm=((1.5, 0.5, 0.5),),
            solvent_template=SimpleNamespace(),
            solvent_count=0,
            box_dimensions_nm=(1.0, 1.0, 1.0),
            solvent_regions_nm=(),
            working_dir=tmp_path / "packmol",
        )
