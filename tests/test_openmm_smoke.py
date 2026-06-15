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
