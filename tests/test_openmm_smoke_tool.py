"""Tests for the optional real-system OpenMM smoke runner."""

from __future__ import annotations

import importlib.util
import sys
from argparse import Namespace
from pathlib import Path

import pytest

from sammd.builders import build_system
from sammd.config import SAMMDConfig


def load_smoke_tool():
    """Load the tools/openmm_smoke.py module without requiring tools as a package."""

    path = Path(__file__).resolve().parents[1] / "tools" / "openmm_smoke.py"
    spec = importlib.util.spec_from_file_location("sammd_openmm_smoke_tool", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_smoke_tool_import_does_not_import_heavy_science_modules() -> None:
    """Keep the smoke tool importable before the science environment is active."""

    sys.modules.pop("openmm", None)
    sys.modules.pop("rdkit", None)

    load_smoke_tool()

    assert "openmm" not in sys.modules
    assert "rdkit" not in sys.modules


def test_auto_solvent_count_uses_sammd_solution_plan() -> None:
    """Auto solvent count should use the SAMMD solution composition plan."""

    smoke = load_smoke_tool()
    config = SAMMDConfig.model_validate(
        {
            "surface": {"slab": {"lateral_size_nm": [2.0, 2.0]}},
            "solvent": {
                "components": [
                    {
                        "name": "ethanol",
                        "smiles": "CCO",
                        "mole_fraction": 1.0,
                        "density_g_ml": smoke.ETHANOL_DENSITY_G_ML,
                        "molar_mass_g_mol": smoke.ETHANOL_MASS_G_MOL,
                    }
                ]
            },
        }
    )
    plan = build_system(config)

    count = smoke.resolve_solvent_count("auto", plan, "ethanol")
    expected = plan.solution.solvent_components[0].count

    assert count == expected


def test_default_run_schedule_records_300_frames_with_2fs_timestep() -> None:
    """Default smoke schedule should prioritize the requested trajectory frame count."""

    smoke = load_smoke_tool()

    schedule = smoke.resolve_run_schedule(
        duration_ns=5.0,
        timestep_fs=2.0,
        steps=None,
        frames=300,
        report_interval=None,
    )

    assert schedule.total_steps == 2500200
    assert schedule.report_interval == 8334
    assert schedule.frames == 300
    assert schedule.simulated_duration_ns == pytest.approx(5.0004)


def test_validate_args_rejects_invalid_solvent_count() -> None:
    """Catch invalid smoke-only CLI values before optional imports."""

    smoke = load_smoke_tool()
    args = Namespace(
        lateral_size_nm=2.0,
        solvent_padding_nm=3.0,
        timestep_fs=0.5,
        friction_per_ps=1.0,
        pd_s_sigma_angstrom=2.2,
        pd_s_epsilon_kcal_mol=1.0,
        duration_ns=5.0,
        sulfur_height_nm=0.0,
        seed=1,
        steps=1,
        frames=300,
        minimize_iterations=1,
        report_interval=1,
        reactant_count=None,
        solvent_count="zero",
        water_count=None,
    )

    with pytest.raises(SystemExit, match="--solvent-count"):
        smoke.validate_args(args)


def test_validate_args_rejects_zero_steps() -> None:
    """Require explicit smoke step overrides to run at least one step."""

    smoke = load_smoke_tool()
    args = Namespace(
        lateral_size_nm=2.0,
        solvent_padding_nm=3.0,
        timestep_fs=0.5,
        friction_per_ps=1.0,
        pd_s_sigma_angstrom=2.2,
        pd_s_epsilon_kcal_mol=1.0,
        duration_ns=5.0,
        sulfur_height_nm=0.0,
        seed=1,
        steps=0,
        frames=300,
        minimize_iterations=1,
        report_interval=1,
        reactant_count=None,
        solvent_count="auto",
        water_count=None,
    )

    with pytest.raises(SystemExit, match="--steps must be positive"):
        smoke.validate_args(args)
