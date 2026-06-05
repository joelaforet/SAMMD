"""Tests for lightweight workflow helpers."""

from pathlib import Path

import pytest

from sammd.workflow import (
    CANONICAL_SMOKE_SOLVENT_NAME,
    ETHANOL_DENSITY_G_ML,
    canonical_smoke_config,
    prepare_outputs,
    resolve_run_schedule,
    smoke_paths,
)


def test_default_run_schedule_records_300_frames_with_2fs_timestep() -> None:
    """Default smoke schedule should prioritize the requested trajectory frame count."""

    schedule = resolve_run_schedule(
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


def test_smoke_paths_and_prepare_outputs_respect_overwrite(tmp_path: Path) -> None:
    """Prepare workflow outputs without clobbering existing artifacts by default."""

    paths = smoke_paths(tmp_path / "smoke")
    prepare_outputs(paths, overwrite=False)
    paths.summary.write_text("{}", encoding="utf-8")

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        prepare_outputs(paths, overwrite=False)

    prepare_outputs(paths, overwrite=True)

    assert paths.output_dir.exists()
    assert not paths.summary.exists()


def test_canonical_smoke_config_uses_ethanol_demo_system() -> None:
    """Build the canonical Pd(111)/propanethiol/cinnamaldehyde/ethanol config."""

    config = canonical_smoke_config(
        lateral_size_nm=2.0,
        solvent_padding_nm=3.0,
        seed=2026,
        timestep_fs=2.0,
        reporter_interval_steps=100,
    )

    solvent = config.solvent.components[0]
    assert config.surface.metal == "Pd"
    assert config.surface.facet == "111"
    assert config.sam.components[0].name == "propanethiol"
    assert config.reactants[0].name == "cinnamaldehyde"
    assert solvent.name == CANONICAL_SMOKE_SOLVENT_NAME
    assert solvent.density_g_ml == ETHANOL_DENSITY_G_ML
