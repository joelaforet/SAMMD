"""Tests for lightweight workflow helpers."""

from pathlib import Path

import pytest

from sammd.workflow import (
    CANONICAL_SMOKE_SOLVENT_NAME,
    ETHANOL_DENSITY_G_ML,
    canonical_smoke_config,
    load_smoke_config,
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


def test_prepare_outputs_removes_directories_and_metrics_when_overwriting(
    tmp_path: Path,
) -> None:
    """Remove nested smoke artifacts and metrics during explicit overwrite."""

    paths = smoke_paths(tmp_path / "smoke")
    paths.packmol_dir.mkdir(parents=True)
    (paths.packmol_dir / "input.inp").write_text("packmol", encoding="utf-8")
    metrics_path = paths.output_dir / "smoke_metrics.csv"
    metrics_path.write_text("metric", encoding="utf-8")

    prepare_outputs(paths, overwrite=True)

    assert not paths.packmol_dir.exists()
    assert not metrics_path.exists()


def test_prepare_outputs_rejects_existing_metrics_without_overwrite(tmp_path: Path) -> None:
    """Protect existing smoke metrics unless overwrite is explicit."""

    paths = smoke_paths(tmp_path / "smoke")
    paths.output_dir.mkdir(parents=True)
    metrics_path = paths.output_dir / "smoke_metrics.csv"
    metrics_path.write_text("metric", encoding="utf-8")

    with pytest.raises(FileExistsError, match=r"smoke_metrics\.csv"):
        prepare_outputs(paths, overwrite=False)


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


def test_load_smoke_config_reads_user_yaml(tmp_path: Path) -> None:
    """Load an explicit smoke YAML instead of generating the canonical config."""

    config_path = tmp_path / "sammd.yaml"
    config_path.write_text(
        "surface:\n  slab:\n    lateral_size_nm: [2.5, 2.5]\n",
        encoding="utf-8",
    )

    config = load_smoke_config(
        config_path,
        lateral_size_nm=2.0,
        solvent_padding_nm=3.0,
        seed=2026,
        timestep_fs=2.0,
        reporter_interval_steps=100,
    )

    assert config.surface.slab.lateral_size_nm == (2.5, 2.5)


def test_run_schedule_supports_explicit_steps_and_report_interval() -> None:
    """Resolve schedules from explicit smoke CLI step controls."""

    interval_schedule = resolve_run_schedule(
        duration_ns=1.0,
        timestep_fs=2.0,
        steps=1000,
        frames=60,
        report_interval=250,
    )
    steps_schedule = resolve_run_schedule(
        duration_ns=1.0,
        timestep_fs=2.0,
        steps=1000,
        frames=60,
        report_interval=None,
    )

    assert interval_schedule.frames == 4
    assert interval_schedule.simulated_duration_ns == pytest.approx(0.002)
    assert steps_schedule.report_interval == 17
    assert steps_schedule.frames == 58
