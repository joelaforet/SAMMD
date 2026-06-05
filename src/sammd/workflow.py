"""Lightweight workflow utilities shared by SAMMD smoke drivers."""

from __future__ import annotations

import math
import shutil
from dataclasses import dataclass
from pathlib import Path

from sammd.config import SAMMDConfig, load_config

ETHANOL_MASS_G_MOL = 46.06844
ETHANOL_DENSITY_G_ML = 0.789
CANONICAL_SMOKE_SOLVENT_NAME = "ethanol"
CANONICAL_SMOKE_SOLVENT_RESIDUE_NAME = "EOH"
CANONICAL_SMOKE_SOLVENT_SMILES = "CCO"
DEFAULT_SMOKE_OUTPUT_DIR = "outputs/openmm_smoke/pd111_propanethiol_cinnamaldehyde"


@dataclass(frozen=True)
class SmokePaths:
    """Resolved files written by the OpenMM smoke workflow."""

    output_dir: Path
    build_config: Path
    topology: Path
    minimized_positions: Path
    final_positions: Path
    trajectory: Path
    thermodynamics: Path
    checkpoint: Path
    state_xml: Path
    system_xml: Path
    anchor_metadata: Path
    summary: Path
    packmol_dir: Path


@dataclass(frozen=True)
class RunSchedule:
    """Resolved integration and reporting schedule."""

    requested_duration_ns: float
    simulated_duration_ns: float
    total_steps: int
    report_interval: int
    frames: int
    timestep_fs: float


def smoke_paths(output_dir: Path) -> SmokePaths:
    """Resolve deterministic OpenMM smoke output paths.

    Parameters
    ----------
    output_dir
        Directory that will contain smoke workflow artifacts.

    Returns
    -------
    SmokePaths
        Bundle of artifact paths used by the smoke workflow.
    """

    return SmokePaths(
        output_dir=output_dir,
        build_config=output_dir / "build_config.yaml",
        topology=output_dir / "topology.cif",
        minimized_positions=output_dir / "minimized_positions.cif",
        final_positions=output_dir / "final_positions.cif",
        trajectory=output_dir / "trajectory.dcd",
        thermodynamics=output_dir / "thermodynamics.csv",
        checkpoint=output_dir / "checkpoint.chk",
        state_xml=output_dir / "state.xml",
        system_xml=output_dir / "system.xml",
        anchor_metadata=output_dir / "anchor_metadata.json",
        summary=output_dir / "smoke_summary.json",
        packmol_dir=output_dir / "packmol",
    )


def prepare_outputs(paths: SmokePaths, *, overwrite: bool) -> None:
    """Create output directory and enforce safe overwrite semantics.

    Parameters
    ----------
    paths
        Artifact paths to prepare.
    overwrite
        Whether existing artifacts should be removed before running.
    """

    paths.output_dir.mkdir(parents=True, exist_ok=True)
    for path in paths.__dict__.values():
        if path == paths.output_dir:
            continue
        if path.exists() and not overwrite:
            raise FileExistsError(f"refusing to overwrite existing smoke output: {path}")
        if path.exists() and overwrite:
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
    metrics_path = paths.output_dir / "smoke_metrics.csv"
    if metrics_path.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite existing smoke output: {metrics_path}")
    if metrics_path.exists() and overwrite:
        metrics_path.unlink()


def canonical_smoke_config(
    *,
    lateral_size_nm: float,
    solvent_padding_nm: float,
    seed: int,
    timestep_fs: float,
    reporter_interval_steps: int,
) -> SAMMDConfig:
    """Return the canonical Pd(111)/propanethiol/cinnamaldehyde/ethanol config.

    Parameters
    ----------
    lateral_size_nm
        Square slab lateral dimension in nanometers.
    solvent_padding_nm
        Solvent padding thickness in nanometers.
    seed
        Deterministic workflow seed.
    timestep_fs
        MD timestep in femtoseconds.
    reporter_interval_steps
        Reporter interval in integration steps.

    Returns
    -------
    SAMMDConfig
        Validated canonical smoke configuration.
    """

    return SAMMDConfig.model_validate(
        {
            "surface": {
                "slab": {
                    "lateral_size_nm": [lateral_size_nm, lateral_size_nm],
                }
            },
            "solvent": {
                "padding_nm": solvent_padding_nm,
                "components": [
                    {
                        "name": CANONICAL_SMOKE_SOLVENT_NAME,
                        "smiles": CANONICAL_SMOKE_SOLVENT_SMILES,
                        "mole_fraction": 1.0,
                        "density_g_ml": ETHANOL_DENSITY_G_ML,
                        "molar_mass_g_mol": ETHANOL_MASS_G_MOL,
                    }
                ],
            },
            "reporters": {"interval_steps": reporter_interval_steps},
            "simulation": {
                "seed": seed,
                "timestep_fs": timestep_fs,
            },
        }
    )


def load_smoke_config(
    config_path: Path | None,
    *,
    lateral_size_nm: float,
    solvent_padding_nm: float,
    seed: int,
    timestep_fs: float,
    reporter_interval_steps: int,
) -> SAMMDConfig:
    """Load a user config or create the canonical smoke config.

    Parameters
    ----------
    config_path
        Optional user-provided YAML config path.
    lateral_size_nm
        Square slab lateral dimension in nanometers for the default config.
    solvent_padding_nm
        Solvent padding thickness in nanometers for the default config.
    seed
        Deterministic workflow seed for the default config.
    timestep_fs
        MD timestep in femtoseconds for the default config.
    reporter_interval_steps
        Reporter interval for the default config.

    Returns
    -------
    SAMMDConfig
        Loaded or generated smoke configuration.
    """

    if config_path is not None:
        return load_config(config_path)
    return canonical_smoke_config(
        lateral_size_nm=lateral_size_nm,
        solvent_padding_nm=solvent_padding_nm,
        seed=seed,
        timestep_fs=timestep_fs,
        reporter_interval_steps=reporter_interval_steps,
    )


def resolve_run_schedule(
    *,
    duration_ns: float,
    timestep_fs: float,
    steps: int | None,
    frames: int,
    report_interval: int | None,
) -> RunSchedule:
    """Resolve integer MD steps and reporter cadence for a target frame count.

    Parameters
    ----------
    duration_ns
        Requested simulation duration in nanoseconds.
    timestep_fs
        Integration timestep in femtoseconds.
    steps
        Optional explicit integration step count.
    frames
        Requested trajectory frame count when deriving reporter cadence.
    report_interval
        Optional explicit reporter interval in integration steps.

    Returns
    -------
    RunSchedule
        Resolved step and reporter schedule.
    """

    if report_interval is not None:
        total_steps = (
            steps if steps is not None else max(1, round(duration_ns * 1.0e6 / timestep_fs))
        )
        resolved_frames = total_steps // report_interval
        return RunSchedule(
            requested_duration_ns=duration_ns,
            simulated_duration_ns=total_steps * timestep_fs / 1.0e6,
            total_steps=total_steps,
            report_interval=report_interval,
            frames=resolved_frames,
            timestep_fs=timestep_fs,
        )

    if steps is not None:
        total_steps = steps
        report_interval = max(1, round(steps / frames))
        resolved_frames = total_steps // report_interval
    else:
        requested_steps = max(1, round(duration_ns * 1.0e6 / timestep_fs))
        report_interval = max(1, math.ceil(requested_steps / frames))
        total_steps = report_interval * frames
        resolved_frames = frames
    return RunSchedule(
        requested_duration_ns=duration_ns,
        simulated_duration_ns=total_steps * timestep_fs / 1.0e6,
        total_steps=total_steps,
        report_interval=report_interval,
        frames=resolved_frames,
        timestep_fs=timestep_fs,
    )
