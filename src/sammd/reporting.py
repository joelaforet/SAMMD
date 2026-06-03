"""Reporter configuration helpers without importing OpenMM."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

THERMODYNAMIC_FIELD_TO_OPENMM: dict[str, str] = {
    "step": "step",
    "time": "time",
    "potential_energy": "potentialEnergy",
    "kinetic_energy": "kineticEnergy",
    "total_energy": "totalEnergy",
    "temperature": "temperature",
    "volume": "volume",
    "density": "density",
    "speed": "speed",
    "elapsed_time": "elapsedTime",
    "remaining_time": "remainingTime",
    "progress": "progress",
}

SUPPORTED_THERMODYNAMIC_FIELDS: tuple[str, ...] = tuple(THERMODYNAMIC_FIELD_TO_OPENMM)


@dataclass(frozen=True)
class StateDataReporterConfig:
    """OpenMM StateDataReporter-compatible keyword configuration."""

    file: str
    report_interval: int
    kwargs: dict[str, bool | int | str]


def get_reporter_fields(fields: list[str], test_all_fields: bool = False) -> list[str]:
    """Resolve requested thermodynamic reporter fields.

    Parameters
    ----------
    fields
        User-requested SAMMD field names.
    test_all_fields
        Whether to request every supported field.

    Returns
    -------
    list[str]
        Resolved field list.
    """

    resolved = list(SUPPORTED_THERMODYNAMIC_FIELDS) if test_all_fields else fields
    unknown = sorted(set(resolved) - set(SUPPORTED_THERMODYNAMIC_FIELDS))
    if unknown:
        msg = f"unsupported reporter fields: {', '.join(unknown)}"
        raise ValueError(msg)
    if len(set(resolved)) != len(resolved):
        msg = "reporter fields must not contain duplicates"
        raise ValueError(msg)
    return resolved


def build_state_data_reporter_config(
    output_path: str,
    interval_steps: int,
    fields: list[str],
    test_all_fields: bool = False,
    total_steps: int | None = None,
) -> StateDataReporterConfig:
    """Build OpenMM StateDataReporter keyword arguments.

    Parameters
    ----------
    output_path
        Reporter output path.
    interval_steps
        Number of MD steps between reports.
    fields
        User-requested SAMMD thermodynamic field names.
    test_all_fields
        Whether to request every supported field.
    total_steps
        Total expected simulation steps, required for progress and remaining time.

    Returns
    -------
    StateDataReporterConfig
        Lightweight reporter configuration object.
    """

    if interval_steps <= 0:
        msg = "interval_steps must be positive"
        raise ValueError(msg)
    if total_steps is not None and total_steps <= 0:
        msg = "total_steps must be positive"
        raise ValueError(msg)
    resolved_fields = get_reporter_fields(fields, test_all_fields=test_all_fields)
    fields_requiring_total_steps = {"progress", "remaining_time"}
    if fields_requiring_total_steps.intersection(resolved_fields) and total_steps is None:
        msg = "progress and remaining_time reporter fields require total_steps"
        raise ValueError(msg)
    kwargs: dict[str, bool | int | str] = {
        THERMODYNAMIC_FIELD_TO_OPENMM[field]: True for field in resolved_fields
    }
    if total_steps is not None:
        kwargs["totalSteps"] = total_steps
    return StateDataReporterConfig(
        file=output_path,
        report_interval=interval_steps,
        kwargs=kwargs,
    )


def create_openmm_reporters(
    reporting_config: Any,
    output_paths: Any,
    *,
    total_steps: int | None = None,
    app_module: Any | None = None,
) -> list[Any]:
    """Create OpenMM runtime reporter objects lazily.

    Parameters
    ----------
    reporting_config
        Reporter config with interval, field, and test-all-fields attributes.
    output_paths
        Resolved output paths with trajectory and thermodynamics attributes.
    total_steps
        Total expected simulation steps, required for progress and remaining time.
    app_module
        Optional injected OpenMM app-like module for tests.

    Returns
    -------
    list[Any]
        OpenMM reporter instances for trajectory and thermodynamic reporting.
    """

    app = app_module if app_module is not None else _import_openmm_app()
    interval_steps = reporting_config.interval_steps
    _prepare_reporter_output_directories(output_paths)
    state_data_config = build_state_data_reporter_config(
        str(output_paths.thermodynamics),
        interval_steps=interval_steps,
        fields=reporting_config.fields,
        test_all_fields=reporting_config.test_all_fields,
        total_steps=total_steps,
    )
    dcd_reporter = app.DCDReporter(str(output_paths.trajectory), interval_steps)
    state_data_reporter = app.StateDataReporter(
        state_data_config.file,
        state_data_config.report_interval,
        separator=",",
        **state_data_config.kwargs,
    )
    return [dcd_reporter, state_data_reporter]


def _prepare_reporter_output_directories(output_paths: Any) -> None:
    """Create runtime reporter output directories before OpenMM opens files."""

    for path in (output_paths.trajectory, output_paths.thermodynamics):
        Path(path).parent.mkdir(parents=True, exist_ok=True)


def _import_openmm_app() -> Any:
    """Import OpenMM application reporters only when runtime construction is requested."""

    try:
        from openmm import app
    except ImportError as error:
        msg = (
            "OpenMM is required to construct runtime reporters. Install OpenMM to use "
            "create_openmm_reporters."
        )
        raise ImportError(msg) from error
    return app
