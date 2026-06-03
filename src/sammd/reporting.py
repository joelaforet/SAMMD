"""Reporter configuration helpers without importing OpenMM."""

from __future__ import annotations

from dataclasses import dataclass

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

    Returns
    -------
    StateDataReporterConfig
        Lightweight reporter configuration object.
    """

    if interval_steps <= 0:
        msg = "interval_steps must be positive"
        raise ValueError(msg)
    resolved_fields = get_reporter_fields(fields, test_all_fields=test_all_fields)
    kwargs: dict[str, bool | int | str] = {
        THERMODYNAMIC_FIELD_TO_OPENMM[field]: True for field in resolved_fields
    }
    return StateDataReporterConfig(
        file=output_path,
        report_interval=interval_steps,
        kwargs=kwargs,
    )
