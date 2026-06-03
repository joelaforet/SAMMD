"""Tests for lightweight reporter configuration."""

import pytest

from sammd.reporting import (
    SUPPORTED_THERMODYNAMIC_FIELDS,
    build_state_data_reporter_config,
    get_reporter_fields,
)


def test_reporter_field_mapping() -> None:
    """Map SAMMD field names to OpenMM StateDataReporter keyword names."""

    config = build_state_data_reporter_config(
        "thermodynamics.csv",
        interval_steps=100,
        fields=["step", "time", "potential_energy", "elapsed_time"],
    )
    assert config.file == "thermodynamics.csv"
    assert config.report_interval == 100
    assert config.kwargs == {
        "step": True,
        "time": True,
        "potentialEnergy": True,
        "elapsedTime": True,
    }


def test_reporter_test_all_fields_mode() -> None:
    """Request every supported reporter field for test coverage."""

    fields = get_reporter_fields(["step"], test_all_fields=True)
    assert fields == list(SUPPORTED_THERMODYNAMIC_FIELDS)
    config = build_state_data_reporter_config(
        "thermodynamics.csv",
        interval_steps=100,
        fields=["step"],
        test_all_fields=True,
    )
    assert len(config.kwargs) == len(SUPPORTED_THERMODYNAMIC_FIELDS)


def test_reporter_invalid_inputs_fail_clearly() -> None:
    """Reject unsupported reporter fields and invalid intervals."""

    with pytest.raises(ValueError, match="unsupported reporter fields"):
        get_reporter_fields(["bad"])
    with pytest.raises(ValueError, match="interval_steps must be positive"):
        build_state_data_reporter_config("thermodynamics.csv", 0, ["step"])
