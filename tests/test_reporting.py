"""Tests for lightweight reporter configuration."""

import builtins
import sys

import pytest

from sammd.config import ReporterConfig
from sammd.io import OutputPaths
from sammd.reporting import (
    SUPPORTED_THERMODYNAMIC_FIELDS,
    build_state_data_reporter_config,
    create_openmm_reporters,
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
        total_steps=1000,
    )
    assert len(config.kwargs) == len(SUPPORTED_THERMODYNAMIC_FIELDS) + 1
    assert config.kwargs["totalSteps"] == 1000


def test_reporter_progress_requires_total_steps() -> None:
    """Require total steps for OpenMM progress and remaining-time fields."""

    with pytest.raises(ValueError, match="require total_steps"):
        build_state_data_reporter_config(
            "thermodynamics.csv",
            interval_steps=100,
            fields=["progress"],
        )

    with pytest.raises(ValueError, match="require total_steps"):
        build_state_data_reporter_config(
            "thermodynamics.csv",
            interval_steps=100,
            fields=["step"],
            test_all_fields=True,
        )


def test_reporter_invalid_inputs_fail_clearly() -> None:
    """Reject unsupported reporter fields and invalid intervals."""

    with pytest.raises(ValueError, match="unsupported reporter fields"):
        get_reporter_fields(["bad"])
    with pytest.raises(ValueError, match="interval_steps must be positive"):
        build_state_data_reporter_config("thermodynamics.csv", 0, ["step"])


def test_reporter_module_does_not_import_openmm_at_import_time() -> None:
    """Keep reporter configuration importable without OpenMM runtime modules."""

    assert "openmm" not in sys.modules


def test_openmm_reporter_runtime_helper_errors_when_unavailable(monkeypatch, tmp_path) -> None:
    """Raise an installation-focused error when runtime reporter construction is requested."""

    real_import = builtins.__import__

    def fake_import(name, globals_=None, locals_=None, fromlist=(), level=0):
        """Block OpenMM imports while preserving all other imports."""

        if name == "openmm":
            raise ImportError("blocked OpenMM")
        return real_import(name, globals_, locals_, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    output_paths = OutputPaths(
        topology=tmp_path / "topology.cif",
        trajectory=tmp_path / "trajectory.dcd",
        thermodynamics=tmp_path / "thermodynamics.csv",
    )

    with pytest.raises(ImportError, match="OpenMM is required to construct runtime reporters"):
        create_openmm_reporters(ReporterConfig(), output_paths)


def test_openmm_reporter_runtime_helper_uses_expected_reporter_arguments(tmp_path) -> None:
    """Create DCD and StateData reporters with deterministic OpenMM arguments."""

    class FakeApp:
        """Minimal OpenMM app reporter namespace for injection tests."""

        class DCDReporter:
            """Fake DCD reporter storing constructor arguments."""

            def __init__(self, file, report_interval):
                self.file = file
                self.report_interval = report_interval

        class StateDataReporter:
            """Fake state data reporter storing constructor arguments."""

            def __init__(self, file, report_interval, **kwargs):
                self.file = file
                self.report_interval = report_interval
                self.kwargs = kwargs

    output_paths = OutputPaths(
        topology=tmp_path / "topology.cif",
        trajectory=tmp_path / "trajectory.dcd",
        thermodynamics=tmp_path / "thermodynamics.csv",
    )
    config = ReporterConfig(interval_steps=250, fields=["step", "temperature", "progress"])

    reporters = create_openmm_reporters(
        config,
        output_paths,
        total_steps=1000,
        app_module=FakeApp,
    )

    assert reporters[0].file == str(tmp_path / "trajectory.dcd")
    assert reporters[0].report_interval == 250
    assert reporters[1].file == str(tmp_path / "thermodynamics.csv")
    assert reporters[1].report_interval == 250
    assert reporters[1].kwargs == {
        "step": True,
        "temperature": True,
        "progress": True,
        "totalSteps": 1000,
        "separator": ",",
    }
