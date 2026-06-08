import logging
import re

import pytest

from sammd.colors import (
    ColoredFormatter,
    TerminalColorSupport,
    detect_color_support,
    get_color_support,
    set_color_support,
    setup_colored_logging,
)


@pytest.fixture(autouse=True)
def reset_color_support():
    set_color_support(None)
    yield
    set_color_support(None)


@pytest.fixture
def restore_root_logging():
    root = logging.getLogger()
    handlers = root.handlers[:]
    level = root.level
    yield
    root.handlers[:] = handlers
    root.setLevel(level)


def make_record(name: str, level: int, message: str = "message") -> logging.LogRecord:
    return logging.LogRecord(name, level, __file__, 1, message, (), None)


def test_no_color_environment_disables_color(monkeypatch):
    class TtyStream:
        def isatty(self) -> bool:
            return True

    monkeypatch.setenv("NO_COLOR", "1")

    assert detect_color_support(TtyStream()) is TerminalColorSupport.NONE


def test_setup_no_color_override_disables_cached_color(restore_root_logging):
    set_color_support(TerminalColorSupport.TRUECOLOR)

    setup_colored_logging(no_color=True)

    assert get_color_support() is TerminalColorSupport.NONE


def test_formatter_includes_timestamp_logger_level_and_message():
    set_color_support(TerminalColorSupport.NONE)

    output = ColoredFormatter().format(make_record("sammd.cli", logging.INFO, "hello"))

    assert re.match(
        r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3} - sammd\.cli - INFO - hello$",
        output,
    )


def test_module_info_color_when_forced():
    set_color_support(TerminalColorSupport.BASIC)

    output = ColoredFormatter().format(
        make_record("sammd.backends.openff.forcefield", logging.INFO)
    )

    assert output.startswith("\033[95m")
    assert output.endswith("\033[0m")
    assert "sammd.backends.openff.forcefield - INFO - message" in output


def test_warning_and_error_use_level_override_colors():
    set_color_support(TerminalColorSupport.BASIC)
    formatter = ColoredFormatter()

    warning = formatter.format(make_record("sammd.cli", logging.WARNING, "careful"))
    error = formatter.format(make_record("sammd.utils", logging.ERROR, "broken"))

    assert warning.startswith("\033[93m")
    assert "sammd.cli - WARNING - careful" in warning
    assert error.startswith("\033[91m")
    assert "sammd.utils - ERROR - broken" in error


def test_unknown_info_logger_is_uncolored():
    set_color_support(TerminalColorSupport.BASIC)

    output = ColoredFormatter().format(make_record("other.module", logging.INFO, "plain"))

    assert "\033[" not in output
    assert "other.module - INFO - plain" in output
