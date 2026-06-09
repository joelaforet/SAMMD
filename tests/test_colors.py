import io
import logging
import re

import pytest

from sammd import colors as colors_module
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
    colors_module._stdout_color_support = None
    yield
    set_color_support(None)
    colors_module._stdout_color_support = None


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


def test_stdout_helper_uses_stdout_detection_when_stderr_is_tty(monkeypatch):
    class Stdout(io.StringIO):
        def isatty(self) -> bool:
            return False

    class Stderr(io.StringIO):
        def isatty(self) -> bool:
            return True

    stdout = Stdout()
    monkeypatch.setattr(colors_module.sys, "stdout", stdout)
    monkeypatch.setattr(colors_module.sys, "stderr", Stderr())
    monkeypatch.setenv("TERM", "xterm-256color")
    monkeypatch.delenv("COLORTERM", raising=False)

    assert get_color_support() is TerminalColorSupport.EXTENDED

    colors_module.echo("plain", phase="ok")

    assert stdout.getvalue() == "plain\n"


def test_setup_colored_logging_preserves_unrelated_root_handlers(
    restore_root_logging,
):
    root = logging.getLogger()
    unrelated = logging.StreamHandler(io.StringIO())
    root.handlers[:] = [unrelated]

    setup_colored_logging()

    sammd_handlers = [
        handler
        for handler in root.handlers
        if getattr(handler, colors_module._SAMMD_HANDLER_ATTR, False)
    ]

    assert unrelated in root.handlers
    assert len(sammd_handlers) == 1


def test_setup_colored_logging_replaces_only_existing_sammd_handler(
    restore_root_logging,
):
    root = logging.getLogger()
    unrelated = logging.StreamHandler(io.StringIO())
    root.handlers[:] = [unrelated]

    setup_colored_logging()
    first_sammd = next(
        handler
        for handler in root.handlers
        if getattr(handler, colors_module._SAMMD_HANDLER_ATTR, False)
    )
    setup_colored_logging(verbose=True)
    sammd_handlers = [
        handler
        for handler in root.handlers
        if getattr(handler, colors_module._SAMMD_HANDLER_ATTR, False)
    ]

    assert unrelated in root.handlers
    assert first_sammd not in root.handlers
    assert len(sammd_handlers) == 1
    assert root.level == logging.DEBUG


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
