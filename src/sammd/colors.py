"""Small colored terminal and logging helpers for SAMMD CLI output."""

from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass
from enum import Enum
from typing import IO

import click


class TerminalColorSupport(Enum):
    """Detected terminal color support."""

    NONE = "none"
    BASIC = "basic"
    EXTENDED = "extended"
    TRUECOLOR = "truecolor"


@dataclass(frozen=True)
class PhaseColor:
    """Color values for one CLI phase."""

    rgb: tuple[int, int, int]
    xterm256: int
    basic_ansi: str


@dataclass(frozen=True)
class ModuleColor:
    """Color values for one logging namespace."""

    rgb: tuple[int, int, int]
    xterm256: int
    basic_ansi: str


class ColorScheme:
    """Logging color scheme with longest-prefix namespace matching."""

    def __init__(
        self,
        module_colors: dict[str, ModuleColor],
        level_overrides: dict[int, ModuleColor] | None = None,
    ) -> None:
        self.module_colors = module_colors
        self.level_overrides = level_overrides or {}

    def color_for(self, logger_name: str, levelno: int) -> ModuleColor | None:
        """Return the level override or the longest matching module color."""

        if levelno in self.level_overrides:
            return self.level_overrides[levelno]

        matches = [
            (prefix, color)
            for prefix, color in self.module_colors.items()
            if logger_name == prefix or logger_name.startswith(f"{prefix}.")
        ]
        if not matches:
            return None
        return max(matches, key=lambda item: len(item[0]))[1]


PHASE_COLORS: dict[str, PhaseColor] = {
    "ok": PhaseColor((80, 200, 120), 78, "\033[92m"),
    "plan": PhaseColor((215, 210, 120), 187, "\033[93m"),
    "build": PhaseColor((175, 215, 175), 151, "\033[92m"),
    "full": PhaseColor((205, 175, 215), 182, "\033[95m"),
    "write": PhaseColor((120, 170, 255), 75, "\033[94m"),
    "next": PhaseColor((215, 175, 215), 182, "\033[95m"),
    "warn": PhaseColor((255, 200, 50), 220, "\033[93m"),
    "error": PhaseColor((255, 80, 80), 196, "\033[91m"),
    "muted": PhaseColor((150, 150, 150), 246, "\033[90m"),
}

_color_support: TerminalColorSupport | None = None
_default_scheme: ColorScheme | None = None


AMBER = ModuleColor((255, 190, 60), 214, "\033[93m")
RED = ModuleColor((255, 80, 80), 196, "\033[91m")


def detect_color_support(stream: IO[str] | None = None) -> TerminalColorSupport:
    """Detect terminal color support while respecting NO_COLOR."""

    if stream is None:
        stream = sys.stderr
    if os.environ.get("NO_COLOR", ""):
        return TerminalColorSupport.NONE
    if not hasattr(stream, "isatty") or not stream.isatty():
        return TerminalColorSupport.NONE
    if os.environ.get("TERM", "") == "dumb":
        return TerminalColorSupport.NONE
    if os.environ.get("COLORTERM", "").lower() in {"truecolor", "24bit"}:
        return TerminalColorSupport.TRUECOLOR
    if "256color" in os.environ.get("TERM", ""):
        return TerminalColorSupport.EXTENDED
    return TerminalColorSupport.BASIC


def get_color_support() -> TerminalColorSupport:
    """Return cached terminal color support."""

    global _color_support
    if _color_support is None:
        _color_support = detect_color_support()
    return _color_support


def set_color_support(support: TerminalColorSupport | None) -> None:
    """Override cached terminal color support, or reset it with None."""

    global _color_support
    _color_support = support


def default_scheme() -> ColorScheme:
    """Return SAMMD's default logging color scheme."""

    return ColorScheme(
        {
            "sammd.cli": ModuleColor((120, 170, 255), 75, "\033[94m"),
            "sammd.core": ModuleColor((80, 200, 120), 78, "\033[92m"),
            "sammd.backends": ModuleColor((175, 215, 175), 151, "\033[92m"),
            "sammd.backends.interchange": ModuleColor((120, 215, 215), 80, "\033[96m"),
            "sammd.backends.openff": ModuleColor((175, 135, 255), 141, "\033[95m"),
            "sammd.backends.packmol": ModuleColor((215, 175, 215), 182, "\033[95m"),
            "sammd.model": ModuleColor((205, 175, 215), 182, "\033[95m"),
            "sammd.runtime": ModuleColor((255, 200, 120), 222, "\033[93m"),
            "sammd.analysis": ModuleColor((120, 215, 170), 79, "\033[92m"),
            "sammd.utils": ModuleColor((150, 150, 150), 246, "\033[90m"),
        },
        {
            logging.WARNING: AMBER,
            logging.ERROR: RED,
            logging.CRITICAL: RED,
        },
    )


def get_scheme() -> ColorScheme:
    """Return cached default logging color scheme."""

    global _default_scheme
    if _default_scheme is None:
        _default_scheme = default_scheme()
    return _default_scheme


def _logging_escape(color: ModuleColor) -> tuple[str, str]:
    support = get_color_support()
    if support is TerminalColorSupport.NONE:
        return "", ""
    if support is TerminalColorSupport.TRUECOLOR:
        r, g, b = color.rgb
        return f"\033[38;2;{r};{g};{b}m", "\033[0m"
    if support is TerminalColorSupport.EXTENDED:
        return f"\033[38;5;{color.xterm256}m", "\033[0m"
    return color.basic_ansi, "\033[0m"


class ColoredFormatter(logging.Formatter):
    """Formatter that colors the complete formatted log line."""

    def __init__(
        self,
        fmt: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt: str | None = None,
        scheme: ColorScheme | None = None,
    ) -> None:
        super().__init__(fmt=fmt, datefmt=datefmt)
        self.scheme = scheme or get_scheme()

    def format(self, record: logging.LogRecord) -> str:
        message = super().format(record)
        color = self.scheme.color_for(record.name, record.levelno)
        if color is None:
            return message
        open_seq, close_seq = _logging_escape(color)
        return f"{open_seq}{message}{close_seq}" if open_seq else message


def setup_colored_logging(verbose: bool = False, no_color: bool = False) -> None:
    """Configure root logging with SAMMD colored output."""

    if no_color:
        set_color_support(TerminalColorSupport.NONE)

    handler = logging.StreamHandler()
    handler.setFormatter(ColoredFormatter())

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(logging.DEBUG if verbose else logging.INFO)


def _escape(phase: str) -> tuple[str, str]:
    support = get_color_support()
    if support is TerminalColorSupport.NONE:
        return "", ""
    color = PHASE_COLORS.get(phase, PHASE_COLORS["muted"])
    if support is TerminalColorSupport.TRUECOLOR:
        r, g, b = color.rgb
        return f"\033[38;2;{r};{g};{b}m", "\033[0m"
    if support is TerminalColorSupport.EXTENDED:
        return f"\033[38;5;{color.xterm256}m", "\033[0m"
    return color.basic_ansi, "\033[0m"


def styled(text: object, phase: str = "muted") -> str:
    """Return text wrapped in the color for a CLI phase."""

    open_seq, close_seq = _escape(phase)
    value = str(text)
    return f"{open_seq}{value}{close_seq}" if open_seq else value


def echo(message: object = "", *, phase: str = "muted") -> None:
    """Print one colored message."""

    click.echo(styled(message, phase))


def rule(title: str, *, phase: str = "build") -> None:
    """Print a compact colored section rule."""

    echo(f"\n{'=' * 72}\n{title}\n{'=' * 72}", phase=phase)


def step(label: str, message: str, *, phase: str, detail: str | None = None) -> None:
    """Print a colored phase label with optional muted detail."""

    label_text = styled(f"{label:<8}", phase)
    if detail is None:
        click.echo(f"{label_text} {message}")
    else:
        click.echo(f"{label_text} {message} {styled(detail, 'muted')}")
