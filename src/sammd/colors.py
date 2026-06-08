"""Small colored terminal helpers for SAMMD CLI output."""

from __future__ import annotations

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


def detect_color_support(stream: IO[str] | None = None) -> TerminalColorSupport:
    """Detect terminal color support while respecting NO_COLOR."""

    if stream is None:
        stream = sys.stdout
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
