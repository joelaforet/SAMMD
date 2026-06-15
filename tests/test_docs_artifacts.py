from __future__ import annotations

import subprocess
from pathlib import Path

STALE_SOLVENT_WORDING = (
    "solvent.padding_nm",
    "Approximate solvent height",
    "approximate count planning",
    "pure water with 3.0 nm padding",
)


def test_tracked_docs_do_not_use_stale_solvent_padding_wording() -> None:
    """Ensure tracked documentation cannot reintroduce old solvent padding semantics."""
    repo_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        ["git", "ls-files", "docs"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )

    offenders: list[str] = []
    for relative_path in result.stdout.splitlines():
        path = repo_root / relative_path
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for stale_wording in STALE_SOLVENT_WORDING:
            if stale_wording in text:
                offenders.append(f"{relative_path}: {stale_wording}")

    assert offenders == []
