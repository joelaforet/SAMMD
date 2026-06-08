"""Command-line interface for SAMMD scaffolding."""

import json
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from time import perf_counter
from typing import Any

import click

from sammd.colors import rule, step
from sammd.core.builders import build_system
from sammd.core.config import CONFIG_TEMPLATE, load_config
from sammd.core.validation import validate_build_plan, validate_output_paths


@contextmanager
def _timed(label: str, message: str, *, phase: str) -> Iterator[None]:
    """Print start/end messages with elapsed wall time."""

    start = perf_counter()
    step(label, message, phase=phase)
    try:
        yield
    finally:
        step("DONE", message, phase="ok", detail=f"{perf_counter() - start:.2f}s")


def _failed_error_messages(*reports: Any) -> list[str]:
    messages: list[str] = []
    for report in reports:
        for gate in report.gates:
            if not gate.passed and gate.severity == "error":
                messages.append(f"{gate.gate_id}: {gate.message}")
    return messages


@click.group()
def main() -> None:
    """Manage lightweight SAMMD configuration files."""


@main.command()
@click.option("-o", "--output", type=click.Path(path_type=Path), default=Path("sammd.yaml"))
@click.option("--force", is_flag=True, help="Overwrite an existing output file.")
def init(output: Path, force: bool) -> None:
    """Write a commented SAMMD YAML template."""

    if output.exists() and not force:
        raise click.ClickException(f"{output} already exists; use --force to overwrite")
    output.write_text(CONFIG_TEMPLATE, encoding="utf-8")
    step("INIT", f"Created a SAMMD configuration template at {output}", phase="build")


@main.command()
@click.argument("config", type=click.Path(exists=True, dir_okay=False, path_type=Path))
def validate(config: Path) -> None:
    """Validate a SAMMD YAML configuration."""

    loaded = load_config(config)
    component_count = len(loaded.sam.components)
    step(
        "OK",
        f"Configuration valid: {loaded.surface.metal}({loaded.surface.facet}) "
        f"with {component_count} SAM component(s)",
        phase="ok",
    )


@main.command()
@click.argument("config", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option(
    "-o",
    "--output-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help="Override the outputs.directory value from the YAML config.",
)
@click.option(
    "--overwrite",
    is_flag=True,
    help="Replace existing build artifacts.",
)
@click.option(
    "--full",
    is_flag=True,
    help="Export full MD simulation files: solvated_system.cif, interchange.json, system.xml.",
)
@click.option(
    "--export-backend",
    "full",
    is_flag=True,
    hidden=True,
    help="Deprecated alias for --full.",
)
def build(config: Path, output_dir: Path | None, overwrite: bool, full: bool) -> None:
    """Build the current plan and optionally write backend files."""

    rule("SAMMD Build", phase="build")
    with _timed("PLAN", "Reading config and constructing deterministic build plan", phase="plan"):
        plan = build_system(config, output_dir=output_dir)
    with _timed("CHECK", "Running lightweight validation gates", phase="plan"):
        build_report = validate_build_plan(plan)
        output_report = validate_output_paths(plan.output_paths)
    failed_messages = _failed_error_messages(build_report, output_report)
    if failed_messages:
        raise click.ClickException("Validation failed: " + "; ".join(failed_messages))

    step("OK", "Lightweight validation gates passed", phase="ok")
    step("SYSTEM", f"Surface: {plan.slab.metal}({plan.slab.facet})", phase="plan")
    step("SYSTEM", f"Binding sites: {len(plan.binding_sites)}", phase="plan")
    step("SYSTEM", f"SAM molecules: {len(plan.sam_placements.placements)}", phase="plan")
    step("SYSTEM", f"Solution counts: {dict(plan.solution.molecule_counts)}", phase="plan")

    try:
        backend_files = None
        with _timed("WRITE", "Writing SAM grafting-density visual check", phase="write"):
            topology = plan.write_topology_cif(overwrite=overwrite)
        with _timed("WRITE", "Writing resolved configuration", phase="write"):
            resolved_config = plan.write_resolved_config(overwrite=overwrite)
        if full:
            from sammd.backends.interchange import backend_build_summary, export_interchange_backend

            rule("Full MD Export", phase="full")
            export_start = perf_counter()

            def progress(message: str) -> None:
                step("FULL", message, phase="full", detail=f"+{perf_counter() - export_start:.2f}s")

            backend_result = export_interchange_backend(
                plan,
                overwrite=overwrite,
                progress=progress,
            )
            backend_files = backend_result.files
            with _timed("WRITE", "Writing backend-aware build summary", phase="write"):
                build_summary = plan.write_build_summary(overwrite=overwrite)
                build_summary.write_text(
                    json.dumps(
                        backend_build_summary(plan, backend_result),
                        indent=2,
                        ensure_ascii=False,
                    )
                    + "\n",
                    encoding="utf-8",
                )
        else:
            with _timed("WRITE", "Writing build summary", phase="write"):
                build_summary = plan.write_build_summary(overwrite=overwrite)
    except FileExistsError as exc:
        raise click.ClickException(str(exc)) from exc
    except (ImportError, NotImplementedError, RuntimeError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc

    rule("Outputs", phase="write")
    step("FILE", f"SAM grafting-density CIF: {topology}", phase="write")
    step("FILE", f"Build summary: {build_summary}", phase="write")
    step("FILE", f"Resolved config: {resolved_config}", phase="write")
    if backend_files is not None:
        step(
            "FILE",
            f"Wrote solvated system CIF: {backend_files['solvated_system']}",
            phase="write",
        )
        step("FILE", f"Interchange JSON: {backend_files['openff_interchange']}", phase="write")
        step("FILE", f"OpenMM system XML: {backend_files['openmm_system']}", phase="write")
        step("FILE", f"Anchor metadata: {backend_files['anchor_metadata']}", phase="write")
        step(
            "NEXT",
            "Use solvated_system.cif and interchange.json for OpenMM simulation setup.",
            phase="next",
        )
    else:
        step(
            "NEXT",
            "Inspect sam_grafting_density.cif; run --full for MD simulation files.",
            phase="next",
        )
