"""Command-line interface for SAMMD scaffolding."""

import json
import logging
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from time import perf_counter
from typing import Any

import click

from sammd.colors import setup_colored_logging
from sammd.core.builders import build_system
from sammd.core.config import CONFIG_TEMPLATE, load_config
from sammd.core.validation import validate_build_plan, validate_output_paths

LOGGER = logging.getLogger(__name__)


@contextmanager
def _timed(message: str) -> Iterator[None]:
    """Log start/end messages with elapsed wall time."""

    start = perf_counter()
    LOGGER.info("%s", message)
    try:
        yield
    finally:
        LOGGER.info("DONE %s %.2fs", message, perf_counter() - start)


def _failed_error_messages(*reports: Any) -> list[str]:
    messages: list[str] = []
    for report in reports:
        for gate in report.gates:
            if not gate.passed and gate.severity == "error":
                messages.append(f"{gate.gate_id}: {gate.message}")
    return messages


@click.group()
@click.option("-v", "--verbose", is_flag=True, help="Enable verbose logging.")
@click.option("--no-color", is_flag=True, help="Disable colored logging output.")
def main(verbose: bool, no_color: bool) -> None:
    """Manage lightweight SAMMD configuration files."""

    setup_colored_logging(verbose=verbose, no_color=no_color)


@main.command()
@click.option("-o", "--output", type=click.Path(path_type=Path), default=Path("sammd.yaml"))
@click.option("--force", is_flag=True, help="Overwrite an existing output file.")
def init(output: Path, force: bool) -> None:
    """Write a commented SAMMD YAML template."""

    if output.exists() and not force:
        raise click.ClickException(f"{output} already exists; use --force to overwrite")
    output.write_text(CONFIG_TEMPLATE, encoding="utf-8")
    LOGGER.info("INIT Created a SAMMD configuration template at %s", output)


@main.command()
@click.argument("config", type=click.Path(exists=True, dir_okay=False, path_type=Path))
def validate(config: Path) -> None:
    """Validate a SAMMD YAML configuration."""

    loaded = load_config(config)
    component_count = len(loaded.sam.components)
    LOGGER.info(
        "OK Configuration valid: %s(%s) with %s SAM component(s)",
        loaded.surface.metal,
        loaded.surface.facet,
        component_count,
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
    help="Export full backend files: solvated_system.cif, interchange.json, anchor_metadata.json.",
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

    LOGGER.info("SAMMD Build")
    with _timed("Reading config and constructing deterministic build plan"):
        plan = build_system(config, output_dir=output_dir)
    with _timed("Running lightweight validation gates"):
        build_report = validate_build_plan(plan)
        output_report = validate_output_paths(plan.output_paths)
    failed_messages = _failed_error_messages(build_report, output_report)
    if failed_messages:
        raise click.ClickException("Validation failed: " + "; ".join(failed_messages))

    LOGGER.info("OK Lightweight validation gates passed")
    LOGGER.info("SYSTEM Surface: %s(%s)", plan.slab.metal, plan.slab.facet)
    LOGGER.info("SYSTEM Binding sites: %s", len(plan.binding_sites))
    LOGGER.info("SYSTEM SAM molecules: %s", len(plan.sam_placements.placements))
    LOGGER.info("SYSTEM Solution counts: %s", dict(plan.solution.molecule_counts))

    try:
        backend_files = None
        with _timed("Writing SAM grafting-density visual check"):
            topology = plan.write_topology_cif(overwrite=overwrite)
        with _timed("Writing resolved configuration"):
            resolved_config = plan.write_resolved_config(overwrite=overwrite)
        if full:
            from sammd.backends.interchange import backend_build_summary, export_interchange_backend

            LOGGER.info("Full MD Export")
            backend_result = export_interchange_backend(
                plan,
                overwrite=overwrite,
            )
            backend_files = backend_result.files
            with _timed("Writing backend-aware build summary"):
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
            with _timed("Writing build summary"):
                build_summary = plan.write_build_summary(overwrite=overwrite)
    except FileExistsError as exc:
        raise click.ClickException(str(exc)) from exc
    except (ImportError, NotImplementedError, RuntimeError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc

    LOGGER.info("Outputs")
    LOGGER.info("FILE SAM grafting-density CIF: %s", topology)
    LOGGER.info("FILE Build summary: %s", build_summary)
    LOGGER.info("FILE Resolved config: %s", resolved_config)
    if backend_files is not None:
        LOGGER.info("FILE Wrote solvated system CIF: %s", backend_files["solvated_system"])
        LOGGER.info("FILE Interchange JSON: %s", backend_files["openff_interchange"])
        LOGGER.info("FILE Anchor metadata: %s", backend_files["anchor_metadata"])
        LOGGER.info(
            "NEXT Load interchange.json and call Interchange.to_openmm() downstream when needed."
        )
    else:
        LOGGER.info("NEXT Inspect sam_grafting_density.cif; run --full for MD simulation files.")
