"""Output path planning and lightweight PDBx/mmCIF ``.cif`` writing helpers."""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from typing import Any

from sammd.model.surfaces import SurfaceSlab

Vector3 = tuple[float, float, float]


@dataclass(frozen=True)
class OutputPaths:
    """Resolved output artifact paths for system building."""

    sam_grafting_density: Path | None = None
    solvated_system: Path | None = None
    openff_interchange: Path | None = None
    anchor_metadata: Path | None = None
    build_summary: Path | None = None
    resolved_config: Path | None = None
    trajectory: Path | None = None
    thermodynamics: Path | None = None


@dataclass(frozen=True)
class AtomRecord:
    """Lightweight atom record for PDBx/mmCIF visualization scaffolds."""

    serial: int
    atom_name: str
    element: str
    residue_name: str
    residue_id: int
    chain_id: str
    component_label: str
    coordinates_nm: Vector3


def plan_output_paths(config: Any, base_dir: str | Path = ".") -> OutputPaths:
    """Resolve deterministic output paths from a SAMMD config or output section.

    Parameters
    ----------
    config
        Top-level config with an ``output`` attribute, or an output config object.
    base_dir
        Base directory used for relative artifact paths.

    Returns
    -------
    OutputPaths
        Resolved paths with validated visualization and reporter suffixes.
    """

    output_config = getattr(config, "outputs", config)
    files = getattr(output_config, "files", output_config)
    root = Path(base_dir)
    sam_grafting_density = _resolve_output_path(root, files.sam_grafting_density)
    solvated_system = _resolve_output_path(root, files.solvated_system)
    openff_interchange = _resolve_output_path(root, files.openff_interchange)
    anchor_metadata = _resolve_output_path(root, files.anchor_metadata)
    build_summary = _resolve_output_path(root, files.build_summary)
    resolved_config = _resolve_output_path(root, files.resolved_config)

    _validate_suffix(sam_grafting_density, ".cif", "SAM grafting-density")
    _validate_suffix(solvated_system, ".cif", "solvated system")
    _validate_suffix(openff_interchange, ".json", "OpenFF Interchange")
    _validate_suffix(anchor_metadata, ".json", "anchor metadata")
    _validate_suffix(build_summary, ".json", "build summary")
    _validate_suffix(resolved_config, ".yaml", "resolved config")
    return OutputPaths(
        sam_grafting_density=sam_grafting_density,
        solvated_system=solvated_system,
        openff_interchange=openff_interchange,
        anchor_metadata=anchor_metadata,
        build_summary=build_summary,
        resolved_config=resolved_config,
    )


def safe_write_text(path: str | Path, text: str, *, overwrite: bool = False) -> Path:
    """Write text atomically without silent overwrite by default.

    Parameters
    ----------
    path
        Destination path.
    text
        Text content to write.
    overwrite
        Whether an existing destination may be replaced.

    Returns
    -------
    Path
        Destination path.
    """

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not overwrite:
        msg = f"refusing to overwrite existing file: {destination}"
        raise FileExistsError(msg)

    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=destination.parent, delete=False
        ) as handle:
            temporary_name = handle.name
            handle.write(text)
        if overwrite:
            os.replace(temporary_name, destination)
        else:
            os.link(temporary_name, destination)
            os.unlink(temporary_name)
    finally:
        if temporary_name is not None and Path(temporary_name).exists():
            Path(temporary_name).unlink()
    return destination


def format_mmcif(
    atom_records: tuple[AtomRecord, ...] | list[AtomRecord],
    *,
    data_name: str = "sammd_scaffold",
    cell_lengths_nm: Vector3 | None = None,
) -> str:
    """Format lightweight atom records as PDBx/mmCIF text.

    Parameters
    ----------
    atom_records
        Atom records with coordinates in nanometers.
    data_name
        PDBx/mmCIF data block name.
    cell_lengths_nm
        Optional periodic cell lengths in nanometers.

    Returns
    -------
    str
        PDBx/mmCIF text with ``_atom_site`` rows and Angstrom coordinates.
    """

    validated_data_name = _validate_data_name(data_name)
    validated_records = _validate_atom_records(atom_records)
    if not validated_records:
        msg = "at least one atom record is required"
        raise ValueError(msg)
    if cell_lengths_nm is not None:
        _validate_cell_lengths(cell_lengths_nm)

    entity_ids = _build_entity_ids(validated_records)
    lines = [f"data_{validated_data_name}", "#"]
    if cell_lengths_nm is not None:
        lines.extend(
            [
                f"_cell.length_a {_format_float(cell_lengths_nm[0] * 10.0)}",
                f"_cell.length_b {_format_float(cell_lengths_nm[1] * 10.0)}",
                f"_cell.length_c {_format_float(cell_lengths_nm[2] * 10.0)}",
                "_cell.angle_alpha 90.000",
                "_cell.angle_beta 90.000",
                "_cell.angle_gamma 90.000",
                "#",
            ]
        )
    lines.extend(
        [
            "loop_",
            "_entity.id",
            "_entity.type",
            "_entity.pdbx_description",
        ]
    )
    for component_label, entity_id in entity_ids.items():
        lines.append(
            " ".join([str(entity_id), "non-polymer", _quote_cif_value(component_label)])
        )
    lines.extend(
        [
            "#",
            "loop_",
            "_sammd_entity.id",
            "_sammd_entity.component_label",
        ]
    )
    for component_label, entity_id in entity_ids.items():
        lines.append(" ".join([str(entity_id), _quote_cif_value(component_label)]))
    lines.append("#")
    lines.extend(
        [
            "loop_",
            "_atom_site.group_PDB",
            "_atom_site.id",
            "_atom_site.type_symbol",
            "_atom_site.label_atom_id",
            "_atom_site.label_comp_id",
            "_atom_site.label_asym_id",
            "_atom_site.label_entity_id",
            "_atom_site.label_seq_id",
            "_atom_site.Cartn_x",
            "_atom_site.Cartn_y",
            "_atom_site.Cartn_z",
            "_atom_site.occupancy",
            "_atom_site.B_iso_or_equiv",
            "_atom_site.pdbx_PDB_model_num",
        ]
    )
    for record in validated_records:
        x_angstrom, y_angstrom, z_angstrom = (
            coordinate * 10.0 for coordinate in record.coordinates_nm
        )
        lines.append(
            " ".join(
                [
                    "HETATM",
                    str(record.serial),
                    _quote_cif_value(record.element),
                    _quote_cif_value(record.atom_name),
                    _quote_cif_value(record.residue_name),
                    _quote_cif_value(record.chain_id),
                    str(entity_ids[record.component_label]),
                    str(record.residue_id),
                    _format_float(x_angstrom),
                    _format_float(y_angstrom),
                    _format_float(z_angstrom),
                    "1.00",
                    "0.00",
                    "1",
                ]
            )
        )
    lines.append("#")
    return "\n".join(lines) + "\n"


def write_mmcif(
    path: str | Path,
    atom_records: tuple[AtomRecord, ...] | list[AtomRecord],
    *,
    data_name: str = "sammd_scaffold",
    cell_lengths_nm: Vector3 | None = None,
    overwrite: bool = False,
) -> Path:
    """Write lightweight PDBx/mmCIF atom records to a ``.cif`` path.

    Parameters
    ----------
    path
        Destination ``.cif`` path. ``.mmcif`` is common elsewhere, but SAMMD
        keeps stable ``.cif`` artifact names.
    atom_records
        Atom records with nanometer coordinates.
    data_name
        PDBx/mmCIF data block name.
    cell_lengths_nm
        Optional periodic cell lengths in nanometers.
    overwrite
        Whether an existing destination may be replaced.

    Returns
    -------
    Path
        Written path.
    """

    destination = Path(path)
    _validate_suffix(destination, ".cif", "SAM grafting-density")
    text = format_mmcif(atom_records, data_name=data_name, cell_lengths_nm=cell_lengths_nm)
    return safe_write_text(destination, text, overwrite=overwrite)


def slab_to_atom_records(slab: SurfaceSlab, *, chain_id: str = "M") -> tuple[AtomRecord, ...]:
    """Convert a planned metal slab into PyMOL-friendly atom records.

    Parameters
    ----------
    slab
        Planned surface slab.
    chain_id
        Chain identifier used for visualization.

    Returns
    -------
    tuple[AtomRecord, ...]
        Deterministic metal atom records labeled by slab layer role.
    """

    records: list[AtomRecord] = []
    for index, (position_nm, layer_index) in enumerate(
        zip(slab.positions_nm, slab.layer_indices, strict=True)
    ):
        layer_role, residue_name = _slab_layer_label(layer_index, slab.layers)
        atom_label = slab.labels[index] if index < len(slab.labels) else f"{slab.metal}{index + 1}"
        records.append(
            AtomRecord(
                serial=index + 1,
                atom_name=atom_label,
                element=slab.metal,
                residue_name=residue_name,
                residue_id=layer_index + 1,
                chain_id=chain_id,
                component_label=f"metal_{layer_role}",
                coordinates_nm=position_nm,
            )
        )
    return tuple(records)


def _resolve_output_path(base_dir: Path, value: str | Path) -> Path:
    """Resolve one configured output path."""

    path = Path(value)
    return path if path.is_absolute() else base_dir / path


def _resolve_optional_output_path(base_dir: Path, value: str | Path | None) -> Path | None:
    """Resolve one optional configured output path."""

    if value is None:
        return None
    return _resolve_output_path(base_dir, value)


def _validate_suffix(path: Path, expected_suffix: str, artifact_name: str) -> None:
    """Validate a practical artifact suffix."""

    if path.suffix.lower() != expected_suffix:
        msg = f"{artifact_name} output must use a '{expected_suffix}' suffix"
        raise ValueError(msg)


def _format_float(value: float) -> str:
    """Format a coordinate deterministically."""

    return f"{value:.6f}"


def _validate_data_name(data_name: str) -> str:
    """Validate an mmCIF data block name without surprising renames."""

    if not data_name:
        msg = "mmCIF data_name must not be empty"
        raise ValueError(msg)
    if any(character.isspace() or character in "'\"" for character in data_name):
        msg = "mmCIF data_name must not contain whitespace or quotes"
        raise ValueError(msg)
    if any(ord(character) < 32 or ord(character) == 127 for character in data_name):
        msg = "mmCIF data_name must not contain control characters"
        raise ValueError(msg)
    return data_name


def _validate_atom_records(
    atom_records: tuple[AtomRecord, ...] | list[AtomRecord],
) -> tuple[AtomRecord, ...]:
    """Validate atom records before writing mmCIF rows."""

    validated_records = tuple(atom_records)
    serials: set[int] = set()
    for record in validated_records:
        if record.serial <= 0:
            msg = "atom record serial must be positive"
            raise ValueError(msg)
        if record.serial in serials:
            msg = f"atom record serial values must be unique; duplicate serial {record.serial}"
            raise ValueError(msg)
        serials.add(record.serial)
        if record.residue_id <= 0:
            msg = "atom record residue_id must be positive"
            raise ValueError(msg)
        for field_name in (
            "atom_name",
            "element",
            "residue_name",
            "chain_id",
            "component_label",
        ):
            _validate_cif_text_value(getattr(record, field_name), field_name)
        if len(record.coordinates_nm) != 3:
            msg = "atom record coordinates_nm must contain exactly three values"
            raise ValueError(msg)
        if not all(
            isinstance(coordinate, int | float) and isfinite(coordinate)
            for coordinate in record.coordinates_nm
        ):
            msg = "atom record coordinates_nm values must be finite numbers"
            raise ValueError(msg)
    return validated_records


def _validate_cell_lengths(cell_lengths_nm: Vector3) -> None:
    """Validate optional periodic cell lengths."""

    if len(cell_lengths_nm) != 3:
        msg = "cell_lengths_nm must contain exactly three values"
        raise ValueError(msg)
    if not all(
        isinstance(length, int | float) and isfinite(length) and length > 0
        for length in cell_lengths_nm
    ):
        msg = "cell_lengths_nm values must be finite positive numbers"
        raise ValueError(msg)


def _validate_cif_text_value(value: str, field_name: str) -> None:
    """Validate lightweight mmCIF string values before quoting."""

    if not isinstance(value, str) or not value:
        msg = f"atom record {field_name} must be a non-empty string"
        raise ValueError(msg)
    if any(ord(character) < 32 or ord(character) == 127 for character in value):
        msg = f"atom record {field_name} must not contain control characters"
        raise ValueError(msg)


def _build_entity_ids(atom_records: tuple[AtomRecord, ...]) -> dict[str, int]:
    """Assign stable numeric entity IDs for unique component labels."""

    entity_ids: dict[str, int] = {}
    for record in atom_records:
        if record.component_label not in entity_ids:
            entity_ids[record.component_label] = len(entity_ids) + 1
    return entity_ids


def _quote_cif_value(value: str) -> str:
    """Quote mmCIF values only when required."""

    if any(ord(character) < 32 or ord(character) == 127 for character in value):
        msg = "mmCIF values must not contain control characters"
        raise ValueError(msg)
    reserved_prefixes = ("_", "#", ";", "data_", "loop_", "save_", "stop_")
    lower_value = value.lower()
    is_reserved_token = value in {".", "?"}
    requires_quotes = (
        is_reserved_token
        or any(character.isspace() for character in value)
        or value.startswith(("_", "#", ";"))
        or lower_value.startswith(reserved_prefixes[3:])
    )
    if "'" in value and '"' in value:
        msg = "mmCIF values containing both single and double quotes are not supported"
        raise ValueError(msg)
    if not requires_quotes and "'" not in value and '"' not in value:
        return value
    if "'" in value:
        return f'"{value}"'
    return f"'{value}'"


def _slab_layer_label(layer_index: int, layers: int) -> tuple[str, str]:
    """Return component and residue labels for a slab layer."""

    if layer_index == 0:
        return "bottom_layer", "PDB"
    if layer_index == layers - 1:
        return "top_layer", "PDT"
    return "slab", "PDM"
