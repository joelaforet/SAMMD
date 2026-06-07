"""Lightweight surface metadata and deterministic slab planning."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil, sqrt

Vector3 = tuple[float, float, float]


@dataclass(frozen=True)
class FccSurfaceMetadata:
    """Crystallographic metadata for an Fcc metal surface."""

    metal: str
    facet: str
    lattice_constant_nm: float

    @property
    def interlayer_spacing_nm(self) -> float:
        """Return the interlayer spacing for the configured Fcc facet."""

        if self.facet != "111":
            msg = f"unsupported Fcc facet '{self.facet}'"
            raise ValueError(msg)
        return self.lattice_constant_nm / sqrt(3)

    def slab_thickness_nm(self, layers: int) -> float:
        """Return first-to-last layer separation for an Fcc slab.

        Parameters
        ----------
        layers
            Number of atomic layers in the slab.

        Returns
        -------
        float
            Physical separation from the first to last atomic layer.
        """

        if layers <= 0:
            msg = "layers must be positive"
            raise ValueError(msg)
        return (layers - 1) * self.interlayer_spacing_nm

    @property
    def nearest_neighbor_spacing_nm(self) -> float:
        """Return the in-plane nearest-neighbor spacing for Fcc(111)."""

        if self.facet != "111":
            msg = f"unsupported Fcc facet '{self.facet}'"
            raise ValueError(msg)
        return self.lattice_constant_nm / sqrt(2)


@dataclass(frozen=True)
class SurfaceSlab:
    """Internal coordinate plan for a centered metal slab.

    The ``slab_extent_nm`` field describes only the metal slab dimensions. It is not a full
    simulation box because SAM and solvent padding are added by later builders.
    """

    metal: str
    facet: str
    requested_lateral_size_nm: tuple[float, float]
    lateral_size_nm: tuple[float, float]
    supercell_counts: tuple[int, int]
    layers: int
    positions_nm: tuple[Vector3, ...]
    slab_extent_nm: Vector3
    top_z_nm: float
    bottom_z_nm: float
    labels: tuple[str, ...]
    layer_indices: tuple[int, ...]


@dataclass(frozen=True)
class BindingSite:
    """Surface binding-site record for later SAM construction."""

    side: str
    site_kind: str
    position_nm: Vector3
    normal: Vector3
    nearest_metal_atom_indices: tuple[int, ...] = ()


# Ambient fcc cubic lattice constants for the CHARMM-INTERFACE Fcc metal set.
# Provenance: WebElements, "Crystal structures of the elements" element pages,
# exact URLs below, "Cell dimensions" table values in picometers, accessed
# 2026-06-07. Values below are converted to nanometers and rounded to the
# precision shown in the source table. Pd keeps the project-default 0.389 nm
# value, consistent with the same 389 pm tabulation.
FCC_SURFACE_LATTICE_CONSTANTS_NM: tuple[tuple[str, float, str], ...] = (
    ("Ag", 0.40853, "https://www.webelements.com/silver/crystal_structure.html"),
    ("Al", 0.40495, "https://www.webelements.com/aluminium/crystal_structure.html"),
    ("Au", 0.40782, "https://www.webelements.com/gold/crystal_structure.html"),
    ("Cu", 0.36149, "https://www.webelements.com/copper/crystal_structure.html"),
    ("Ni", 0.35238, "https://www.webelements.com/nickel/crystal_structure.html"),
    ("Pb", 0.49508, "https://www.webelements.com/lead/crystal_structure.html"),
    ("Pd", 0.389, "https://www.webelements.com/palladium/crystal_structure.html"),
    ("Pt", 0.39242, "https://www.webelements.com/platinum/crystal_structure.html"),
)


FCC_SURFACE_REGISTRY: dict[tuple[str, str], FccSurfaceMetadata] = {
    (symbol, "111"): FccSurfaceMetadata(
        metal=symbol,
        facet="111",
        lattice_constant_nm=lattice_constant_nm,
    )
    for symbol, lattice_constant_nm, _source_url in FCC_SURFACE_LATTICE_CONSTANTS_NM
}


def get_fcc_surface_metadata(metal: str, facet: str) -> FccSurfaceMetadata:
    """Return registered Fcc surface metadata.

    Parameters
    ----------
    metal
        Metal element symbol.
    facet
        Miller-index facet label.

    Returns
    -------
    FccSurfaceMetadata
        Registered surface metadata.
    """

    try:
        return FCC_SURFACE_REGISTRY[(metal, facet)]
    except KeyError as error:
        supported = ", ".join(f"{item[0]}({item[1]})" for item in sorted(FCC_SURFACE_REGISTRY))
        msg = f"unsupported Fcc surface '{metal}({facet})'; supported surfaces: {supported}"
        raise ValueError(msg) from error


def plan_fcc111_slab(
    metal: str,
    lateral_size_nm: tuple[float, float],
    layers: int,
    *,
    centered: bool = True,
) -> SurfaceSlab:
    """Plan a deterministic centered registered Fcc(111) slab in nanometers.

    Parameters
    ----------
    metal
        Registered Fcc metal element symbol.
    lateral_size_nm
        Rectangular lateral dimensions in nanometers.
    layers
        Number of Fcc(111) atomic layers.
    centered
        Whether to center the slab around ``z=0``.

    Returns
    -------
    SurfaceSlab
        Slab positions and metadata suitable for later topology construction.
    """

    metadata = get_fcc_surface_metadata(metal, "111")
    if len(lateral_size_nm) != 2 or any(dimension <= 0 for dimension in lateral_size_nm):
        msg = "lateral_size_nm must contain two positive dimensions"
        raise ValueError(msg)
    if layers <= 0:
        msg = "layers must be positive"
        raise ValueError(msg)
    if not centered:
        msg = "only centered Fcc(111) slabs are supported by the lightweight planner"
        raise ValueError(msg)

    spacing_nm = metadata.nearest_neighbor_spacing_nm
    row_spacing_nm = spacing_nm * sqrt(3) / 2
    layer_spacing_nm = metadata.interlayer_spacing_nm
    layer_offsets = _abc_layer_offsets(spacing_nm, row_spacing_nm)
    supercell_counts = _commensurate_supercell_counts(lateral_size_nm, spacing_nm, row_spacing_nm)
    effective_lateral_size_nm = (
        supercell_counts[0] * spacing_nm,
        supercell_counts[1] * row_spacing_nm,
    )
    z_origin = (layers - 1) * layer_spacing_nm / 2

    positions: list[Vector3] = []
    labels: list[str] = []
    layer_indices: list[int] = []
    serial = 1
    for layer in range(layers):
        offset_x, offset_y = layer_offsets[layer % 3]
        z_nm = layer * layer_spacing_nm - z_origin
        for x_nm, y_nm in _iter_periodic_triangular_lattice_points(
            supercell_counts,
            effective_lateral_size_nm,
            spacing_nm,
            row_spacing_nm,
            offset_x,
            offset_y,
        ):
            positions.append((x_nm, y_nm, z_nm))
            labels.append(f"{metadata.metal}{serial}")
            layer_indices.append(layer)
            serial += 1

    slab_extent_nm = (
        effective_lateral_size_nm[0],
        effective_lateral_size_nm[1],
        metadata.slab_thickness_nm(layers),
    )
    return SurfaceSlab(
        metal=metadata.metal,
        facet=metadata.facet,
        requested_lateral_size_nm=lateral_size_nm,
        lateral_size_nm=effective_lateral_size_nm,
        supercell_counts=supercell_counts,
        layers=layers,
        positions_nm=tuple(positions),
        slab_extent_nm=slab_extent_nm,
        top_z_nm=z_origin,
        bottom_z_nm=-z_origin,
        labels=tuple(labels),
        layer_indices=tuple(layer_indices),
    )


def plan_pd111_slab(
    lateral_size_nm: tuple[float, float],
    layers: int,
    *,
    centered: bool = True,
) -> SurfaceSlab:
    """Plan a deterministic centered Pd(111) slab in nanometers."""

    return plan_fcc111_slab("Pd", lateral_size_nm, layers, centered=centered)


def generate_binding_sites(
    slab: SurfaceSlab, site_kind: str = "fcc_hollow"
) -> tuple[BindingSite, ...]:
    """Generate deterministic top and bottom Fcc(111) binding sites.

    Parameters
    ----------
    slab
        Planned slab from :func:`plan_fcc111_slab`.
    site_kind
        Surface site label. ``fcc_hollow`` and ``hcp_hollow`` are supported for Fcc(111).

    Returns
    -------
    tuple[BindingSite, ...]
        Binding sites on both exposed faces.
    """

    supported_site_kinds = {"fcc_hollow", "hcp_hollow"}
    if site_kind not in supported_site_kinds:
        supported = ", ".join(sorted(supported_site_kinds))
        msg = f"unsupported binding site kind '{site_kind}'; supported site kinds: {supported}"
        raise ValueError(msg)
    metadata = get_fcc_surface_metadata(slab.metal, slab.facet)
    spacing_nm = metadata.nearest_neighbor_spacing_nm
    row_spacing_nm = spacing_nm * sqrt(3) / 2
    layer_offsets = _abc_layer_offsets(spacing_nm, row_spacing_nm)
    sites: list[BindingSite] = []

    for side, layer, z_nm, normal, outward_step in (
        ("bottom", 0, slab.bottom_z_nm, (0.0, 0.0, -1.0), -1),
        ("top", slab.layers - 1, slab.top_z_nm, (0.0, 0.0, 1.0), 1),
    ):
        if site_kind == "fcc_hollow":
            site_offset_x, site_offset_y = layer_offsets[(layer + outward_step) % 3]
        else:
            site_offset_x, site_offset_y = layer_offsets[(layer - outward_step) % 3]
        face_indices = tuple(
            index for index, layer_index in enumerate(slab.layer_indices) if layer_index == layer
        )
        for x_nm, y_nm in _iter_periodic_triangular_lattice_points(
            slab.supercell_counts,
            slab.lateral_size_nm,
            spacing_nm,
            row_spacing_nm,
            site_offset_x,
            site_offset_y,
        ):
            nearest = _nearest_indices_xy(
                (x_nm, y_nm), slab.positions_nm, face_indices, slab.lateral_size_nm, count=3
            )
            sites.append(BindingSite(side, site_kind, (x_nm, y_nm, z_nm), normal, nearest))

    return tuple(sites)


def _abc_layer_offsets(spacing_nm: float, row_spacing_nm: float) -> tuple[tuple[float, float], ...]:
    """Return deterministic ABC in-plane offsets for an Fcc(111) slab."""

    hollow_offset = (spacing_nm / 2, row_spacing_nm / 3)
    return (
        (0.0, 0.0),
        hollow_offset,
        (2 * hollow_offset[0], 2 * hollow_offset[1]),
    )


def _commensurate_supercell_counts(
    requested_lateral_size_nm: tuple[float, float], spacing_nm: float, row_spacing_nm: float
) -> tuple[int, int]:
    """Return integer rectangular cell counts for a periodic Fcc(111) surface."""

    nx = ceil(requested_lateral_size_nm[0] / spacing_nm)
    ny = ceil(requested_lateral_size_nm[1] / row_spacing_nm)
    if nx < 2 or ny < 2:
        msg = "Fcc(111) lateral cell must contain at least two columns and two rows"
        raise ValueError(msg)
    if ny % 2:
        ny += 1
    if nx * ny < 3:
        msg = "Fcc(111) lateral cell must contain at least three surface atoms"
        raise ValueError(msg)
    return nx, ny


def _iter_periodic_triangular_lattice_points(
    supercell_counts: tuple[int, int],
    lateral_size_nm: tuple[float, float],
    spacing_nm: float,
    row_spacing_nm: float,
    offset_x: float,
    offset_y: float,
) -> tuple[tuple[float, float], ...]:
    """Yield sorted points from a wrapped 2D triangular surface lattice."""

    nx, ny = supercell_counts
    points: list[tuple[float, float]] = []
    for j in range(ny):
        for i in range(nx):
            x_raw = i * spacing_nm + (j % 2) * spacing_nm / 2 + offset_x
            y_raw = j * row_spacing_nm + offset_y
            x_nm = _wrap_centered(x_raw, lateral_size_nm[0])
            y_nm = _wrap_centered(y_raw, lateral_size_nm[1])
            points.append((round(x_nm, 12), round(y_nm, 12)))
    return tuple(sorted(points, key=lambda point: (point[1], point[0])))


def _wrap_centered(value_nm: float, length_nm: float) -> float:
    """Wrap a coordinate into the centered periodic interval."""

    return ((value_nm + length_nm / 2) % length_nm) - length_nm / 2


def _nearest_indices_xy(
    xy_nm: tuple[float, float],
    positions_nm: tuple[Vector3, ...],
    candidate_indices: tuple[int, ...],
    lateral_size_nm: tuple[float, float],
    *,
    count: int,
) -> tuple[int, ...]:
    """Return nearest atom indices using minimum-image lateral distances."""

    if len(candidate_indices) < count:
        msg = f"cannot assign {count} nearest atoms from {len(candidate_indices)} candidates"
        raise ValueError(msg)
    ranked = sorted(
        candidate_indices,
        key=lambda index: (
            _minimum_image_delta(positions_nm[index][0] - xy_nm[0], lateral_size_nm[0]) ** 2
            + _minimum_image_delta(positions_nm[index][1] - xy_nm[1], lateral_size_nm[1]) ** 2,
            index,
        ),
    )
    return tuple(ranked[:count])


def _minimum_image_delta(delta_nm: float, length_nm: float) -> float:
    """Return the nearest periodic displacement for one dimension."""

    return delta_nm - round(delta_nm / length_nm) * length_nm
