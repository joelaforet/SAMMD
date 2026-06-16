"""Deterministic SAM placement planning without molecular export dependencies."""

from __future__ import annotations

from dataclasses import dataclass
from math import floor, pi, sqrt, tau
from random import Random

from sammd.core.config import SAMComponentConfig, SAMConfig
from sammd.model.metal_sulfur import MetalSulfurLJOverrideSpec, default_metal_sulfur_interaction
from sammd.model.surfaces import BindingSite

Vector3 = tuple[float, float, float]
DEFAULT_SULFUR_HEIGHT_NM = 0.18


@dataclass(frozen=True)
class SAMAnchorPose:
    """Dependency-free anchor pose for future full SAM coordinate generation."""

    site_position_nm: Vector3
    sulfur_position_nm: Vector3
    axis_direction: Vector3
    normal: Vector3
    azimuth_rad: float
    sulfur_height_nm: float
    site_kind: str
    nearest_metal_atom_indices: tuple[int, ...]
    metal_sulfur_interaction: MetalSulfurLJOverrideSpec
    attachment_mode: str


@dataclass(frozen=True)
class SAMPlacement:
    """Assignment of one SAM component to one surface binding site."""

    component_name: str
    component_residue_name: str
    component_smiles: str
    side: str
    site_kind: str
    position_nm: Vector3
    normal: Vector3
    anchor_pose: SAMAnchorPose
    anchor_metadata: dict[str, object]


@dataclass(frozen=True)
class SAMPlacementPlan:
    """Deterministic SAM occupancy plan for both slab faces."""

    placements: tuple[SAMPlacement, ...]
    selected_sites_per_side: int
    lateral_area_nm2: float
    seed: int


def plan_sam_placements(
    sam_config: SAMConfig,
    binding_sites: list[BindingSite] | tuple[BindingSite, ...],
    lateral_area_nm2: float,
    *,
    seed: int,
    lateral_size_nm: tuple[float, float] | None = None,
) -> SAMPlacementPlan:
    """Select binding sites and assign SAM components deterministically.

    Parameters
    ----------
    sam_config
        Validated SAM composition and grafting-density configuration.
    binding_sites
        Candidate binding sites on top and bottom faces.
    lateral_area_nm2
        Area of one exposed face in square nanometers. Target site counts use half-up rounding.
    seed
        Random seed controlling deterministic site and component shuffling.
    lateral_size_nm
        Optional periodic x/y box lengths. When supplied, site spacing uses
        minimum-image lateral distances so selected anchors do not cluster
        across periodic boundaries.

    Returns
    -------
    SAMPlacementPlan
        Selected site occupancy and component assignments.
    """

    if lateral_area_nm2 <= 0:
        msg = "lateral_area_nm2 must be positive"
        raise ValueError(msg)

    selected_sites_per_side = floor(lateral_area_nm2 / sam_config.grafting_density + 0.5)
    if selected_sites_per_side <= 0:
        msg = "grafting density selects no SAM sites"
        raise ValueError(msg)

    metal_sulfur_interaction = default_metal_sulfur_interaction()
    requested_site_kinds = {metal_sulfur_interaction.site_kind}
    supplied_site_kinds = {site.site_kind for site in binding_sites}
    missing_site_kinds = sorted(requested_site_kinds - supplied_site_kinds)
    if missing_site_kinds:
        supplied = ", ".join(sorted(supplied_site_kinds)) or "none"
        msg = (
            f"SAM anchor site kind(s) {', '.join(missing_site_kinds)} were requested, "
            f"but supplied binding sites contain: {supplied}"
        )
        raise ValueError(msg)
    requested_site_kind = next(iter(requested_site_kinds))

    rng = Random(seed)
    placements: list[SAMPlacement] = []
    for side in ("bottom", "top"):
        side_sites = [
            site
            for site in binding_sites
            if site.side == side and site.site_kind == requested_site_kind
        ]
        if len(side_sites) < selected_sites_per_side:
            msg = (
                f"not enough {side} binding sites for grafting density; need "
                f"{selected_sites_per_side}, found {len(side_sites)}"
            )
            raise ValueError(msg)

        selected_sites = _select_spaced_sites(
            side_sites,
            selected_sites_per_side,
            rng,
            lateral_size_nm=lateral_size_nm,
        )
        components = _components_for_selected_sites(sam_config, selected_sites_per_side, rng)
        for placement_index, (site, component) in enumerate(
            zip(selected_sites, components, strict=True)
        ):
            anchor_pose = plan_anchor_pose(
                site,
                azimuth_rad=sam_azimuth_rad(placement_index),
                metal_sulfur_interaction=metal_sulfur_interaction,
            )
            placements.append(
                SAMPlacement(
                    component_name=component.name,
                    component_residue_name=component.residue_name,
                    component_smiles=component.smiles,
                    side=site.side,
                    site_kind=site.site_kind,
                    position_nm=site.position_nm,
                    normal=site.normal,
                    anchor_pose=anchor_pose,
                    anchor_metadata={
                        "mode": anchor_pose.attachment_mode,
                        "site": anchor_pose.site_kind,
                        "metal_sulfur_interaction": (
                            anchor_pose.metal_sulfur_interaction.to_summary()
                        ),
                        "nearest_metal_atom_indices": anchor_pose.nearest_metal_atom_indices,
                        "azimuth_rad": anchor_pose.azimuth_rad,
                        "sulfur_height_nm": anchor_pose.sulfur_height_nm,
                    },
                )
            )

    return SAMPlacementPlan(
        placements=tuple(placements),
        selected_sites_per_side=selected_sites_per_side,
        lateral_area_nm2=lateral_area_nm2,
        seed=seed,
    )


def sam_azimuth_rad(placement_index: int) -> float:
    """Return a deterministic golden-angle azimuth for a placement index."""

    if placement_index < 0:
        msg = "placement_index must be non-negative"
        raise ValueError(msg)
    golden_angle_rad = pi * (3.0 - sqrt(5.0))
    return (placement_index * golden_angle_rad) % tau


def plan_anchor_pose(
    site: BindingSite,
    sulfur_height_nm: float = DEFAULT_SULFUR_HEIGHT_NM,
    azimuth_rad: float = 0.0,
    metal_sulfur_interaction: MetalSulfurLJOverrideSpec | None = None,
) -> SAMAnchorPose:
    """Plan a sulfur anchor pose from a binding-site plane coordinate."""

    if metal_sulfur_interaction is None:
        metal_sulfur_interaction = default_metal_sulfur_interaction()
    if sulfur_height_nm <= 0:
        msg = "sulfur_height_nm must be positive"
        raise ValueError(msg)
    if site.site_kind != metal_sulfur_interaction.site_kind:
        msg = (
            f"metal-S interaction site kind {metal_sulfur_interaction.site_kind!r} "
            f"does not match binding site {site.site_kind!r}"
        )
        raise ValueError(msg)
    if len(site.nearest_metal_atom_indices) != metal_sulfur_interaction.pairs_per_anchor:
        msg = (
            f"metal-S interaction requires {metal_sulfur_interaction.pairs_per_anchor} "
            "nearest metal atom indices per anchor"
        )
        raise ValueError(msg)
    sulfur_position_nm = tuple(
        position + normal * sulfur_height_nm
        for position, normal in zip(site.position_nm, site.normal, strict=True)
    )
    return SAMAnchorPose(
        site_position_nm=site.position_nm,
        sulfur_position_nm=sulfur_position_nm,
        axis_direction=site.normal,
        normal=site.normal,
        azimuth_rad=azimuth_rad,
        sulfur_height_nm=sulfur_height_nm,
        site_kind=site.site_kind,
        nearest_metal_atom_indices=site.nearest_metal_atom_indices,
        metal_sulfur_interaction=metal_sulfur_interaction,
        attachment_mode=metal_sulfur_interaction.mode,
    )


def _components_for_selected_sites(
    sam_config: SAMConfig, selected_sites: int, rng: Random
) -> tuple[SAMComponentConfig, ...]:
    """Return a shuffled component list matching the selected site count."""

    explicit_counts = [component.count for component in sam_config.components]
    if all(count is not None for count in explicit_counts):
        sam_config.validate_component_counts(selected_sites)
        counts = tuple(count or 0 for count in explicit_counts)
    else:
        counts = _fraction_counts(sam_config.components, selected_sites)

    components: list[SAMComponentConfig] = []
    for component, count in zip(sam_config.components, counts, strict=True):
        components.extend([component] * count)
    rng.shuffle(components)
    return tuple(components)


def _select_spaced_sites(
    sites: list[BindingSite],
    selected_sites: int,
    rng: Random,
    *,
    lateral_size_nm: tuple[float, float] | None = None,
) -> list[BindingSite]:
    """Select sites by farthest-point sampling to avoid local SAM clustering."""

    if lateral_size_nm is not None and (
        len(lateral_size_nm) != 2 or any(dimension <= 0.0 for dimension in lateral_size_nm)
    ):
        msg = "lateral_size_nm must contain two positive dimensions"
        raise ValueError(msg)

    shuffled_sites = list(sites)
    rng.shuffle(shuffled_sites)
    chosen = [shuffled_sites[0]]
    remaining = shuffled_sites[1:]
    while len(chosen) < selected_sites:
        next_site = max(
            remaining,
            key=lambda site: min(
                _site_distance_squared(site, other, lateral_size_nm=lateral_size_nm)
                for other in chosen
            ),
        )
        chosen.append(next_site)
        remaining.remove(next_site)
    return sorted(chosen, key=lambda site: site.position_nm)


def _site_distance_squared(
    left: BindingSite,
    right: BindingSite,
    *,
    lateral_size_nm: tuple[float, float] | None = None,
) -> float:
    """Return squared site distance, using minimum-image x/y when configured."""

    if lateral_size_nm is None:
        return _distance_squared(left.position_nm, right.position_nm)
    dx = _minimum_image_delta(left.position_nm[0] - right.position_nm[0], lateral_size_nm[0])
    dy = _minimum_image_delta(left.position_nm[1] - right.position_nm[1], lateral_size_nm[1])
    dz = left.position_nm[2] - right.position_nm[2]
    return dx * dx + dy * dy + dz * dz


def _minimum_image_delta(delta_nm: float, length_nm: float) -> float:
    """Return the nearest periodic displacement for one dimension."""

    return delta_nm - round(delta_nm / length_nm) * length_nm


def _distance_squared(left: Vector3, right: Vector3) -> float:
    """Return squared Euclidean distance between two 3D points."""

    return sum(
        (left_value - right_value) ** 2
        for left_value, right_value in zip(left, right, strict=True)
    )


def _fraction_counts(
    components: list[SAMComponentConfig], selected_sites: int
) -> tuple[int, ...]:
    """Convert fractional composition to integer counts with exact total."""

    raw_counts = [(component.fraction or 0.0) * selected_sites for component in components]
    base_counts = [int(raw_count) for raw_count in raw_counts]
    remaining = selected_sites - sum(base_counts)
    remainders = sorted(
        enumerate(raw_counts),
        key=lambda item: (-(item[1] - int(item[1])), components[item[0]].name, item[0]),
    )
    for index, _raw_count in remainders[:remaining]:
        base_counts[index] += 1
    return tuple(base_counts)
