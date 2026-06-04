"""Deterministic SAM placement planning without molecular backends."""

from __future__ import annotations

from dataclasses import dataclass
from math import floor
from random import Random

from sammd.config import SAMComponentConfig, SAMConfig
from sammd.surfaces import BindingSite

Vector3 = tuple[float, float, float]


@dataclass(frozen=True)
class SAMPlacement:
    """Assignment of one SAM component to one surface binding site."""

    component_name: str
    component_smiles: str
    side: str
    site_kind: str
    position_nm: Vector3
    normal: Vector3
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

    Returns
    -------
    SAMPlacementPlan
        Selected site occupancy and component assignments.
    """

    if lateral_area_nm2 <= 0:
        msg = "lateral_area_nm2 must be positive"
        raise ValueError(msg)

    selected_sites_per_side = floor(lateral_area_nm2 / sam_config.grafting_density.value + 0.5)
    if selected_sites_per_side <= 0:
        msg = "grafting density selects no SAM sites"
        raise ValueError(msg)

    requested_site_kinds = {sam_config.anchor.site}
    requested_site_kinds.update(
        component.anchor.site for component in sam_config.components if component.anchor is not None
    )
    if len(requested_site_kinds) > 1:
        requested = ", ".join(sorted(requested_site_kinds))
        msg = f"mixed SAM anchor site kinds are not supported yet; requested: {requested}"
        raise ValueError(msg)
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

        selected_sites = _select_spaced_sites(side_sites, selected_sites_per_side, rng)
        components = _components_for_selected_sites(sam_config, selected_sites_per_side, rng)
        for site, component in zip(selected_sites, components, strict=True):
            anchor = component.anchor or sam_config.anchor
            placements.append(
                SAMPlacement(
                    component_name=component.name,
                    component_smiles=component.smiles,
                    side=site.side,
                    site_kind=site.site_kind,
                    position_nm=site.position_nm,
                    normal=site.normal,
                    anchor_metadata={
                        "mode": anchor.mode,
                        "site": anchor.site,
                        "nonbonded_scale_factor": anchor.nonbonded.scale_factor,
                        "nearest_metal_atom_indices": site.nearest_metal_atom_indices,
                    },
                )
            )

    return SAMPlacementPlan(
        placements=tuple(placements),
        selected_sites_per_side=selected_sites_per_side,
        lateral_area_nm2=lateral_area_nm2,
        seed=seed,
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
) -> list[BindingSite]:
    """Select sites by farthest-point sampling to avoid local SAM clustering."""

    shuffled_sites = list(sites)
    rng.shuffle(shuffled_sites)
    chosen = [shuffled_sites[0]]
    remaining = shuffled_sites[1:]
    while len(chosen) < selected_sites:
        next_site = max(
            remaining,
            key=lambda site: min(
                _distance_squared(site.position_nm, other.position_nm) for other in chosen
            ),
        )
        chosen.append(next_site)
        remaining.remove(next_site)
    return sorted(chosen, key=lambda site: site.position_nm)


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
