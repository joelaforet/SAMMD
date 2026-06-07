"""Tests for lightweight surface metadata."""

import pytest

from sammd.surfaces import (
    FCC_SURFACE_REGISTRY,
    _nearest_indices_xy,
    generate_binding_sites,
    get_fcc_surface_metadata,
    plan_fcc111_slab,
    plan_pd111_slab,
)


def test_pd111_surface_metadata() -> None:
    """Expose Pd(111) lattice and interlayer spacing metadata."""

    metadata = get_fcc_surface_metadata("Pd", "111")

    assert FCC_SURFACE_REGISTRY[("Pd", "111")] == metadata
    assert metadata.lattice_constant_nm == 0.389
    assert metadata.interlayer_spacing_nm == pytest.approx(0.2245892547, rel=1e-6)
    assert metadata.slab_thickness_nm(8) == pytest.approx(7 * 0.389 / 3**0.5)


def test_unsupported_surface_fails_clearly() -> None:
    """Reject surfaces outside the lightweight metadata registry."""

    with pytest.raises(ValueError, match="unsupported Fcc surface"):
        get_fcc_surface_metadata("Fe", "111")


def test_fcc111_slab_supports_non_pd_registered_metal() -> None:
    """Plan a centered Pt(111) slab with metal-specific labels and spacing."""

    metadata = get_fcc_surface_metadata("Pt", "111")
    slab = plan_fcc111_slab("Pt", (1.0, 1.0), 3)

    assert slab.metal == "Pt"
    assert slab.facet == "111"
    assert slab.labels[:3] == ("Pt1", "Pt2", "Pt3")
    assert slab.supercell_counts == (4, 6)
    assert len(slab.positions_nm) == 72
    assert slab.bottom_z_nm == pytest.approx(-metadata.interlayer_spacing_nm, rel=1e-6)
    assert slab.top_z_nm == pytest.approx(metadata.interlayer_spacing_nm, rel=1e-6)
    assert slab.lateral_size_nm[0] == pytest.approx(
        slab.supercell_counts[0] * metadata.nearest_neighbor_spacing_nm
    )
    assert slab.lateral_size_nm != plan_pd111_slab((1.0, 1.0), 3).lateral_size_nm


def test_pd111_slab_geometry_is_centered_and_deterministic() -> None:
    """Plan a small centered Pd(111) slab with deterministic atom metadata."""

    slab = plan_pd111_slab((1.0, 1.0), 3)

    assert slab.metal == "Pd"
    assert slab.facet == "111"
    assert slab.requested_lateral_size_nm == (1.0, 1.0)
    assert slab.supercell_counts == (4, 6)
    assert len(slab.positions_nm) == 72
    assert len(slab.labels) == len(slab.positions_nm)
    assert slab.labels[:3] == ("Pd1", "Pd2", "Pd3")
    assert slab.bottom_z_nm == pytest.approx(-0.2245892547, rel=1e-6)
    assert slab.top_z_nm == pytest.approx(0.2245892547, rel=1e-6)
    assert slab.top_z_nm == pytest.approx(-slab.bottom_z_nm)
    assert slab.slab_extent_nm == pytest.approx((*slab.lateral_size_nm, 2 * 0.389 / 3**0.5))
    assert plan_pd111_slab((1.0, 1.0), 3) == slab


def test_noncommensurate_lateral_size_is_adjusted_upward() -> None:
    """Report the effective commensurate periodic cell, not the requested minimum."""

    slab = plan_pd111_slab((1.01, 1.01), 3)

    assert slab.requested_lateral_size_nm == (1.01, 1.01)
    assert slab.supercell_counts == (4, 6)
    assert slab.lateral_size_nm[0] >= 1.01
    assert slab.lateral_size_nm[1] >= 1.01
    assert slab.supercell_counts[1] % 2 == 0


def test_pd111_binding_sites_include_opposite_faces() -> None:
    """Generate top and bottom fcc hollow sites with opposite normals."""

    slab = plan_pd111_slab((1.0, 1.0), 3)
    sites = generate_binding_sites(slab)
    top_sites = [site for site in sites if site.side == "top"]
    bottom_sites = [site for site in sites if site.side == "bottom"]

    assert len(top_sites) == 24
    assert len(bottom_sites) == 24
    assert {site.normal for site in top_sites} == {(0.0, 0.0, 1.0)}
    assert {site.normal for site in bottom_sites} == {(0.0, 0.0, -1.0)}
    assert {site.position_nm[2] for site in top_sites} == {slab.top_z_nm}
    assert {site.position_nm[2] for site in bottom_sites} == {slab.bottom_z_nm}
    assert all(len(site.nearest_metal_atom_indices) == 3 for site in sites)


def test_binding_sites_support_non_pd_fcc111_slab() -> None:
    """Generate hollow sites for a registered non-Pd Fcc(111) slab."""

    slab = plan_fcc111_slab("Au", (1.0, 1.0), 3)
    sites = generate_binding_sites(slab)

    assert {site.side for site in sites} == {"bottom", "top"}
    assert {site.site_kind for site in sites} == {"fcc_hollow"}
    assert len(sites) == 48
    assert all(len(site.nearest_metal_atom_indices) == 3 for site in sites)


def test_fcc_and_hcp_hollows_are_stack_aware_on_both_faces() -> None:
    """Distinguish outward fcc and inward hcp hollow projections by slab side."""

    slab = plan_pd111_slab((1.0, 1.0), 3)
    fcc_sites = generate_binding_sites(slab, site_kind="fcc_hollow")
    hcp_sites = generate_binding_sites(slab, site_kind="hcp_hollow")

    for side in ("bottom", "top"):
        fcc_positions = {site.position_nm[:2] for site in fcc_sites if site.side == side}
        hcp_positions = {site.position_nm[:2] for site in hcp_sites if site.side == side}
        assert len(fcc_positions) == len(hcp_positions) == 24
        assert fcc_positions != hcp_positions

    bottom_fcc = {site.position_nm[:2] for site in fcc_sites if site.side == "bottom"}
    top_fcc = {site.position_nm[:2] for site in fcc_sites if site.side == "top"}
    assert bottom_fcc != top_fcc


def test_hollow_nearest_metals_use_periodic_minimum_image() -> None:
    """Assign edge hollow neighbors across the lateral periodic boundary."""

    slab = plan_pd111_slab((1.0, 1.0), 3)
    face_indices = [index for index, layer in enumerate(slab.layer_indices) if layer == 0]
    edge_site, naive_nearest = next(
        (site, _naive_nearest_xy(site.position_nm, slab, face_indices))
        for site in generate_binding_sites(slab)
        if site.side == "bottom"
        and site.position_nm[0] < -0.4
        and site.nearest_metal_atom_indices
        != _naive_nearest_xy(site.position_nm, slab, face_indices)
    )

    assert edge_site.nearest_metal_atom_indices != naive_nearest
    assert any(slab.positions_nm[index][0] > 0.4 for index in edge_site.nearest_metal_atom_indices)


def test_periodic_surface_neighbors_are_uniform_across_seams() -> None:
    """Keep Pd nearest-neighbor spacing uniform at x and y periodic seams."""

    metadata = get_fcc_surface_metadata("Pd", "111")
    slab = plan_pd111_slab((1.0, 1.0), 3)
    face_indices = [index for index, layer in enumerate(slab.layer_indices) if layer == 0]
    edge_indices = [
        index
        for index in face_indices
        if abs(abs(slab.positions_nm[index][0]) - slab.lateral_size_nm[0] / 2) < 0.15
        or abs(abs(slab.positions_nm[index][1]) - slab.lateral_size_nm[1] / 2) < 0.15
    ]

    assert edge_indices
    for index in edge_indices:
        distances = sorted(
            _minimum_image_xy_distance(slab.positions_nm[index], slab.positions_nm[other], slab)
            for other in face_indices
            if other != index
        )
        assert distances[0] == pytest.approx(metadata.nearest_neighbor_spacing_nm, rel=1e-6)


def test_too_small_lateral_cells_fail_clearly() -> None:
    """Reject cells too small to define three-fold hollow sites."""

    with pytest.raises(ValueError, match="at least two columns and two rows"):
        plan_pd111_slab((0.2, 0.2), 3)

    with pytest.raises(ValueError, match="cannot assign 3 nearest atoms from 2 candidates"):
        _nearest_indices_xy(
            (0.0, 0.0),
            ((0.0, 0.0, 0.0), (0.1, 0.0, 0.0)),
            (0, 1),
            (1.0, 1.0),
            count=3,
        )


def test_unsupported_binding_site_kind_fails_clearly() -> None:
    """Reject future site labels until their geometry is implemented."""

    slab = plan_pd111_slab((1.0, 1.0), 3)

    with pytest.raises(ValueError, match="unsupported binding site kind 'bridge'"):
        generate_binding_sites(slab, site_kind="bridge")


def _minimum_image_xy_distance(
    first: tuple[float, float, float], second: tuple[float, float, float], slab
) -> float:
    """Return minimum-image distance in the slab lateral plane."""

    dx = first[0] - second[0]
    dy = first[1] - second[1]
    dx -= round(dx / slab.lateral_size_nm[0]) * slab.lateral_size_nm[0]
    dy -= round(dy / slab.lateral_size_nm[1]) * slab.lateral_size_nm[1]
    return (dx**2 + dy**2) ** 0.5


def _naive_nearest_xy(
    position: tuple[float, float, float], slab, face_indices: list[int]
) -> tuple[int, ...]:
    """Return nearest lateral atoms without periodic wrapping."""

    return tuple(
        sorted(
            face_indices,
            key=lambda index: (slab.positions_nm[index][0] - position[0]) ** 2
            + (slab.positions_nm[index][1] - position[1]) ** 2,
        )[:3]
    )
