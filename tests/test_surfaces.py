"""Tests for lightweight surface metadata."""

import pytest

from sammd.surfaces import (
    FCC_SURFACE_REGISTRY,
    generate_binding_sites,
    get_fcc_surface_metadata,
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
        get_fcc_surface_metadata("Au", "111")


def test_pd111_slab_geometry_is_centered_and_deterministic() -> None:
    """Plan a small centered Pd(111) slab with deterministic atom metadata."""

    slab = plan_pd111_slab((1.0, 1.0), 3)

    assert slab.metal == "Pd"
    assert slab.facet == "111"
    assert len(slab.positions_nm) == 36
    assert len(slab.labels) == len(slab.positions_nm)
    assert slab.labels[:3] == ("Pd1", "Pd2", "Pd3")
    assert slab.bottom_z_nm == pytest.approx(-0.2245892547, rel=1e-6)
    assert slab.top_z_nm == pytest.approx(0.2245892547, rel=1e-6)
    assert slab.top_z_nm == pytest.approx(-slab.bottom_z_nm)
    assert slab.slab_extent_nm == pytest.approx((1.0, 1.0, 2 * 0.389 / 3**0.5))
    assert plan_pd111_slab((1.0, 1.0), 3) == slab


def test_pd111_binding_sites_include_opposite_faces() -> None:
    """Generate top and bottom fcc hollow sites with opposite normals."""

    slab = plan_pd111_slab((1.0, 1.0), 3)
    sites = generate_binding_sites(slab)
    top_sites = [site for site in sites if site.side == "top"]
    bottom_sites = [site for site in sites if site.side == "bottom"]

    assert len(top_sites) == 12
    assert len(bottom_sites) == 12
    assert {site.normal for site in top_sites} == {(0.0, 0.0, 1.0)}
    assert {site.normal for site in bottom_sites} == {(0.0, 0.0, -1.0)}
    assert {site.position_nm[2] for site in top_sites} == {slab.top_z_nm}
    assert {site.position_nm[2] for site in bottom_sites} == {slab.bottom_z_nm}
    assert all(len(site.nearest_metal_atom_indices) == 3 for site in sites)


def test_fcc_and_hcp_hollows_are_stack_aware_on_both_faces() -> None:
    """Distinguish outward fcc and inward hcp hollow projections by slab side."""

    slab = plan_pd111_slab((1.0, 1.0), 3)
    fcc_sites = generate_binding_sites(slab, site_kind="fcc_hollow")
    hcp_sites = generate_binding_sites(slab, site_kind="hcp_hollow")

    for side in ("bottom", "top"):
        fcc_positions = {site.position_nm[:2] for site in fcc_sites if site.side == side}
        hcp_positions = {site.position_nm[:2] for site in hcp_sites if site.side == side}
        assert len(fcc_positions) == len(hcp_positions) == 12
        assert fcc_positions != hcp_positions

    bottom_fcc = {site.position_nm[:2] for site in fcc_sites if site.side == "bottom"}
    top_fcc = {site.position_nm[:2] for site in fcc_sites if site.side == "top"}
    assert bottom_fcc != top_fcc


def test_hollow_nearest_metals_use_periodic_minimum_image() -> None:
    """Assign edge hollow neighbors across the lateral periodic boundary."""

    slab = plan_pd111_slab((1.0, 1.0), 3)
    edge_site = next(
        site
        for site in generate_binding_sites(slab)
        if site.side == "bottom" and site.position_nm[0] < -0.4
    )
    face_indices = [index for index, layer in enumerate(slab.layer_indices) if layer == 0]
    naive_nearest = tuple(
        sorted(
            face_indices,
            key=lambda index: (slab.positions_nm[index][0] - edge_site.position_nm[0]) ** 2
            + (slab.positions_nm[index][1] - edge_site.position_nm[1]) ** 2,
        )[:3]
    )

    assert edge_site.nearest_metal_atom_indices != naive_nearest
    assert 2 in edge_site.nearest_metal_atom_indices
    assert slab.positions_nm[2][0] > 0.4


def test_unsupported_binding_site_kind_fails_clearly() -> None:
    """Reject future site labels until their geometry is implemented."""

    slab = plan_pd111_slab((1.0, 1.0), 3)

    with pytest.raises(ValueError, match="unsupported binding site kind 'bridge'"):
        generate_binding_sites(slab, site_kind="bridge")
