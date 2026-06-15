"""Smoke-tool parity tests for runtime packing helpers."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest


def load_smoke_module() -> ModuleType:
    """Load the excluded temporary smoke tool from its source path."""

    module_path = Path(__file__).parents[1] / "temporary" / "openmm_smoke.py"
    spec = importlib.util.spec_from_file_location("openmm_smoke_test_module", module_path)
    if spec is None or spec.loader is None:
        msg = f"could not load smoke module from {module_path}"
        raise ImportError(msg)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_smoke_auto_solvent_count_allows_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    """Return the recomputed auto count exactly, including zero."""

    smoke = load_smoke_module()

    def plan_solution_composition(config: object, volume_nm3: float) -> SimpleNamespace:
        """Return a zero-count solvent component for auto-count parity."""

        assert volume_nm3 == pytest.approx(0.0)
        return SimpleNamespace(
            solvent_components=(SimpleNamespace(name="ethanol", count=0),),
        )

    monkeypatch.setattr(smoke, "plan_solution_composition", plan_solution_composition)

    count = smoke.resolve_solvent_count(
        "auto",
        SimpleNamespace(config=SimpleNamespace()),
        "ethanol",
        planning_volume_nm3=0.0,
    )

    assert count == 0


def test_smoke_explicit_zero_solvent_count_is_valid() -> None:
    """Allow explicit zero solvent count to reach the zero-solvent path."""

    smoke = load_smoke_module()

    args = SimpleNamespace(
        lateral_size_nm=2.0,
        solvent_padding_nm=3.0,
        timestep_fs=2.0,
        friction_per_ps=1.0,
        pd_s_sigma_angstrom=2.0,
        pd_s_epsilon_kcal_mol=1.0,
        duration_ns=1.0,
        sulfur_height_nm=0.18,
        seed=1,
        steps=1,
        minimize_iterations=0,
        frames=1,
        report_interval=1,
        reactant_count=None,
        solvent_count="0",
    )

    smoke.validate_args(args)


def test_smoke_solvent_chain_wrapping_skips_reserved_metal_chain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Wrap solvent chains through runtime-allowed IDs without using metal chain M."""

    smoke = load_smoke_module()
    monkeypatch.setattr(smoke, "MAX_RESIDUES_PER_CHAIN", 1)
    assigner = smoke.ComponentResidueAssigner()

    identities = assigner.allocate("ethanol", "EOH", 10)

    assert [identity.chain_id for identity in identities] == [
        "D",
        "E",
        "F",
        "G",
        "H",
        "I",
        "J",
        "K",
        "L",
        "N",
    ]
    assert "M" not in {identity.chain_id for identity in identities}


def test_smoke_build_skips_solvent_allocation_when_resolved_count_is_zero(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Allow the full smoke build path to complete without zero-count solvent identities."""

    smoke = load_smoke_module()

    class FakeForce:
        """Minimal force object for smoke build unit tests."""

        PME = "PME"

        def setNonbondedMethod(self, method: object) -> None:  # noqa: N802
            """Accept a nonbonded method without constructing OpenMM objects."""

        def setCutoffDistance(self, cutoff: object) -> None:  # noqa: N802
            """Accept a cutoff without constructing OpenMM objects."""

        def setUseSwitchingFunction(self, enabled: bool) -> None:  # noqa: N802
            """Accept switching configuration without constructing OpenMM objects."""

        def setSwitchingDistance(self, distance: object) -> None:  # noqa: N802
            """Accept switching distance without constructing OpenMM objects."""

        def setUseDispersionCorrection(self, enabled: bool) -> None:  # noqa: N802
            """Accept dispersion configuration without constructing OpenMM objects."""

        def addException(self, *args: object, **kwargs: object) -> None:  # noqa: N802
            """Accept exception parameters without constructing OpenMM objects."""

    class FakeSystem:
        """Minimal OpenMM system replacement for smoke build unit tests."""

        def __init__(self) -> None:
            self.forces = []

        def setDefaultPeriodicBoxVectors(self, *vectors: object) -> None:  # noqa: N802
            """Accept periodic box vectors without constructing OpenMM objects."""

        def addForce(self, force: object) -> None:  # noqa: N802
            """Record added forces for summary metadata."""

            self.forces.append(force)

        def getNumParticles(self) -> int:  # noqa: N802
            """Return a stable particle count for summary metadata."""

            return 0

    class FakeTopology:
        """Minimal topology replacement for smoke build unit tests."""

        def setPeriodicBoxVectors(self, vectors: object) -> None:  # noqa: N802
            """Accept periodic box vectors without constructing OpenMM objects."""

    class FakeVec3(tuple):
        """Tuple-backed vector that supports OpenMM-like unit multiplication."""

        def __new__(cls, *coordinates: float) -> FakeVec3:
            return super().__new__(cls, coordinates)

        def __mul__(self, scalar: object) -> FakeVec3:
            return self

    fake_openmm = SimpleNamespace(
        System=FakeSystem,
        NonbondedForce=FakeForce,
        HarmonicBondForce=FakeForce,
        HarmonicAngleForce=FakeForce,
        PeriodicTorsionForce=FakeForce,
        CMMotionRemover=lambda: SimpleNamespace(),
        Vec3=FakeVec3,
    )
    fake_unit = SimpleNamespace(
        nanometer=1.0,
        elementary_charge=1.0,
        kilojoule_per_mole=1.0,
        Quantity=lambda values, unit: (values, unit),
    )
    modules = SimpleNamespace(
        openmm=fake_openmm,
        app=SimpleNamespace(Topology=FakeTopology),
        unit=fake_unit,
    )
    plan = SimpleNamespace(
        config=SimpleNamespace(
            parameterization=SimpleNamespace(nonbonded_cutoff=1.0),
            packing=SimpleNamespace(packmol=SimpleNamespace(tolerance=1.8)),
        ),
        slab=SimpleNamespace(positions_nm=((0.0, 0.0, 0.0),)),
        sam_placements=SimpleNamespace(placements=(SimpleNamespace(),)),
        box_plan=SimpleNamespace(dimensions_nm=(2.0, 2.0, 4.0)),
    )

    def add_pd_position(*args: object, **kwargs: object) -> None:
        """Populate Pd positions without adding chemical details."""

        positions_nm = args[6]
        positions_nm.append((1.0, 1.0, 1.0))

    def add_sam_position(*args: object, **kwargs: object) -> None:
        """Populate SAM positions without adding chemical details."""

        positions_nm = args[9]
        positions_nm.append((1.0, 1.0, 1.2))

    def add_reactant_position(*args: object, **kwargs: object) -> None:
        """Populate reactant positions without adding chemical details."""

        positions_nm = args[9]
        positions_nm.append((1.0, 1.0, 2.0))

    def fail_solvent_call(*args: object, **kwargs: object) -> None:
        """Fail if the zero-count branch tries to pack or add solvent."""

        raise AssertionError("zero-count solvent should not be packed or added")

    monkeypatch.setattr(smoke, "add_pd_slab", add_pd_position)
    monkeypatch.setattr(smoke, "add_sam_layer", add_sam_position)
    monkeypatch.setattr(
        smoke,
        "place_reactants_above_surface",
        lambda *args: (((1.0, 1.0, 2.0),),),
    )
    monkeypatch.setattr(smoke, "add_reactants", add_reactant_position)
    monkeypatch.setattr(smoke, "pack_solution_with_packmol", fail_solvent_call)
    monkeypatch.setattr(smoke, "add_solvent_molecules", fail_solvent_call)

    smoke_build = smoke.build_openmm_smoke_system(
        modules,
        plan,
        SimpleNamespace(),
        SimpleNamespace(),
        SimpleNamespace(name="ethanol"),
        solvent_count=0,
        reactant_count=1,
        sulfur_height_nm=0.18,
        solvent_padding_nm=1.0,
        packmol_working_dir=tmp_path,
        pressure_bar=1.0,
        temperature_k=300.0,
        pd_s_sigma_nm=0.22,
        pd_s_epsilon_kcal_mol=1.0,
    )

    assert smoke_build.solvent_count == 0
    assert "ethanol" not in smoke_build.component_chain_ranges
    assert smoke_build.runtime_solvent_geometry.molecule_counts == {"ethanol": 0}


def test_smoke_fixed_solute_containment_accepts_tolerance() -> None:
    """Allow tiny numerical drift at runtime box boundaries."""

    smoke = load_smoke_module()

    smoke.ensure_positions_inside_box(
        ((-1.0e-10, 0.5, 1.0 + 1.0e-10),),
        (1.0, 1.0, 1.0),
        context="test fixed solute",
    )


def test_smoke_fixed_solute_containment_rejects_outside_box() -> None:
    """Reject shifted fixed-solute coordinates outside runtime dimensions."""

    smoke = load_smoke_module()

    with pytest.raises(
        ValueError,
        match=r"test fixed solute atom 2 z-coordinate 1\.1 nm lies outside runtime box",
    ):
        smoke.ensure_positions_inside_box(
            ((0.5, 0.5, 0.5), (0.5, 0.5, 1.1)),
            (1.0, 1.0, 1.0),
            context="test fixed solute",
        )


def test_smoke_solvent_clearance_uses_configured_packmol_tolerance() -> None:
    """Use non-default Packmol tolerance when sizing runtime solvent clearance."""

    smoke = load_smoke_module()
    plan = SimpleNamespace(
        config=SimpleNamespace(
            packing=SimpleNamespace(packmol=SimpleNamespace(tolerance=2.5)),
        ),
        box_plan=SimpleNamespace(dimensions_nm=(2.0, 2.0, 1.0)),
    )

    box, z_shift, regions = smoke.actual_solvent_packing_geometry(
        plan,
        ((0.5, 0.5, 0.0), (0.5, 0.5, 1.0)),
        2.0,
    )

    assert box == pytest.approx((2.0, 2.0, 3.0))
    assert z_shift == pytest.approx(1.0)
    assert regions[0][2] == pytest.approx((0.0, 0.75))
    assert regions[1][2] == pytest.approx((2.25, 3.0))


def test_smoke_runtime_solvent_bounds_are_shifted() -> None:
    """Report z-bounds in the same runtime frame as Packmol regions."""

    smoke = load_smoke_module()
    plan = SimpleNamespace(
        config=SimpleNamespace(
            packing=SimpleNamespace(packmol=SimpleNamespace(tolerance=1.0)),
        ),
        box_plan=SimpleNamespace(dimensions_nm=(2.0, 2.0, 1.0)),
    )

    geometry = smoke.build_runtime_solvent_geometry(
        plan,
        ((0.5, 0.5, -1.0), (0.5, 0.5, 1.0)),
        2.0,
        solvent_boundary_positions_nm=((0.5, 0.5, -0.5), (0.5, 0.5, 0.5)),
    )

    assert geometry.z_shift_nm == pytest.approx(1.5)
    assert geometry.solvent_boundary_z_bounds_nm == pytest.approx((1.0, 2.0))
    assert geometry.fixed_solute_z_bounds_nm == pytest.approx((0.5, 2.5))


def test_smoke_nonzero_solvent_rejects_clearance_removed_regions() -> None:
    """Do not silently accept empty solvent reservoirs for requested solvent."""

    smoke = load_smoke_module()
    geometry = smoke.RuntimeSolventGeometry(
        solvent_boundary_z_bounds_nm=(1.0, 2.0),
        fixed_solute_z_bounds_nm=(0.5, 2.5),
        solvent_regions_nm=(),
        solvent_count_planning_volume_nm3=0.0,
        solvent_padding_nm=0.2,
        solvent_padding_per_face_nm=0.1,
        solvent_clearance_nm=0.2,
        dimensions_nm=(2.0, 2.0, 3.0),
        z_shift_nm=1.5,
        molecule_counts={},
    )

    with pytest.raises(ValueError, match="padding per face to exceed Packmol clearance"):
        smoke.validate_runtime_solvent_regions_for_count(geometry, 1)

    smoke.validate_runtime_solvent_regions_for_count(geometry, 0)


def test_smoke_zero_solvent_count_skips_packmol_after_containment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Return no solvent positions for zero counts without writing Packmol files."""

    smoke = load_smoke_module()

    def fail_if_packmol_runs(*args: object, **kwargs: object) -> None:
        """Fail if zero-count placement reaches Packmol execution."""

        raise AssertionError("Packmol should not run for zero solvent count")

    monkeypatch.setattr(smoke, "run_packmol", fail_if_packmol_runs)

    solution = smoke.pack_solution_with_packmol(
        topology=SimpleNamespace(),
        solute_positions_nm=((0.5, 0.5, 0.5),),
        solvent_template=SimpleNamespace(),
        solvent_count=0,
        box_dimensions_nm=(1.0, 1.0, 1.0),
        solvent_regions_nm=(),
        working_dir=tmp_path / "packmol",
    )

    assert solution.solvent_positions_nm == ()
    assert not (tmp_path / "packmol").exists()


def test_smoke_zero_solvent_count_still_validates_fixed_solute_containment(
    tmp_path: Path,
) -> None:
    """Reject invalid fixed solute coordinates before zero-count short-circuiting."""

    smoke = load_smoke_module()

    with pytest.raises(ValueError, match="fixed solute Packmol atom 1"):
        smoke.pack_solution_with_packmol(
            topology=SimpleNamespace(),
            solute_positions_nm=((1.5, 0.5, 0.5),),
            solvent_template=SimpleNamespace(),
            solvent_count=0,
            box_dimensions_nm=(1.0, 1.0, 1.0),
            solvent_regions_nm=(),
            working_dir=tmp_path / "packmol",
        )
