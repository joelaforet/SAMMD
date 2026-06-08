"""Tests for optional OpenMM runtime helpers."""

import builtins
import math
import sys

import pytest

from sammd.backends.openmm_runtime import (
    add_position_restraints,
    add_sulfur_metal_lj_exceptions,
    add_sulfur_metal_lj_scaling,
    create_langevin_integrator,
    create_openmm_simulation,
    require_openmm,
)
from sammd.core.config import ReporterConfig
from sammd.core.io import OutputPaths


class FakeUnitValue:
    """Small unit-tagged value for fake OpenMM unit operations."""

    def __init__(self, value, unit_name):
        self.value = value
        self.unit_name = unit_name


class FakeUnit:
    """Minimal unit object supporting multiplication and division."""

    def __init__(self, name):
        self.name = name

    def __rmul__(self, value):
        return FakeUnitValue(value, self.name)

    def __rtruediv__(self, value):
        return FakeUnitValue(value, f"1/{self.name}")

    def __pow__(self, power):
        return FakeUnit(f"{self.name}^{power}")


class FakeUnitModule:
    """Minimal OpenMM unit namespace."""

    kelvin = FakeUnit("kelvin")
    picosecond = FakeUnit("picosecond")
    femtoseconds = FakeUnit("femtoseconds")
    nanometer = FakeUnit("nanometer")
    kilojoule_per_mole = FakeUnit("kilojoule_per_mole")
    elementary_charge = FakeUnit("elementary_charge")


class FakeOpenMM:
    """Minimal OpenMM namespace for runtime helper tests."""

    class LangevinIntegrator:
        """Fake integrator storing constructor arguments."""

        def __init__(self, temperature, friction, timestep):
            self.temperature = temperature
            self.friction = friction
            self.timestep = timestep

    class CustomExternalForce:
        """Fake external force storing restraint configuration."""

        def __init__(self, expression):
            self.expression = expression
            self.global_parameters = []
            self.per_particle_parameters = []
            self.particles = []

        def addGlobalParameter(self, name, value):  # noqa: N802
            self.global_parameters.append((name, value))

        def addPerParticleParameter(self, name):  # noqa: N802
            self.per_particle_parameters.append(name)

        def addParticle(self, atom_index, parameters):  # noqa: N802
            self.particles.append((atom_index, parameters))

    class CustomBondForce:
        """Fake custom bond force storing pair correction parameters."""

        def __init__(self, expression):
            self.expression = expression
            self.per_bond_parameters = []
            self.bonds = []
            self.uses_pbc = False

        def addPerBondParameter(self, name):  # noqa: N802
            self.per_bond_parameters.append(name)

        def setUsesPeriodicBoundaryConditions(self, value):  # noqa: N802
            self.uses_pbc = value

        def addBond(self, atom_index, other_atom_index, parameters):  # noqa: N802
            self.bonds.append((atom_index, other_atom_index, parameters))

    class Platform:
        """Fake platform registry."""

        @staticmethod
        def getPlatformByName(name):  # noqa: N802
            return f"platform:{name}"


class FakeSystem:
    """Minimal OpenMM system with particles and forces."""

    def __init__(self, particle_count=2):
        self.particle_count = particle_count
        self.forces = []

    def getNumParticles(self):  # noqa: N802
        return self.particle_count

    def addForce(self, force):  # noqa: N802
        self.forces.append(force)
        return len(self.forces) - 1

    def getNumForces(self):  # noqa: N802
        return len(self.forces)

    def getForce(self, index):  # noqa: N802
        return self.forces[index]


class NonbondedForce:
    """Fake NonbondedForce identified by class name."""

    def __init__(self, parameters, exceptions=()):
        self.parameters = parameters
        self.exceptions = list(exceptions)

    def getParticleParameters(self, index):  # noqa: N802
        return self.parameters[index]

    def getNumExceptions(self):  # noqa: N802
        return len(self.exceptions)

    def getExceptionParameters(self, index):  # noqa: N802
        return self.exceptions[index]

    def addException(self, atom1, atom2, chargeprod, sigma, epsilon, replace=False):  # noqa: N802
        self.exceptions.append((atom1, atom2, chargeprod, sigma, epsilon, replace))
        return len(self.exceptions) - 1


class FakeApp:
    """Minimal OpenMM app namespace for simulation and reporters."""

    class DCDReporter:
        """Fake trajectory reporter."""

        def __init__(self, file, report_interval):
            self.file = file
            self.report_interval = report_interval

    class StateDataReporter:
        """Fake thermodynamic reporter."""

        def __init__(self, file, report_interval, **kwargs):
            self.file = file
            self.report_interval = report_interval
            self.kwargs = kwargs

    class Simulation:
        """Fake simulation storing constructor values and positions."""

        def __init__(self, topology, system, integrator, platform=None):
            self.topology = topology
            self.system = system
            self.integrator = integrator
            self.platform = platform
            self.reporters = []
            self.context = FakeContext()


class FakeContext:
    """Fake simulation context storing positions."""

    def setPositions(self, positions):  # noqa: N802
        self.positions = positions



def test_openmm_runtime_module_does_not_import_openmm_at_import_time() -> None:
    """Keep optional runtime helpers importable without OpenMM."""

    sys.modules.pop("openmm", None)
    import sammd.backends.openmm_runtime

    assert sammd.backends.openmm_runtime is not None
    assert "openmm" not in sys.modules


def test_require_openmm_error_mentions_science_pixi_environment(monkeypatch) -> None:
    """Raise a clear optional dependency error when OpenMM is unavailable."""

    real_import = builtins.__import__

    def fake_import(name, globals_=None, locals_=None, fromlist=(), level=0):
        """Block OpenMM imports while preserving other imports."""

        if name == "openmm":
            raise ImportError("blocked OpenMM")
        return real_import(name, globals_, locals_, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(ImportError, match="CUDA pixi environment"):
        require_openmm()


def test_create_langevin_integrator_uses_expected_units() -> None:
    """Build a Langevin integrator from kelvin, inverse ps, and fs inputs."""

    integrator = create_langevin_integrator(
        310.0,
        2.0,
        1.0,
        openmm_module=FakeOpenMM,
        unit_module=FakeUnitModule,
    )

    assert integrator.temperature.value == 310.0
    assert integrator.temperature.unit_name == "kelvin"
    assert integrator.friction.value == 2.0
    assert integrator.friction.unit_name == "1/picosecond"
    assert integrator.timestep.value == 1.0
    assert integrator.timestep.unit_name == "femtoseconds"


def test_create_langevin_integrator_rejects_invalid_values() -> None:
    """Reject nonphysical integrator inputs before importing OpenMM."""

    with pytest.raises(ValueError, match="temperature_k"):
        create_langevin_integrator(
            0.0,
            1.0,
            1.0,
            openmm_module=FakeOpenMM,
            unit_module=FakeUnitModule,
        )


def test_add_position_restraints_constructs_expected_force() -> None:
    """Add harmonic restraints with one particle per selected atom."""

    system = FakeSystem(particle_count=3)
    force = add_position_restraints(
        system,
        [0, 2],
        [(0.1, 0.2, 0.3), (0.4, 0.5, 0.6)],
        k_kj_mol_nm2=10000.0,
        openmm_module=FakeOpenMM,
    )

    assert force.expression == "0.5*k*((x-x0)^2+(y-y0)^2+(z-z0)^2)"
    assert force.global_parameters == [("k", 10000.0)]
    assert force.per_particle_parameters == ["x0", "y0", "z0"]
    assert force.particles == [(0, [0.1, 0.2, 0.3]), (2, [0.4, 0.5, 0.6])]
    assert system.forces == [force]


def test_add_sulfur_metal_lj_exceptions_replaces_selected_pairs() -> None:
    """Use smoke-equivalent selected-pair NonbondedForce exceptions."""

    system = FakeSystem(particle_count=3)
    nonbonded = NonbondedForce(parameters=[(0, 0, 0), (0, 0, 0), (0, 0, 0)])
    system.addForce(nonbonded)

    metadata = add_sulfur_metal_lj_exceptions(
        system,
        [(1, 0), (1, 2)],
        sigma_nm=0.22,
        epsilon_kj_mol=8.368,
        unit_module=FakeUnitModule,
    )

    assert metadata.pairs_requested == 2
    assert metadata.pairs_added == 2
    assert metadata.sigma_nm == (0.22, 0.22)
    assert metadata.epsilon_kj_mol == (8.368, 8.368)
    assert nonbonded.exceptions[0][-1] is True
    assert nonbonded.exceptions[1][-1] is True


def test_add_position_restraints_validates_inputs() -> None:
    """Reject invalid restraint constants, indices, and coordinates."""

    system = FakeSystem(particle_count=2)
    with pytest.raises(ValueError, match="k_kj_mol_nm2"):
        add_position_restraints(system, [0], [(0.0, 0.0, 0.0)], k_kj_mol_nm2=-1.0)
    with pytest.raises(ValueError, match="outside system"):
        add_position_restraints(system, [2], [(0.0, 0.0, 0.0)], openmm_module=FakeOpenMM)
    with pytest.raises(ValueError, match="non-negative integers"):
        add_position_restraints(system, [True], [(0.0, 0.0, 0.0)], openmm_module=FakeOpenMM)
    with pytest.raises(ValueError, match="duplicates"):
        add_position_restraints(
            system,
            [0, 0],
            [(0.0, 0.0, 0.0), (0.1, 0.1, 0.1)],
            openmm_module=FakeOpenMM,
        )
    with pytest.raises(ValueError, match="length must match"):
        add_position_restraints(system, [0, 1], [(0.0, 0.0, 0.0)], openmm_module=FakeOpenMM)
    with pytest.raises(ValueError, match="finite xyz"):
        add_position_restraints(system, [0], [(math.nan, 0.0, 0.0)], openmm_module=FakeOpenMM)


def test_lj_scaling_validates_scale_factor_and_noop() -> None:
    """Validate scale factor and return metadata without adding a force at 1x."""

    system = FakeSystem(particle_count=2)
    system.addForce(NonbondedForce([(0.0, 0.30, 2.0), (0.0, 0.50, 8.0)]))
    with pytest.raises(ValueError, match="scale_factor"):
        add_sulfur_metal_lj_scaling(system, [(0, 1)], scale_factor=0.0)

    metadata = add_sulfur_metal_lj_scaling(system, [(0, 1)], scale_factor=1.0)

    assert metadata.force_added is False
    assert metadata.pairs_requested == 1
    assert metadata.pairs_added == 0
    assert len(system.forces) == 1


def test_lj_scaling_scale_factor_one_still_requires_nonbonded_force() -> None:
    """Require baseline nonbonded parameters even when scaling is a no-op."""

    system = FakeSystem(particle_count=2)

    with pytest.raises(ValueError, match="NonbondedForce"):
        add_sulfur_metal_lj_scaling(system, [(0, 1)], scale_factor=1.0)


@pytest.mark.parametrize(
    ("pairs", "error"),
    [
        ([("0", 1)], "integers"),
        ([(0.0, 1)], "integers"),
        ([(False, 1)], "integers"),
        ([(0,)], "exactly two"),
        ([(0, 1, 2)], "exactly two"),
        ([(0, 0)], "self-pairs"),
        ([(0, 1), (0, 1)], "duplicate"),
        ([(0, 1), (1, 0)], "duplicate"),
        ([(-1, 1)], "non-negative"),
    ],
)
def test_lj_scaling_rejects_invalid_pairs(pairs, error) -> None:
    """Reject malformed or ambiguous sulfur-metal pair specifications."""

    system = FakeSystem(particle_count=2)

    with pytest.raises(ValueError, match=error):
        add_sulfur_metal_lj_scaling(system, pairs, scale_factor=1.0)


def test_lj_scaling_raises_without_nonbonded_force() -> None:
    """Require existing NonbondedForce parameters for anchor scaling."""

    system = FakeSystem(particle_count=2)

    with pytest.raises(ValueError, match="NonbondedForce"):
        add_sulfur_metal_lj_scaling(
            system,
            [(0, 1)],
            scale_factor=4.0,
            openmm_module=FakeOpenMM,
            unit_module=FakeUnitModule,
        )


def test_lj_scaling_rejects_pairs_with_nonbonded_exceptions() -> None:
    """Reject explicit pairs covered by existing exceptions or exclusions."""

    system = FakeSystem(particle_count=2)
    system.addForce(
        NonbondedForce(
            [(0.0, 0.30, 2.0), (0.0, 0.50, 8.0)],
            exceptions=[(1, 0, 0.0, 0.0, 0.0)],
        )
    )

    with pytest.raises(ValueError, match="exceptions or exclusions"):
        add_sulfur_metal_lj_scaling(
            system,
            [(0, 1)],
            scale_factor=4.0,
            openmm_module=FakeOpenMM,
            unit_module=FakeUnitModule,
        )


def test_lj_scaling_adds_expected_custom_bond_force() -> None:
    """Compute Lorentz-Berthelot sigma and scaled epsilon correction."""

    system = FakeSystem(particle_count=2)
    system.addForce(NonbondedForce([(0.0, 0.30, 2.0), (0.0, 0.50, 8.0)]))

    metadata = add_sulfur_metal_lj_scaling(
        system,
        [(0, 1)],
        scale_factor=4.0,
        openmm_module=FakeOpenMM,
        unit_module=FakeUnitModule,
    )

    force = system.forces[1]
    assert metadata.force_added is True
    assert metadata.force_index == 1
    assert metadata.sigma_nm == pytest.approx((0.40,))
    assert metadata.epsilon_delta_kj_mol == pytest.approx((12.0,))
    assert force.expression == "4*epsilon_delta*((sigma/r)^12-(sigma/r)^6)"
    assert force.uses_pbc is True
    assert force.bonds == [(0, 1, [0.40, 12.0])]


def test_create_openmm_simulation_attaches_reporters(tmp_path) -> None:
    """Configure a simulation with positions, platform, and SAMMD reporters."""

    output_paths = OutputPaths(
        topology=tmp_path / "topology.cif",
        trajectory=tmp_path / "traj/run.dcd",
        thermodynamics=tmp_path / "reports/state.csv",
    )
    simulation = create_openmm_simulation(
        topology="topology",
        system="system",
        positions="positions",
        reporting_config=ReporterConfig(interval_steps=25, fields=["step"]),
        output_paths=output_paths,
        platform_name="Reference",
        openmm_module=FakeOpenMM,
        app_module=FakeApp,
        unit_module=FakeUnitModule,
    )

    assert simulation.platform == "platform:Reference"
    assert simulation.context.positions == "positions"
    assert len(simulation.reporters) == 2
    assert simulation.reporters[0].file == str(tmp_path / "traj/run.dcd")
    assert simulation.reporters[1].kwargs == {"separator": ",", "step": True}
    assert not (tmp_path / "traj").exists()
    assert not (tmp_path / "reports").exists()


def test_create_openmm_simulation_prepares_reporter_directories_when_requested(tmp_path) -> None:
    """Create reporter directories only when simulation setup explicitly opts in."""

    output_paths = OutputPaths(
        topology=tmp_path / "topology.cif",
        trajectory=tmp_path / "traj/run.dcd",
        thermodynamics=tmp_path / "reports/state.csv",
    )

    create_openmm_simulation(
        topology="topology",
        system="system",
        positions="positions",
        reporting_config=ReporterConfig(interval_steps=25, fields=["step"]),
        output_paths=output_paths,
        openmm_module=FakeOpenMM,
        app_module=FakeApp,
        unit_module=FakeUnitModule,
        prepare_reporter_directories=True,
    )

    assert (tmp_path / "traj").is_dir()
    assert (tmp_path / "reports").is_dir()
