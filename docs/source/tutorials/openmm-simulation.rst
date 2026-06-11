OpenMM simulation from SAMMD files
==================================

This page shows how to run OpenMM after SAMMD has written the OpenFF/OpenMM
files. The split is simple: SAMMD builds/exports files. OpenMM runs
minimization, equilibration, production, trajectories, and reporters.

Use the raw OpenMM Python API here, not a SAMMD OpenMM wrapper.

Before starting
---------------

From the repository root, build the files in a CUDA-labeled pixi environment.
Run ``nvidia-smi`` on the machine first, then choose an environment whose CUDA
version is not newer than the CUDA version shown there. For example, use
``cuda-12-4`` for CU Boulder Blanca older-GPU nodes and ``cuda-12-6`` on PSC
Bridges2. The default example here uses ``cuda-12-4``:

.. tip::

   You can run each command with ``pixi run -e cuda-12-4 ...`` or enter the
   environment once with ``pixi shell -e cuda-12-4``. Leave the environment with
   ``exit`` before switching to another pixi environment.

.. code-block:: bash

   pixi run -e cuda-12-4 sammd build sammd.yaml --output-dir outputs --overwrite --full

This writes files such as ``interchange.json``, ``solvated_system.cif``,
``solvated_system_pymol.pdb``, and ``anchor_metadata.json``. The
``solvated_system.cif`` file is a PDBx/mmCIF structure using SAMMD's stable
``.cif`` artifact name. The ``solvated_system_pymol.pdb`` file includes explicit
``CONECT`` records for PyMOL visualization. In this tutorial we load
``interchange.json`` first, then ask OpenFF Interchange for the OpenMM objects.

Full copy/paste script
----------------------

Save this as a Python script or run it cell by cell in a notebook from the
repository root.

.. code-block:: python

   import math
   from pathlib import Path

   import matplotlib.pyplot as plt
   import pandas as pd
   from openff.interchange import Interchange
   from openmm import LangevinMiddleIntegrator, MonteCarloAnisotropicBarostat, MonteCarloBarostat, Vec3, unit
   from openmm.app import DCDReporter, Simulation, StateDataReporter
   from sammd.backends.interchange_plugins import register_interchange_plugin_collection

   output_dir = Path("outputs")
   interchange_path = output_dir / "interchange.json"
   trajectory_path = output_dir / "trajectory.dcd"
   thermo_path = output_dir / "thermodynamics.csv"

   required_paths = [interchange_path]
   missing_paths = [path for path in required_paths if not path.is_file()]
   if missing_paths:
       missing = ", ".join(str(path) for path in missing_paths)
       raise FileNotFoundError(f"Missing SAMMD output file(s): {missing}")

   register_interchange_plugin_collection()
   interchange = Interchange.model_validate_json(interchange_path.read_text(encoding="utf-8"))
   system = interchange.to_openmm()
   topology = interchange.to_openmm_topology()
   positions = interchange.get_positions(include_virtual_sites=True).to_openmm()

   temperature = 300.0 * unit.kelvin
   friction = 1.0 / unit.picosecond
   timestep = 2.0 * unit.femtosecond
   equilibration_time_ps = 100.0
   production_time_ns = 10.0
   desired_trajectory_frames = 300
   desired_thermo_points = 1000

   def steps_from_time(time, time_unit, timestep):
       """Convert a desired time to an integer number of OpenMM steps."""
       return int(round((time * time_unit / timestep).value_in_unit(unit.dimensionless)))

   def interval_from_count(total_steps, desired_count):
       """Convert a desired number of saved points to an integer report interval."""
       if desired_count < 1:
           raise ValueError("desired_count must be at least 1")
       return max(1, total_steps // desired_count)

   equilibration_steps = steps_from_time(equilibration_time_ps, unit.picosecond, timestep)
   production_steps = steps_from_time(production_time_ns, unit.nanosecond, timestep)
   trajectory_interval = interval_from_count(production_steps, desired_trajectory_frames)
   thermo_interval = interval_from_count(production_steps, desired_thermo_points)

   print(f"Equilibration steps: {equilibration_steps}")
   print(f"Production steps: {production_steps}")
   print(f"Trajectory interval: every {trajectory_interval} steps")
   print(f"Thermo interval: every {thermo_interval} steps")

   integrator = LangevinMiddleIntegrator(temperature, friction, timestep)
   simulation = Simulation(topology, system, integrator)
   simulation.context.setPositions(positions)

   initial_state = simulation.context.getState(getEnergy=True)
   initial_energy = initial_state.getPotentialEnergy()
   initial_energy_value = initial_energy.value_in_unit(unit.kilojoule_per_mole)
   if not math.isfinite(initial_energy_value):
       raise ValueError(f"Initial potential energy is not finite: {initial_energy}")
   print(f"Initial potential energy: {initial_energy}")

   simulation.minimizeEnergy()
   minimized_state = simulation.context.getState(getEnergy=True)
   print(f"Minimized potential energy: {minimized_state.getPotentialEnergy()}")

   simulation.context.setVelocitiesToTemperature(temperature)
   simulation.step(equilibration_steps)

   simulation.reporters.append(DCDReporter(str(trajectory_path), trajectory_interval))
   simulation.reporters.append(
       StateDataReporter(
           str(thermo_path),
           thermo_interval,
           step=True,
           time=True,
           potentialEnergy=True,
           kineticEnergy=True,
           totalEnergy=True,
           temperature=True,
           speed=True,
           separator=",",
       )
   )
   simulation.step(production_steps)

   thermo = pd.read_csv(thermo_path)
   print(thermo.head())

   plt.figure(figsize=(7, 4))
   plt.plot(thermo["Time (ps)"], thermo["Potential Energy (kJ/mole)"])
   plt.xlabel("Time (ps)")
   plt.ylabel("Potential energy (kJ/mol)")
   plt.tight_layout()
   plt.show()

   plt.figure(figsize=(7, 4))
   plt.plot(thermo["Time (ps)"], thermo["Temperature (K)"])
   plt.xlabel("Time (ps)")
   plt.ylabel("Temperature (K)")
   plt.tight_layout()
   plt.show()

Why the step helper functions matter
------------------------------------

OpenMM uses integer step counts and integer reporter intervals. Students often
start from human numbers such as "10 ns" or "300 frames". The helper functions
``steps_from_time`` and ``interval_from_count`` convert those numbers for you, so
you do not need to do the unit math by hand. In the example above,
``production_time_ns = 10.0``, ``desired_trajectory_frames = 300``, and
``desired_thermo_points = 1000``.

Initial energy check
--------------------

The first energy can be a large positive number. That is common for a starting
structure and is one reason we minimize before MD. The first important check is
not whether the number is small. The first important check is that the energy is
finite. ``nan`` or ``inf`` usually means something is seriously wrong with the
starting system.

NVT first
---------

This tutorial uses NVT for equilibration and production. In NVT, the number of
particles and the volume stay fixed, and the thermostat targets the chosen
temperature. The instantaneous temperature will still fluctuate. This is a good
default first run because the box shape does not change while you learn the
workflow.

Optional NPT note
-----------------

Use NPT only when pressure control makes sense for your system. Choose one of
these examples, not both. For a bulk fluid, add ``MonteCarloBarostat`` before
creating ``Simulation``:

.. code-block:: python

   pressure = 1.0 * unit.atmosphere
   system.addForce(MonteCarloBarostat(pressure, temperature))
   integrator = LangevinMiddleIntegrator(temperature, friction, timestep)
   simulation = Simulation(topology, system, integrator)

For a slab or interface, it is often safer to keep ``x`` and ``y`` fixed and
allow only ``z`` to change. Use ``MonteCarloAnisotropicBarostat`` before creating
``Simulation``:

.. code-block:: python

   pressure = 1.0 * unit.atmosphere
   system.addForce(
       MonteCarloAnisotropicBarostat(
           Vec3(1.0, 1.0, 1.0) * unit.atmosphere,
           temperature,
           False,
           False,
           True,
       )
   )
   integrator = LangevinMiddleIntegrator(temperature, friction, timestep)
   simulation = Simulation(topology, system, integrator)

A barostat does not replace the thermostat. Keep the ``LangevinMiddleIntegrator``
for temperature control.

About Interchange Reload
------------------------

SAMMD stores the sulfur-metal pair overrides in a plugin collection inside
``interchange.json``. Register that collection before calling
``Interchange.model_validate_json`` so ``interchange.to_openmm()`` can apply the
OpenMM exceptions from the reloaded artifact.

View the DCD in PyMOL
---------------------

After production finishes, open the starting structure and then load the DCD
trajectory into the same object:

.. code-block:: text

   load outputs/solvated_system_pymol.pdb, sammd_system
   load_traj outputs/trajectory.dcd, sammd_system

The DCD uses the atom order from the OpenMM topology. Loading the PyMOL PDB first
gives PyMOL the atoms and explicit connectivity, then ``load_traj`` adds the
frames.
