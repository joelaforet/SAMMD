Recommended build-to-OpenMM workflow
====================================

This tutorial shows the recommended path for students. SAMMD checks the YAML,
builds the same starting structure each time, and writes files you can load in
OpenMM. SAMMD builds; OpenMM runs.

.. note::

   Use the ``default`` environment for initialization, validation, and builds.
   CUDA-labeled environments are only needed when you want a specific GPU OpenMM
   pin for downstream simulation work.

1. Create config
----------------

From the repository root, create a starter project directory:

.. code-block:: bash

   pixi run sammd init -o sammd-project

The command writes ``sammd-project/sammd.yaml``. The template follows the
defaults summarized in :doc:`yaml-configuration`. It describes the surface, SAM,
solution composition, and output names used for the build step. It does not
describe minimization, equilibration, production MD, thermostats, barostats,
trajectories, or reporters.

2. Validate config
------------------

Validate the YAML before building anything:

.. code-block:: bash

   pixi run sammd validate sammd-project/sammd.yaml

Validation checks that SAMMD can interpret the file and reports configuration
errors before any output files are written.

3. Build the starting model
---------------------------

Build the starting model:

.. code-block:: bash

   pixi run sammd build sammd-project/sammd.yaml --output-dir sammd-project/outputs --overwrite

This command writes these output files:

* ``sam_grafting_density.cif`` as a PDBx/mmCIF visual smoke test for the Pd
  slab and SAM sulfur anchor positions
* ``build_summary.json`` for a JSON build report that scripts can read
* ``resolved_config.yaml`` for the exact validated input used for the build
* ``interchange.json`` for the primary OpenFF Interchange export
* ``solvated_system.cif`` for the full slab + SAMs + reactants + solvent
  PDBx/mmCIF coordinates you can inspect and load in OpenMM
* ``solvated_system_pymol.pdb`` for PyMOL visualization with explicit ``CONECT``
  records
* ``anchor_metadata.json`` for SAM anchor metadata

The result includes the validated configuration and resolved output paths. It
also includes a centered registered Fcc(111) slab, using Pd(111) by default,
internal ``fcc_hollow`` binding sites, sulfur atoms marking the planned SAM
anchors, approximate solution counts, full SAM molecule coordinates, and a
parameterized Interchange export.

4. Inspect outputs
------------------

Open ``sammd-project/outputs/sam_grafting_density.cif`` in a molecule viewer
such as PyMOL to inspect the configured surface and the sulfur anchor atoms.
This is a useful smoke test: you can see the Pd(111) slab geometry and check
whether the thiol sulfur atoms land at the intended three-fold hollow sites with
the expected grafting density. It does not contain the rest of each SAM molecule
or solvent coordinates.

Use ``sammd-project/outputs/build_summary.json`` to confirm the same build
choices in a machine-readable form. Use
``sammd-project/outputs/resolved_config.yaml`` when you need the exact validated
YAML that produced the plan.

5. Interchange output files
---------------------------

``sammd build`` writes files for OpenFF/OpenMM in the default environment. Use a
CUDA-labeled pixi environment only when you want a matching GPU OpenMM pin. Run
``nvidia-smi`` on the machine first, then choose an environment whose CUDA
version is not newer than the CUDA version shown there. For example, use
``cuda-12-4`` for CU Boulder Blanca older-GPU nodes and ``cuda-12-6`` for PSC
Bridges2.

The build command writes these Interchange files:

* ``interchange.json`` for the primary OpenFF Interchange export
* ``solvated_system.cif`` for the full slab + SAMs + reactants + solvent
  PDBx/mmCIF coordinates you can inspect and load in OpenMM
* ``solvated_system_pymol.pdb`` for PyMOL visualization with explicit ``CONECT``
  records
* ``anchor_metadata.json`` for SAM anchor metadata

The ``interchange.json`` file is OpenFF Interchange JSON written with
``Interchange.model_dump_json`` and reloaded with
``Interchange.model_validate_json``. Pre-1.0 Interchange JSON compatibility is
not guaranteed across OpenFF Interchange versions. Configs that include salt are
rejected until Interchange export supports salt.

Inspect ``sammd-project/outputs/solvated_system.cif`` if you want to see full
SAM molecules, solvent, and reactants.
``sammd-project/outputs/sam_grafting_density.cif`` remains the separate
grafting-density smoke test.

6. Use these files with OpenMM
------------------------------

After Interchange export writes ``interchange.json`` and the other output files,
students use them in their own OpenMM Python API script. Follow these steps:

* reload the OpenFF Interchange data from ``interchange.json`` with
  ``Interchange.model_validate_json``
* export an OpenMM ``System`` from that Interchange object with
  ``interchange.to_openmm()``
* load positions from ``solvated_system.cif`` for the constructed coordinates
* create and run a raw OpenMM ``Simulation`` for minimization, equilibration,
  production, and reporters

SAMMD does not include helper wrappers for OpenMM simulations. Students use the
OpenMM Python API directly after SAMMD writes the export output files. The key
idea is unchanged: SAMMD builds; OpenMM runs.

7. Other engines
----------------

OpenMM is the recommended path for students. Interchange may support GROMACS,
LAMMPS, and Amber later, but they do not have beginner command-line workflows in
this version.

Notebook version
----------------

The related notebook ``notebooks/building_systems_with_sammd.ipynb`` demonstrates
the planning, build, and inspection steps interactively.
