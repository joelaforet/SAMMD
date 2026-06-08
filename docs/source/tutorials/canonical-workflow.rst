Recommended build-to-OpenMM workflow
====================================

This tutorial shows the recommended path for students. SAMMD checks the YAML,
builds the same starting structure each time, and, if requested, writes files
you can load in OpenMM. SAMMD builds; OpenMM runs.

1. Create config
----------------

From the repository root, create a starter YAML file:

.. code-block:: bash

   sammd init -o sammd.yaml

The template follows the defaults summarized in :doc:`yaml-configuration`. It
describes the surface, SAM, solution composition, and output names used for the
build step. It does not describe minimization, equilibration, production MD,
thermostats, barostats, trajectories, or reporters.

2. Validate config
------------------

Validate the YAML before building anything:

.. code-block:: bash

   sammd validate sammd.yaml

Validation checks that SAMMD can interpret the file and reports configuration
errors before any output files are written.

3. Build the starting model
---------------------------

Build the starting model:

.. code-block:: bash

   sammd build sammd.yaml --output-dir outputs --overwrite

In this version, this command writes exactly three output files:

* ``topology.cif`` so you can inspect the planned topology
* ``build_summary.json`` for a JSON build report that scripts can read
* ``resolved_config.yaml`` for the exact validated input used for the build

The result includes the validated configuration and resolved output paths. It
also includes a centered registered Fcc(111) slab, using Pd(111) by default,
internal ``fcc_hollow`` binding sites, placeholder sulfur anchors for the SAM,
and approximate solution counts. SAMMD writes full SAM molecule coordinates and
a parameterized backend system only when you run from a CUDA-labeled pixi
environment with ``--export-backend``.

4. Inspect outputs
------------------

Open ``outputs/topology.cif`` in a molecule viewer such as PyMOL to inspect the
configured surface and the placeholder sulfur anchors. Use it to check the
surface size, exposed faces, and where the SAM anchors will go before moving on.

Use ``outputs/build_summary.json`` to confirm the same build choices in a
machine-readable form. Use ``outputs/resolved_config.yaml`` when you need the
exact validated YAML that produced the plan.

You may see those filenames in resolved paths or metadata, but the default
lightweight command does not write ``positions.cif``, ``interchange.json``,
``system.xml``, or ``anchor_metadata.json``.

5. Optional backend output files
--------------------------------

Use a CUDA-labeled pixi environment when you want SAMMD to write files for
OpenFF/OpenMM. Run ``nvidia-smi`` on the machine first, then choose an
environment whose CUDA version is not newer than the CUDA version shown there.
For example, use ``cuda-12-4`` for CU Boulder Blanca older-GPU nodes and
``cuda-12-6`` for PSC Bridges2.

.. code-block:: bash

   pixi run -e cuda-12-6 sammd build sammd.yaml --output-dir outputs --overwrite --export-backend

That command writes these additional files:

* ``interchange.json`` for the primary OpenFF Interchange export
* ``positions.cif`` for coordinates you can inspect and load in OpenMM
* ``system.xml`` for an OpenMM file, not the primary OpenFF Interchange output
* ``anchor_metadata.json`` for SAM anchor metadata

The ``interchange.json`` file is OpenFF Interchange JSON written with
``Interchange.model_dump_json`` and reloaded with
``Interchange.model_validate_json``. Pre-1.0 Interchange JSON compatibility is
not guaranteed across OpenFF Interchange versions. Configs that include salt are
rejected until backend export supports salt.

6. Use these files with OpenMM
------------------------------

After backend export writes ``interchange.json`` and the other output files,
students use them in their own OpenMM Python API script. Follow these steps:

* reload the OpenFF Interchange data from ``interchange.json`` with
  ``Interchange.model_validate_json``
* export an OpenMM ``System`` from that Interchange object with
  ``interchange.to_openmm()``
* load positions from ``positions.cif`` for the constructed coordinates
* optionally use ``system.xml`` only as an OpenMM file
* create and run a raw OpenMM ``Simulation`` for minimization, equilibration,
  production, and reporters

SAMMD does not include helper wrappers for OpenMM simulations. Students use the
OpenMM Python API directly after SAMMD writes the backend output files. The key
idea is unchanged: SAMMD builds; OpenMM runs.

7. Other engines
----------------

OpenMM is the recommended path for students. Interchange may support GROMACS,
LAMMPS, and Amber later, but they do not have beginner command-line workflows in
this version.

Notebook version
----------------

The related notebook ``notebooks/building_systems_with_sammd.ipynb`` demonstrates the
default lightweight build and inspection steps interactively.
