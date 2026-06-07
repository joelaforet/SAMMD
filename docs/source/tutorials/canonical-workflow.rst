Canonical build-to-OpenMM workflow
==================================

This tutorial shows the current student-facing path: SAMMD builds a checked,
deterministic starting plan, and OpenMM will run the molecular dynamics after
future backend artifacts exist. SAMMD builds; OpenMM runs.

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

3. Build system/plan
--------------------

Build the current deterministic system plan:

.. code-block:: bash

   sammd build sammd.yaml --output-dir outputs --overwrite

Today this command writes exactly three build artifacts:

* ``topology.cif`` for topology inspection of the deterministic plan
* ``build_summary.json`` for the machine-readable build report
* ``resolved_config.yaml`` for the exact validated input used for the build

The returned build plan contains the validated configuration, a centered
registered Fcc(111) slab defaulting to Pd(111), internal ``fcc_hollow`` binding
sites, deterministic SAM sulfur anchor placeholders, approximate solution
counts, and resolved output paths. Full SAM molecule coordinates and a
parameterized backend system remain future backend work.

4. Inspect current outputs
--------------------------

Open ``outputs/topology.cif`` in a molecule viewer such as PyMOL to inspect the
configured surface and SAM sulfur anchor placeholders at planned sulfur
positions. Use it to check the surface size, exposed faces, and placement
pattern before moving on.

Use ``outputs/build_summary.json`` to confirm the same build choices in a
machine-readable form. Use ``outputs/resolved_config.yaml`` when you need the
exact validated YAML that produced the plan.

The current release does not write ``positions.cif``, ``interchange.json``,
``system.xml``, or ``anchor_metadata.json``. Those names may appear in resolved
paths or planning metadata, but they are reserved target artifacts, not current
outputs.

5. Reserved future backend artifacts
------------------------------------

SAMMD reserves these future backend construction artifacts so the student path
has stable names when full construction is implemented:

* ``interchange.json`` for the primary portable OpenFF Interchange export
* ``positions.cif`` for fully constructed, human-inspectable/OpenMM-loadable coordinates
* ``system.xml`` for an OpenMM convenience export, not the primary portable artifact
* ``anchor_metadata.json`` for SAM anchor metadata export

The future ``interchange.json`` target is planned as OpenFF Interchange JSON
written with ``Interchange.model_dump_json`` and reloaded with
``Interchange.model_validate_json``. SAMMD does not write it in this lightweight
release, and pre-1.0 Interchange JSON compatibility is not guaranteed across
OpenFF Interchange versions.

6. Future OpenMM handoff
------------------------

After a future SAMMD backend writes ``interchange.json`` and companion
artifacts, students will hand those build artifacts to their own OpenMM Python
API script. The intended teaching path is:

* reload the portable OpenFF Interchange data from ``interchange.json`` with
  ``Interchange.model_validate_json``
* export an OpenMM ``System`` from that Interchange object with
  ``interchange.to_openmm()``
* load positions from ``positions.cif`` for the constructed coordinates
* optionally use ``system.xml`` only as a convenience OpenMM system export
* create and run a raw OpenMM ``Simulation`` for minimization, equilibration,
  production, and reporters

SAMMD does not provide OpenMM simulation wrappers for this handoff. Students use
the OpenMM Python API directly after SAMMD produces the future build artifacts.

That handoff is not runnable in this lightweight release because
``positions.cif``, ``interchange.json``, ``system.xml``, and
``anchor_metadata.json`` are reserved target artifacts, not current outputs. The
important boundary is unchanged: SAMMD builds; OpenMM runs.

7. Other engines
----------------

OpenMM is the student teaching path. GROMACS, LAMMPS, and Amber are future
downstream exports from Interchange, not beginner workflow commands.

Notebook version
----------------

The related notebook ``notebooks/canonical_workflow.ipynb`` demonstrates the
current lightweight build and inspection contract interactively.
