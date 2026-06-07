Canonical lightweight workflow
==============================

This tutorial mirrors what SAMMD can do today without OpenMM, OpenFF, RDKit, or
mBuild. It is meant to be runnable by undergraduate users and honest about the
current MVP boundary.

Install for development
-----------------------

From the repository root, install SAMMD in editable mode with the lightweight
development dependencies:

.. code-block:: bash

   python -m pip install -e ".[dev]"

Create and validate a configuration
-----------------------------------

Start from the template YAML and validate it before building a plan:

.. code-block:: bash

   sammd init -o sammd.yaml
   sammd validate sammd.yaml

The template follows the defaults summarized in :doc:`yaml-configuration`.

Build the first plan from the command line
------------------------------------------

For a first run, no Python script is needed. This command validates the YAML,
builds the current plan, and writes ``outputs/topology.cif``,
``outputs/build_summary.json``, and ``outputs/resolved_config.yaml``:

.. code-block:: bash

   sammd build sammd.yaml --output-dir outputs --overwrite

Open ``outputs/topology.cif`` in a molecule viewer such as PyMOL to inspect the
configured surface and SAM sulfur anchor placeholders at planned sulfur
positions. This file is the first build artifact to check before moving on to
backend construction and later OpenMM simulation setup.

Build a deterministic plan from Python
--------------------------------------

Use Python when you want to inspect the plan object or write custom analysis
scripts. Save this as ``run_plan.py`` in the same directory as ``sammd.yaml``:

.. code-block:: python

   from sammd import build_system, load_config

   config = load_config("sammd.yaml")
   plan = build_system(config, output_dir="outputs")

   print(plan.slab.metal, plan.slab.facet)
   print(plan.slab.lateral_size_nm)
   print(len(plan.binding_sites))
   print(len(plan.sam_placements.placements))
   print(plan.solution.molecule_counts)
   print(plan.output_paths.topology)

The returned object is a lightweight build plan. It contains the validated
configuration, a commensurate centered registered Fcc(111) slab defaulting to
Pd(111), internal ``fcc_hollow`` binding sites chosen by SAMMD's thiol-on-metal
default, deterministic SAM placement choices, approximate solution counts, and
resolved build output paths.

Write the topology inspection file
----------------------------------

.. code-block:: python

   path = plan.write_topology_cif(overwrite=True)
   print(path)

Run it with:

.. code-block:: bash

   python run_plan.py

This writes the configured ``topology.cif`` path for inspection in tools such as
PyMOL. Use it to check the surface size, exposed faces, and SAM sulfur anchor
placement before starting simulation-specific OpenMM work. Full SAM molecule
coordinates remain future backend work.

Build artifacts
---------------

The current lightweight builder writes these artifacts today:

* ``topology.cif`` for topology inspection of the deterministic plan
* ``build_summary.json`` for the machine-readable build report
* ``resolved_config.yaml`` for the exact validated input

The configuration also reserves names for future backend construction artifacts:

* ``positions.cif`` for fully constructed build-time coordinates
* ``interchange.json`` for OpenFF Interchange export
* ``system.xml`` for an OpenMM system

The YAML intentionally does not define equilibration, production MD,
thermostats, barostats, or trajectory writing. Those OpenMM concepts are taught
and controlled separately from the system-building config.

Future direct OpenMM step
-------------------------

After a future SAMMD backend writes ``positions.cif``, ``interchange.json``, and
``system.xml``, students will use those build artifacts from their own OpenMM
Python API script for minimization, equilibration, production, and reporters.
That direct OpenMM step is not runnable in this lightweight release because the
backend exports are reserved target artifacts, not current outputs.

Notebook version
----------------

The same workflow is available as ``notebooks/canonical_workflow.ipynb`` for
interactive critique and future expansion.
