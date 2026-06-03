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

Build a deterministic plan from Python
--------------------------------------

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
configuration, a commensurate centered Pd(111) slab, fcc or hcp hollow binding
sites, deterministic SAM placement choices, approximate solution counts, and
resolved output paths.

Write the planned slab visualization file
-----------------------------------------

.. code-block:: python

   path = plan.write_planned_slab_mmcif(overwrite=True)
   print(path)

This writes ``planned_slab.cif`` next to the future topology path. The file is a
slab-only visualization scaffold for inspection in tools such as PyMOL. It is
not a complete topology and does not include SAM, solvent, salt, or reactant
molecules.

Future backend artifacts
------------------------

The configuration already resolves names for future production artifacts:

* ``topology.cif`` for a full system topology and starting coordinates
* ``trajectory.dcd`` for OpenMM trajectory frames
* ``thermodynamics.csv`` for OpenMM reporter output

Those files are not generated yet because full OpenFF/OpenMM construction is
deferred beyond this milestone.

Notebook version
----------------

The same workflow is available as ``notebooks/canonical_workflow.ipynb`` for
interactive critique and future expansion.
