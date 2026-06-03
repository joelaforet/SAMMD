SAMMD documentation
===================

SAMMD is a configuration-first prototype for planning self-assembled monolayer
MD systems. The current milestone validates YAML inputs, creates deterministic
Pd(111)/SAM/solution build plans, writes a slab-only ``planned_slab.cif`` file,
and exposes lightweight orientation analysis helpers.

Full OpenFF/OpenMM construction is intentionally deferred. Future milestones are
expected to produce full ``topology.cif``, ``trajectory.dcd``, and
``thermodynamics.csv`` artifacts from the same configuration style.

.. toctree::
   :maxdepth: 2
   :caption: Tutorials

   tutorials/canonical-workflow
   tutorials/yaml-configuration

.. toctree::
   :maxdepth: 2
   :caption: Reference

   reference/api

.. toctree::
   :maxdepth: 2
   :caption: Explanation

   explanation/project-scope

.. toctree::
   :maxdepth: 2
   :caption: Contributor guide

   contributor/developer-guide
