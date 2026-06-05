SAMMD documentation
===================

SAMMD is a configuration-first prototype for planning self-assembled monolayer
MD systems. The current milestone validates YAML inputs, creates deterministic
Pd(111)/SAM/solution build plans, writes a slab-only ``planned_slab.cif`` file,
and exposes lightweight orientation analysis helpers.

Public production OpenFF/OpenMM construction is intentionally deferred. The
repository includes a prototype OpenMM notebook for short smoke-test runs, but it
still uses private backend modules and should not be interpreted as production
MD.

New students should begin with :doc:`student-start-here`.

.. toctree::
   :maxdepth: 2
   :caption: Start here

   student-start-here

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
