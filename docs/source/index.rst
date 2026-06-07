SAMMD documentation
===================

SAMMD is a configuration-first package for building self-assembled monolayer MD
systems. The YAML file describes the surface, SAM chemistry, reactants, solvent,
salts, packing, parameterization, and build outputs. OpenMM simulation protocols
are intentionally taught and controlled separately.

If you are new to Python or molecular dynamics, start with
:doc:`tutorials/canonical-workflow`. The command-line path gets you to a first
``topology.cif`` file before you need to write a Python script.

The current teaching workflow starts with validation and deterministic build
planning, then progresses toward OpenFF/OpenMM-backed system construction and
student-written OpenMM simulation scripts.

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
