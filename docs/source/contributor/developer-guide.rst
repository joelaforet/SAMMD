Contributor and developer guide
===============================

Current package map
-------------------

``sammd.core.config``
   Pydantic v2 models, YAML loading, CLI template text, and validation rules.

``sammd.cli``
   ``sammd init``, ``sammd validate``, and ``sammd build`` commands.

``sammd.model.surfaces``
   Registered Fcc(111) lattice metadata, commensurate slab planning, and hollow
   binding-site generation.

``sammd.model.sam``
   Deterministic SAM site selection and component assignment.

``sammd.model.solvation``
   Approximate solvent, salt, and reactant molecule-count planning.

``sammd.core.io`` and ``sammd.runtime.reporting``
   Output path planning, topology CIF writing helpers, and future reporter field
   metadata.

``sammd.backends.forcefields``
   Dependency-free INTERFACE metal parameter metadata and OFFXML export helpers.

``sammd.analysis``
   Dependency-free orientation geometry primitives for future trajectory
   analysis.

Development expectations
------------------------

Keep the MVP importable without heavy scientific dependencies. Imports of
OpenMM, OpenFF, MDAnalysis, ParmEd, mBuild, PDBFixer, or related export tools
should stay lazy and out of module top level.

OpenFF/OpenMM system construction should stay explicit and testable. New work
should preserve lazy imports while treating OpenFF as part of the normal
build/export path in SAMMD pixi environments.
