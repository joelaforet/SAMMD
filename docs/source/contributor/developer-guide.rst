Contributor and developer guide
===============================

Current package map
-------------------

``sammd.config``
   Pydantic v2 models, YAML loading, CLI template text, and validation rules.

``sammd.cli``
   Lightweight ``sammd init``, ``sammd validate``, and ``sammd build`` commands.

``sammd.surfaces``
   Registered Fcc(111) lattice metadata, commensurate slab planning, and hollow
   binding-site generation.

``sammd.sam``
   Deterministic SAM site selection and component assignment.

``sammd.solvation``
   Approximate solvent, salt, and reactant molecule-count planning.

``sammd.io`` and ``sammd.reporting``
   Output path planning, topology CIF writing helpers, and future reporter field
   metadata.

``sammd.forcefields``
   Lightweight INTERFACE metal parameter metadata and OFFXML export helpers.

``sammd.analysis``
   Dependency-free orientation geometry primitives for future trajectory
   analysis.

Development expectations
------------------------

Keep the MVP importable without heavy scientific dependencies. Imports of
OpenMM, OpenFF, MDAnalysis, ParmEd, mBuild, PDBFixer, or related backend tools
should stay lazy and out of module top level.

Full OpenFF/OpenMM system construction is deferred. New work should preserve the
current lightweight path while making future backend integration explicit and
testable.
