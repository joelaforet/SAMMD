YAML configuration tutorial
===========================

SAMMD checks the YAML file with Pydantic v2. Unknown keys are rejected so
spelling mistakes fail early.

.. note::

   If you already ran ``pixi shell -e default``, use ``sammd`` commands
   directly. If you are not inside a pixi shell, prefix commands with
   ``pixi run``.

Minimal starting point
----------------------

Generate a complete template with resolved defaults:

.. code-block:: bash

   pixi run sammd init -o sammd-project

This creates ``sammd-project/sammd.yaml``. Edit that file before validating or
building the system.

Important sections
------------------

``surface``
   Selects an Fcc(111) metal surface from the INTERFACE force field. SAMMD
   supports ``Ag``, ``Al``, ``Au``, ``Cu``, ``Ni``, ``Pb``, ``Pd``, and ``Pt``
   with ``facet: "111"`` and defaults to Pd(111). You only set the surface
   size in ``x`` and ``y``. SAMMD chooses the slab thickness from the metal
   geometry and nonbonded cutoff.

``sam``
   Defines grafting density and one or more neutral thiol SAM components.
   Components should include the HS/implicit-H thiol sulfur in the SMILES, not a
   pre-deprotonated thiolate. SAMMD models the metal-S attachment as a stronger
   nonbonded interaction, not as a covalent bond or chemical reaction. You cannot
   change this interaction in this beginner YAML file. For each Fcc(111)
   ``fcc_hollow`` anchor, SAMMD finds the three nearest metal atoms. OpenMM later
   uses that information to strengthen the sulfur-metal Lennard-Jones
   interaction. You do not set the anchor site or sulfur height here.
   Components need a human-readable name, a three-character ``residue_name``, a
   SMILES string, and either fractions that sum to 1.0 or explicit counts.
   Advanced users may set ``extended_length_nm`` to change the estimated fully
   extended SAM length used to size the box. If you do not set it, SAMMD
   estimates the length from the SMILES string and uses at least 0.95 nm.

``solvent``
   Defines ``padding``, the total solvent reservoir thickness in ``z`` across
   both exposed SAM faces. SAMMD splits this value equally, so ``padding: 3.0``
   creates about 1.5 nm of initial solvent above the SAM and 1.5 nm below it.
   Solvent is packed into those explicit reservoir regions, not throughout the
   slab/SAM region, and solvent counts are planned from the combined reservoir
   volume. This can intentionally underpack the initial cell; use a short NPT
   equilibration to let the box shrink or relax. Solvent mole fractions are
   normalized only over solvent components. Each component needs a
   three-character ``residue_name``. Non-water solvents need density and molar
   mass unless SAMMD has a supported built-in value.

``salts`` and ``reactants``
   Define optional ions and reactants. Reactants use exactly one of ``count`` or
   ``concentration``. Reactant concentration is mM. Salt concentration is M, and
   salts define separate cation and anion entries with explicit stoichiometry so
   each ion can have its own residue name.

``packing``
   Defines PACKMOL packing options such as tolerance and maximum loop count.

``parameterization``
   Records the OpenFF small-molecule force field, charge model, INTERFACE metal
   force-field file, and nonbonded cutoff. ``sammd build`` uses these choices to
   create a complete OpenMM-ready system. The INTERFACE metal file gives the base
   slab LJ parameters. SAMMD records sulfur-metal LJ changes in the build
   summary; you do not configure them here.

``outputs``
     Names build output files. The ``sam_grafting_density`` key controls the
     slab-and-sulfur visual check file. The ``solvated_system`` key controls the
     full CIF written by ``sammd build`` with slab, SAMs, reactants, and
     solvent. The ``pymol_system`` key controls the PDB written by ``sammd build``
     with explicit connectivity for PyMOL. It also names files such as
     ``interchange.json``, ``anchor_metadata.json``, ``build_summary.json``, and
     ``resolved_config.yaml``.
     These are not MD trajectory files. ``interchange.json`` stores OpenFF
     Interchange data. Interchange is not yet at version 1.0, so this JSON
     format may change between versions. For this tutorial, use OpenMM.
     This version does not include GROMACS, LAMMPS, Amber, or OpenMM XML exports.

Resolved defaults to notice
---------------------------

* The surface defaults to a ``[2.0, 2.0]`` nm Pd(111) size in ``x`` and ``y``.
* SAMMD chooses the slab thickness automatically.
* The SAM defaults to neutral propanethiol ``CCCS`` at ``0.25 nm^2 / molecule``.
* The solvent defaults to ethanol ``CCO`` with 3.0 nm total padding, split as
  1.5 nm per exposed SAM face.
* The default reactant is one cinnamaldehyde molecule.
* The default seed is 2026, so placement planning is reproducible.

Limitations in this version
---------------------------

This YAML file controls how SAMMD builds the starting system and records
force-field choices. It does not configure OpenMM simulation protocols,
thermostats, barostats, equilibration stages, or trajectory saving.

Beginner glossary
-----------------

``SAM``
   Self-assembled monolayer: molecules attached to a surface in an organized
   layer.

``MD``
   Molecular dynamics: a simulation method that moves atoms over time using a
   force field.

``Fcc(111) slab``
   A flat metal surface model with a face-centered-cubic crystal structure.
   ``111`` names the exposed crystal face; Pd(111) is the default starting
   point.

``grafting density``
   How much surface area is assigned to each attached SAM molecule. Smaller
   values place more molecules on the surface; larger values place fewer
   molecules on the surface.

``SMILES``
   A short text string that describes a molecule, for example ``CCO`` for
   ethanol.

``mole fraction``
   The fraction of one solvent component within the solvent mixture. Solvent
   mole fractions should add to 1.0.

``topology``
   The atoms, bonds, residue names, and, for some files, starting coordinates.
   The ``residue_name`` fields in the YAML control how components appear in
   topology files and molecular viewers.

``trajectory``
   Saved frames from an MD simulation. This YAML file does not configure
   trajectories; students will learn OpenMM simulation control separately.

``sam_grafting_density.cif``
   The first PDBx/mmCIF ``.cif`` structure file to inspect after the default
   ``sammd build``. It is a visual smoke test showing the configured surface and
   planned sulfur anchor positions for the SAM. Use it to check slab geometry,
   three-fold hollow-site placement, and grafting density. Full SAM, solvent,
   and reactant coordinates are created in ``solvated_system.cif`` by
   ``sammd build``.
   Trajectory frames are created later by OpenMM simulation scripts.
