YAML configuration tutorial
===========================

SAMMD uses a strict Pydantic v2 schema loaded from YAML. Unknown keys are
rejected so that spelling mistakes fail early.

Minimal starting point
----------------------

Generate a complete template with resolved defaults:

.. code-block:: bash

   sammd init -o sammd.yaml

Important sections
------------------

``surface``
   Selects a registered Fcc INTERFACE surface. Today the registry supports
   ``metal: Pd`` and ``facet: "111"``. Users provide only the lateral ``x`` and
   ``y`` size. SAMMD chooses the slab thickness automatically from the metal
   geometry and nonbonded cutoff.

``sam``
   Defines grafting density and one or more neutral thiol SAM components.
   Components should include the HS/implicit-H thiol sulfur in the SMILES, not a
   pre-deprotonated thiolate. Metal-S attachment is represented/planned
   internally as a strengthened nonbonded interaction, not as covalent, quantum,
   or reactive chemistry, and it is not yet a student-facing YAML knob.
   Components need a human-readable name, a three-character ``residue_name``, a
   SMILES string, and either fractions that sum to 1.0 or explicit counts.

``solvent``
   Defines a requested z-padding/count-planning value for solvent above the slab
   and solvent mole fractions normalized over solvent components only. Final box
   construction details are owned by later build stages. Each component needs a
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
   force-field resource, and nonbonded cutoff selected for future backend export.
   The current lightweight builder validates and records these choices without
   constructing a parameterized backend system.

``outputs``
   Names current build artifacts such as ``topology.cif``,
   ``build_summary.json``, and ``resolved_config.yaml``. It also reserves future
   backend artifact names such as ``positions.cif``, ``interchange.json``, and
   ``system.xml``. These are not MD trajectory outputs.

Resolved defaults to notice
---------------------------

* The surface defaults to a ``[2.0, 2.0]`` nm Pd(111) lateral size
* Slab thickness is hidden and chosen automatically
* The SAM defaults to neutral propanethiol ``CCCS`` at ``0.25 nm^2 / molecule``
* The solvent defaults to ethanol ``CCO`` with a requested 3.0 nm z-padding/count-planning value
* The default reactant is one cinnamaldehyde molecule
* The default seed is 2026 for reproducible placement planning

Current limitations
-------------------

This config defines system construction choices and records parameterization
selections for future backend export. OpenMM simulation protocols, thermostats,
barostats, equilibration stages, and trajectory saving are intentionally kept
out of this release's YAML file.

Beginner glossary
-----------------

``SAM``
   Self-assembled monolayer: molecules attached to a surface in an organized
   layer.

``MD``
   Molecular dynamics: a simulation method that moves atoms over time using a
   force field.

``Pd(111) slab``
   A flat palladium surface model. ``111`` names the crystal face being exposed.

``grafting density``
   How much surface area is assigned to each attached SAM molecule. Smaller
   values place more molecules on the surface.

``SMILES``
   A short text string that describes a molecule, for example ``CCO`` for
   ethanol.

``mole fraction``
   The fraction of one solvent component within the solvent mixture. Solvent
   mole fractions should add to 1.0.

``topology``
   The atoms, bonds, residue names, and starting coordinates for a full system.
   The ``residue_name`` fields in the YAML control how components appear in
   topology files and molecular viewers.

``trajectory``
   Saved frames from an MD simulation. This YAML file does not configure
   trajectories; students will learn OpenMM simulation control separately.

``topology.cif``
   The first structure file to inspect after ``sammd build``. It shows the
   configured surface and SAM anchor placements; trajectory frames are produced
   later by OpenMM simulation scripts.
