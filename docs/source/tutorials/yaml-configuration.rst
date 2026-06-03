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
   Selects the supported MVP surface. Today this is ``metal: Pd`` and
   ``facet: "111"``. The nested ``slab`` section controls layer count, requested
   lateral dimensions, centered placement, double-sided geometry, and Pd
   positional restraint metadata. The builder adjusts lateral dimensions to a
   commensurate Pd(111) lattice.

``sam``
   Defines grafting density, anchor model, binding site, and one or more SAM
   components. Current lightweight builds support ``fcc_hollow`` and
   ``hcp_hollow`` anchors. Components may use fractions that sum to 1.0 or
   explicit counts, but not both.

``solvent``
   Defines water model, padding used for approximate count planning, and solvent
   volume fractions. Water defaults to TIP3P-like density metadata. Co-solvents
   need density and molar mass unless a supported built-in value exists.

``salts`` and ``reactants``
   Define molar concentrations. The MVP converts concentrations into molecule or
   ion counts using an approximate planning volume, not a final packed box.

``output``
   Names future ``topology.cif``, ``trajectory.dcd``, and
   ``thermodynamics.csv`` artifacts. The current build plan can write only
   ``planned_slab.cif``.

``reporters``
   Selects thermodynamic reporter fields that will be used when OpenMM reporting
   is implemented.

``simulation``
   Stores timestep, temperature, pressure, nonbonded cutoff, slab cutoff buffer,
   and deterministic seed. The slab must be thicker than cutoff plus buffer.

Resolved defaults to notice
---------------------------

* The slab defaults to 8 centered, double-sided Pd(111) layers
* The requested lateral size defaults to ``[5.0, 5.0]`` nm
* The SAM defaults to propanethiol ``CCCS`` at ``0.25 nm^2 / molecule``
* The solvent defaults to pure water with 3.0 nm padding
* The default reactant is cinnamaldehyde at 0.05 M
* The default seed is 2026 for reproducible placement planning

Current limitations
-------------------

The schema already contains future-facing choices, but the MVP builder rejects
one-sided slabs, off-center slabs, bridge/atop anchors, and full topology writing
with clear errors.
