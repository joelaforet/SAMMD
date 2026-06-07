Scientific assumptions
======================

SAMMD currently builds a lightweight, deterministic plan for self-assembled
monolayer systems. These assumptions describe what that plan means physically
and what remains outside SAMMD's ownership in this release.

Surface and SAM placement
-------------------------

SAMMD uses registered Fcc(111) metal surfaces, defaulting to Pd(111). The slab is
centered at the origin, with SAMs placed normal to the surface: +z for the top
face and -z for the bottom face.

The hollow-site placement strategy is an internal ``fcc_hollow`` default. SAMMD
selects sulfur pairs to the three nearest hollow-site metal atoms for each placed
thiol anchor. Beginner YAML input does not expose adsorption-site selection in
this release.

Beginner SAM inputs should be neutral thiols with an HS/implicit-H thiol sulfur,
not pre-deprotonated thiolates. SAMMD uses the sulfur atom for placement while
keeping the beginner chemistry description neutral.

Box planning
------------

Padding is measured from the fully extended SAM tips to the box boundary. This
keeps the requested spacing tied to the longest planned SAM conformation rather
than only to the metal slab.

Force-field assumptions
-----------------------

Base metal Lennard-Jones parameters come from the INTERFACE Fcc metal data. The
target route for organic molecules is OpenFF. The current lightweight code
records and validates force-field choices, but it does not yet export a full
OpenFF/OpenMM backend system.

The selected metal-S Lennard-Jones override is an internal, post-export proxy for
a tunable strengthened nonbonded interaction. It is not covalent, quantum, or
reactive chemisorption, and it is not currently a beginner YAML knob.

Simulation boundary
-------------------

SAMMD does not own minimization, equilibration, production, trajectory writing,
or reporter setup. Those OpenMM simulation protocol choices are handled in
separate scripts or future teaching material after system construction artifacts
exist.
