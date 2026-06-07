First-release build contract
============================

SAMMD's first release is a configuration-first builder/exporter contract. A user
edits a YAML file, validates it, and runs a CLI build that writes deterministic
inspection artifacts. SAMMD does not own equilibration or production simulation
wrappers; downstream OpenMM simulation scripts are taught separately and should
use the OpenMM Python API directly.

CLI contract
------------

The supported first-release command-line surface is:

.. list-table::
   :header-rows: 1

   * - Command
     - Contract
   * - ``sammd init [-o PATH] [--force]``
     - Write a commented starter YAML configuration. Existing files are protected
       unless ``--force`` is supplied.
   * - ``sammd validate CONFIG``
     - Load and validate a YAML configuration without writing build artifacts.
   * - ``sammd build CONFIG --output-dir DIR --overwrite``
     - Build the current deterministic plan and write the currently implemented
       artifacts into ``DIR``. Existing artifacts are protected unless
       ``--overwrite`` is supplied.

The ``build`` command does not run minimization, equilibration, production MD,
trajectory writing, or reporter setup.

Python API contract
-------------------

The supported first-release Python surface is:

.. list-table::
   :header-rows: 1

   * - Name
     - Contract
   * - ``load_config(path)``
     - Load and validate a YAML file into ``SAMMDConfig``.
   * - ``load_config_dict(data)``
     - Validate an already parsed mapping into ``SAMMDConfig``.
   * - ``build_system(config, output_dir=None, seed=None)``
     - Return a lightweight ``SAMMDBuildPlan`` from a ``SAMMDConfig``, YAML path,
       or parsed mapping.

The object returned by ``build_system`` is documented as ``SAMMDBuildPlan``. It
exposes deterministic slab, SAM placement, solution composition, output paths,
``build_summary()``, and artifact writers for the current plan, but it is not a
top-level public import in ``sammd.__all__``. ``SAMMDBuildPlan`` is not an OpenMM
``System``, an OpenFF ``Interchange``, or a simulation wrapper. Code that needs
full backend construction should treat ``full_construction_available`` as false
until the OpenFF/OpenMM construction backend lands.

The ``build_summary()`` SAM section records the first-release metal-S interaction
strategy as dependency-free metadata. The canonical mode is
``nonbonded_lj_override``: for each neutral thiol anchor, the backend should use
the three nearest registered Fcc(111) hollow-site metal atoms as selected pairs
for a post-export OpenMM pair-specific LJ override with ``sigma = 0.22 nm`` and
``epsilon = 2.0 kcal/mol``. This is a strengthened nonbonded attraction layered
on top of the base INTERFACE metal LJ model, not covalent, quantum, or reactive
chemisorption. The first release records the strategy and selected pair indices
but does not implement full OpenFF/Interchange construction or OpenMM export.

Validation gates
----------------

The internal ``sammd.validation`` module provides dependency-free gates for the
current lightweight build plan and topology CIF text. These gates check surface
atom metadata lengths, non-empty top and bottom binding-site labels, SAM counts,
solution-volume/box-volume agreement, finite positive box dimensions/bounds and
volume consistency, slab/box lateral-size agreement, SAM anchor metadata,
metal-S pair counts and slab-local indices, canonical metal-S strategy metadata,
current/reserved output suffixes, and lightweight topology CIF atom counts and
cell lengths.

These gates intentionally do not require OpenMM, OpenFF, or full backend
artifacts. Missing reserved target artifacts such as ``positions.cif``,
``interchange.json``, ``system.xml``, and ``anchor_metadata.json`` are not
failures in the current release.

Future backend validation gates should cover full constructed atom counts,
topology/positions/system agreement, no severe overlaps, charge and parameter
assignment completeness, applied metal-S overrides, export reloadability, finite
minimized coordinates, and lowered minimization energy.

Artifact contract
-----------------

The first-release output names are reserved so user scripts and documentation can
refer to stable paths while backend work proceeds. Future backend exports should
treat ``interchange.json`` as the primary portable system artifact.

.. list-table::
   :header-rows: 1

   * - Artifact
     - Status
     - Contract
   * - ``topology.cif``
     - Current
     - Lightweight topology-inspection CIF for the deterministic plan, including
       SAM sulfur anchor placeholders at planned sulfur positions. This is a
       human-inspectable/OpenMM-loadable structure file, not a parameterized
       backend system.
   * - ``build_summary.json``
     - Current
     - Machine-readable summary of the validated plan, output paths, and
       backend-ready metal-S LJ override metadata.
   * - ``resolved_config.yaml``
     - Current
     - Validated YAML configuration used for the build.
   * - ``positions.cif``
     - Target
     - Reserved for fully constructed coordinates from the future backend. This
       is a human-inspectable/OpenMM-loadable structure file paired with the
       backend system artifact.
   * - ``interchange.json``
     - Target
     - Reserved for the future primary portable OpenFF Interchange export. The
       planned JSON path is ``Interchange.model_dump_json`` for saving and
       ``Interchange.model_validate_json`` for reload. SAMMD does not write this
       artifact in the current release, records no concrete
       ``openff-interchange`` package version until a real artifact is written,
       and treats pre-1.0 Interchange JSON compatibility as not guaranteed
       across versions.
   * - ``system.xml``
     - Target
     - Reserved for a future OpenMM convenience export derived from the backend
       system, not the primary portable SAMMD artifact.
   * - ``anchor_metadata.json``
     - Target
     - Reserved for a future SAM anchor metadata export.

Current limitation
------------------

Today ``sammd build`` writes only ``topology.cif``, ``build_summary.json``, and
``resolved_config.yaml``. It does not write ``positions.cif``,
``interchange.json``, ``system.xml``, or ``anchor_metadata.json`` in the current
lightweight release. Full OpenFF/OpenMM construction and OpenMM-ready exports
remain future backend work, including full SAM molecule coordinates. Public
SAMMD APIs should not add equilibration or production simulation helpers as part
of this contract.
