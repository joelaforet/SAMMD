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

Artifact contract
-----------------

The first-release output names are reserved so user scripts and documentation can
refer to stable paths while backend work proceeds.

.. list-table::
   :header-rows: 1

   * - Artifact
     - Status
     - Contract
   * - ``topology.cif``
     - Current
     - Lightweight topology-inspection CIF for the deterministic plan.
   * - ``build_summary.json``
     - Current
     - Machine-readable summary of the validated plan and output paths.
   * - ``resolved_config.yaml``
     - Current
     - Validated YAML configuration used for the build.
   * - ``positions.cif``
     - Target
     - Reserved for fully constructed positions from the future backend.
   * - ``interchange.json``
     - Target
     - Reserved for future OpenFF Interchange export.
   * - ``system.xml``
     - Target
     - Reserved for future OpenMM System XML export.

Current limitation
------------------

Today ``sammd build`` writes only ``topology.cif``, ``build_summary.json``, and
``resolved_config.yaml``. Full OpenFF/OpenMM construction and OpenMM-ready exports
remain future backend work. Public SAMMD APIs should not add equilibration or
production simulation helpers as part of this contract.
