Student start here
==================

This page is the recommended entry point for a new undergraduate user. It is
organized using the Diataxis documentation pattern: start with a tutorial, then
use how-to guidance for specific tasks, explanation pages for background, and
reference pages when you need exact API details.

What SAMMD does today
---------------------

SAMMD is currently a configuration-first package for planning self-assembled
monolayer molecular dynamics systems on metal supports. The stable public API can
load and validate YAML, build deterministic Pd(111)/SAM/solution plans, and write
a slab visualization file for inspection.

The repository also contains an instructor-facing OpenMM workflow prototype for a
small Pd(111)/propanethiol/cinnamaldehyde/ethanol system. That prototype is useful
for seeing where SAMMD is going, but it still uses private backend modules and is
not yet the public student API.

Recommended learning path
-------------------------

1. Install SAMMD from the repository root:

   .. code-block:: bash

      python -m pip install -e ".[dev]"

2. Generate and validate a starting configuration:

   .. code-block:: bash

      sammd init -o sammd.yaml
      sammd validate sammd.yaml

3. Follow the first tutorial:

   :doc:`tutorials/canonical-workflow`

4. Learn what each YAML section means:

   :doc:`tutorials/yaml-configuration`

5. Read the project scope before interpreting results as science:

   :doc:`explanation/project-scope`

Tutorials: learning by doing
----------------------------

Use tutorials when you are new and want a guided path with commands to run.

Start with :doc:`tutorials/canonical-workflow`. It shows the lightweight workflow
that is safe for new users:

* create ``sammd.yaml``
* validate the configuration
* build a deterministic plan from Python
* inspect the planned Pd(111) slab
* write ``planned_slab.cif`` for visualization

Then use :doc:`tutorials/yaml-configuration` to understand the editable parts of
the configuration file.

How-to: common student tasks
----------------------------

Use these task recipes when you know what you want to change.

Change the SAM molecule
~~~~~~~~~~~~~~~~~~~~~~~

Edit the ``sam.components`` section in ``sammd.yaml``. For the current milestone,
SAMMD expects thiol SAM components with exactly one sulfur anchor. Validate after
editing:

.. code-block:: bash

   sammd validate sammd.yaml

Change the solvent or reactant concentration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Edit ``solvent.components`` for solvent mole fractions and ``reactants`` for
millimolar reactant concentrations. SAMMD converts those inputs into approximate
finite-system molecule counts during planning.

After editing, rebuild the plan:

.. code-block:: python

   from sammd import build_system, load_config

   config = load_config("sammd.yaml")
   plan = build_system(config, output_dir="outputs")
   print(plan.solution.molecule_counts)

Inspect the planned slab
~~~~~~~~~~~~~~~~~~~~~~~~

Build a plan and write the slab visualization file:

.. code-block:: python

   path = plan.write_planned_slab_mmcif(overwrite=True)
   print(path)

Open ``planned_slab.cif`` in a molecular viewer such as PyMOL. This file is a
visualization scaffold, not a complete simulation topology.

Run the instructor-facing OpenMM prototype
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If your instructor has set up the optional science environment, the repository
contains a notebook-shaped OpenMM prototype:

.. code-block:: bash

   pixi run -e science python examples/canonical_openmm_workflow.py --overwrite

This is useful for demonstration and smoke validation. Do not treat the output as
production MD or as evidence of scientific convergence.

Explanation: concepts worth understanding
-----------------------------------------

Use explanation pages when you need background rather than commands.

Read :doc:`explanation/project-scope` to understand what SAMMD currently supports
and what is intentionally out of scope for this milestone.

Important current limitations:

* Pd(111) is the supported MVP surface facet.
* The public API produces a build plan, not a complete production simulation.
* The slab visualization file does not include SAM, reactant, or solvent atoms.
* The OpenMM example is still a prototype because it imports private backend
  modules.

Reference: exact API details
----------------------------

Use reference pages when you need exact names, signatures, or model fields.

The main public entry points are documented in :doc:`reference/api`:

* ``sammd.load_config``
* ``sammd.build_system``
* ``sammd.SAMMDConfig``
* orientation analysis helpers in ``sammd.analysis``

What to ask before changing a science input
-------------------------------------------

Before changing a SAM, solvent, reactant, or run condition, write down:

* What experimental system am I trying to represent?
* Which input in ``sammd.yaml`` maps to that experimental condition?
* Is this supported by the current SAMMD milestone?
* Am I looking at a planning artifact, a smoke-test artifact, or a production MD
  result?

If you are unsure, ask your instructor before interpreting the output.
