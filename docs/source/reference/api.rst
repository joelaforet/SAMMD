API reference
=============

Core workflow
-------------

.. autofunction:: sammd.load_config

.. autofunction:: sammd.load_config_dict

.. autofunction:: sammd.build_system

.. autoclass:: sammd.builders.SAMMDBuildPlan
   :members:

.. autoclass:: sammd.builders.CompositionPlanningBox
   :members:

Configuration model
-------------------

.. autoclass:: sammd.SAMMDConfig
   :members:

Orientation analysis helper
---------------------------

This lightweight analysis helper is available for tutorial inspection workflows.
It is not part of the system build/export contract.

.. autofunction:: sammd.analysis.analyze_orientation

.. autoclass:: sammd.analysis.OrientationResult
   :members:
