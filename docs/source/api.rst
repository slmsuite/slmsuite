*************
API Reference
*************

This page provides an auto-generated summary of slmsuite's API.
You can find the source directly on [GitHub](https://github.com/QPG-MIT/slmsuite).

Holography
==========
As a core functionality, :mod:`slmsuite` optimizes nearfield phase profiles
(applied to :mod:`~slmsuite.hardware.slms`) to produce desired farfield results
(measured by :mod:`~slmsuite.hardware.cameras`). These methods are provided in:

.. currentmodule:: slmsuite.holography
.. autosummary::
   :toctree: _autosummary
   :template: custom-module-template.rst
   :recursive:

   algorithms

Additionally, quality-of-life support methods are provided,
divided into slm- and camera- centric categories:

.. currentmodule:: slmsuite.holography
.. autosummary::
   :toctree: _autosummary
   :template: custom-module-template.rst
   :recursive:

   toolbox
   analysis

Hardware
========
A core concept of :mod:`slmsuite` is **experimental** holography.
Thus, we require interfaces to control the hardware used in experiment.
While some common hardware implementions are included, we welcome
`contributions <https://github.com/QPG-MIT/slmsuite/blob/main/CONTRIBUTING.md>`_
to expand the scope and utility of the package!
Hardware is divided into two main categories:

.. currentmodule:: slmsuite.hardware
.. autosummary::
   :toctree: _autosummary
   :template: custom-module-template.rst
   :recursive:

   slms
   cameras

The unification of these two categories is stored in:

.. currentmodule:: slmsuite.hardware
.. autosummary::
   :toctree: _autosummary
   :template: custom-module-template.rst
   :recursive:

   cameraslms

Miscellaneous
=============
Additional functions to handle minutiae.

.. currentmodule:: slmsuite.misc
.. autosummary::
   :toctree: _autosummary
   :template: custom-module-template.rst
   :recursive:

   files
   fitfunctions
