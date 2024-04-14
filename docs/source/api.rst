*************
API Reference
*************

This page provides an auto-generated summary of |slmsuite|_'s API. You can
find the source on `GitHub <https://github.com/QPG-MIT/slmsuite>`_.

|slmsuite|_ is divided into three main modules:
algorithms and analysis in :mod:`holography`,
connectivity to physical devices via :mod:`hardware`, and
other useful functions in :mod:`misc`.

Holography
==========

As a core functionality, |slmsuite|_ **optimizes** nearfield phase profiles
(applied to :mod:`~slmsuite.hardware.slms`) to produce desired farfield results
(measured by :mod:`~slmsuite.hardware.cameras`). These methods are provided in:

.. currentmodule:: slmsuite.holography
.. autosummary::
   :toctree: _autosummary
   :template: custom-module-template.rst
   :recursive:

   algorithms

Additionally, quality-of-life support methods are provided,
divided into SLM- and camera- centric categories:

.. currentmodule:: slmsuite.holography
.. autosummary::
   :toctree: _autosummary
   :template: custom-module-template.rst
   :recursive:

   toolbox
   analysis

Hardware
========

A central concept of |slmsuite|_ is **experimental** holography.
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

.. |slmsuite| replace:: :mod:`slmsuite`
.. _slmsuite: https://github.com/QPG-MIT/slmsuite