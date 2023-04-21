# The first line of the docstring might read awkward alone,
# but it's meant to parallel the slms module description.
"""
The sensor arrays used to measure results.
Computer vision hardware is connected to python by a myriad of SDKs, often provided by hardware
vendors. However, these SDKs likewise have a myriad of function names and hardware-specific
quirks. Thus, cameras in :mod:`slmsuite` are integrated as subclasses of
the abstract class :class:`.Camera`, which requires subclasses to implement a number of methods
relevant for SLM feedback (see below).
These subclasses are effectively wrappers for the given SDK, but also include
quality-of-life features such as image transformations (flips, rotates) and useful common methods.
A number of SDKs are supported, including:

 - :class:`.AlliedVision`, for AlliedVision cameras, through :mod:`vimba`
   (`GitHub <https://github.com/alliedvision/VimbaPython>`_),
 - :class:`.Cheetah640`, for Xenics Cheetah cameras, through the
   `Xenics SDK <https://www.xenics.com/software/>`_,
 - :class:`.FLIR`, for Teledyne FLIR cameras, through the :mod:`PySpin`
   interface of the `Spinnaker SDK <https://www.flir.com/products/spinnaker-sdk/>`_,
 - :class:`.MMCore`, for the general microscope control package Micro-Manager,
   through :mod:`pymmcore` (`GitHub <https://github.com/micro-manager/pymmcore>`_), and
 - :class:`.ThorCam`, for Thorlabs scientific cameras, through :mod:`thorlabs_tsi_sdk`
   and the `ThorCam SDK <https://www.thorlabs.com/software_pages/ViewSoftwarePage.cfm?Code=ThorCam>`_.

Tip
~~~~~~~~
While the superclass :class:`.Camera` only requires a small number of features to
be implemented as class functions, any further control of a camera interface can be
accessed by using the given SDK object directly (usually the attribute :attr:`cam` of
the subclass) or writing additional functions into the subclass.
"""
