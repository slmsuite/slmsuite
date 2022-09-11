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

 - :class:`.AlliedVision`, for AlliedVision cameras, through :mod:`vimba` [1]_,
 - :class:`.Cheetah640`, for Xenics Cheetah cameras, through the Xenics SDK [2]_,
 - :class:`.FLIR`, for Teledyne FLIR cameras, through the :mod:`PySpin`
   interface of the Spinnaker SDK [3]_,
 - :class:`.MMCore`, for the general microscope control package Micro-Manager,
   through :mod:`pymmcore` [4]_, and
 - :class:`.ThorCam`, for Thorlabs scientific cameras, through :mod:`thorlabs_tsi_sdk` [5]_.

Tip
~~~~~~~~
While the superclass :class:`.Camera` only requires a small number of features to
be implemented as class functions, any further control of a camera interface can be
accessed by using the given SDK object directly (usually the attribute :attr:`cam` of
the subclass) or writing additional functions into the subclass.

References
----------
.. [1] https://github.com/alliedvision/VimbaPython
.. [2] https://www.xenics.com/software/
.. [3] https://www.flir.com/products/spinnaker-sdk/
.. [4] https://github.com/micro-manager/pymmcore
.. [5] https://www.thorlabs.com/software_pages/ViewSoftwarePage.cfm?Code=ThorCam
"""
from .camera import Camera
from .allied_vision import AlliedVision
from .xenics import Cheetah640
from .flir import FLIR
from .mm_core import MMCore
from .thorlabs import ThorCam
