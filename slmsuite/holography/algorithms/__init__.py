"""
GPU-accelerated holography algorithms.

This module is currently focused on
`Gerchberg-Saxton (GS) <http://www.u.arizona.edu/~ppoon/GerchbergandSaxton1972.pdf>`_
iterative Fourier transform phase retrieval algorithms
via the :class:`~slmsuite.holography.algorithms.Hologram` class;
however, support for complex holography and other algorithms
(e.g. `gradient descent algorithms <https://doi.org/10.1364/AO.21.002758>`_)
is also planned. Additionally, so-called Weighted Gerchberg-Saxton (WGS) algorithms for hologram
generation with or without closed-loop camera feedback are supported, especially for
the `generation of optical focus arrays <https://doi.org/10.1364/OL.44.003178>`_,
a subset of general image formation. We also support `Mixed Region Amplitude Freedom (MRAF)
<https://doi.org/10.1007/s10043-018-0456-x>`_ feedback.

Tip
~~~
This module makes use of the GPU-accelerated computing library :mod:`cupy`
(`GitHub <https://docs.cupy.dev/en/stable/reference/index.html>`_)
If :mod:`cupy` is not supported, then :mod:`numpy` is used as a fallback, though
CPU alone is significantly slower. Using :mod:`cupy` is highly encouraged.

Important
---------
:mod:`slmsuite` follows the ``shape = (h, w)`` and ``vector = (x, y)`` formalism adopted by
the :mod:`numpy` ecosystem. :mod:`numpy`, :mod:`scipy`, :mod:`matplotlib`, etc generally follow this
formalism. The ``shape`` and indexing of an array or image always uses the inverted ``(h, w)`` form,
but other functions such as ``numpy.meshgrid(x, y)`` (default), ``scipy.odr.Data(x, y)``, or
``matplotlib.pyplot.scatter(x, y)`` use the standard cartesian ``(x, y)`` form that is more familiar
to users. This is not ideal and causes confusion, but this is the formalism generally
adopted by the community.

Important
~~~~~~~~~
This package uses a number of bases or coordinate spaces. Some coordinate spaces are
directly used by the user (most often the camera basis ``"ij"`` used for feedback).
Other bases are less often used directly, but are important to how holograms are
optimized under the hood (esp. ``"knm"``, the coordinate space of optimization).

.. list-table:: Bases used in :mod:`slmsuite`.
   :widths: 20 80
   :header-rows: 1

   * - Basis
     - Meaning
   * - ``"ij"``
     - Pixel basis of the camera. Centered at ``(i, j) = (cam.shape[1],
       cam.shape[0])/2``. Is in the image space of the camera.
   * - ``"kxy"``
     - Normalized (floating point) basis of the SLM's :math:`k`-space in normalized units.
       Centered at ``(kx, ky) = (0, 0)``. This basis is what the SLM projects in angular
       space (which maps to the camera's image space via the Fourier transform implemented by free
       space and solidified by a lens).
   * - ``"knm"``
     - Pixel basis of the SLM's computational :math:`k`-space.  Centered at ``(kn, km) =
       (0, 0)``. ``"knm"`` is a discrete version of the continuous ``"kxy"``. This is
       important because holograms need to be stored in computer memory, a discrete
       medium with pixels, rather than being purely continuous. For instance, in
       :class:`SpotHologram`, spots targeting specific continuous angles are rounded to
       the nearest discrete pixels of ``"knm"`` space in practice.
       Then, this ``"knm"`` space image is handled as a
       standard image/array, and operations such as the discrete Fourier transform
       (instrumental for numerical hologram optimization) can be applied.

See the first tip in :class:`Hologram` to learn more about ``"kxy"`` and ``"knm"``
space.
"""
from slmsuite.holography.algorithms._header import *

from slmsuite.holography.algorithms._hologram import Hologram as _Hologram
from slmsuite.holography.algorithms._feedback import FeedbackHologram as _FeedbackHologram
from slmsuite.holography.algorithms._spots import SpotHologram as _SpotHologram
from slmsuite.holography.algorithms._spots import FreeSpotHologram as _FreeSpotHologram

# Hack to get automodule to put the classes in the correct location.
class Hologram(_Hologram):
    pass

class FeedbackHologram(_FeedbackHologram):
    pass

class SpotHologram(_SpotHologram):
    pass

class FreeSpotHologram(_FreeSpotHologram):
    pass