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

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import cv2
from tqdm import tqdm
import warnings
import pprint

# Import numpy and scipy dependencies.
import numpy as np
import scipy.fft as spfft
from scipy.ndimage import gaussian_filter1d as sp_gaussian_filter1d
from scipy.ndimage import affine_transform as sp_affine_transform
from scipy.ndimage import gaussian_filter as sp_gaussian_filter

# Try to import cupy, but revert to base numpy/scipy upon ImportError.
try:
    import cupy as cp
    import cupyx.scipy.fft as cpfft
    from cupyx.scipy.ndimage import gaussian_filter1d as cp_gaussian_filter1d
    from cupyx.scipy.ndimage import gaussian_filter as cp_gaussian_filter
    from cupyx.scipy.ndimage import affine_transform as cp_affine_transform
except ImportError:
    cp = np
    cpfft = spfft
    cp_gaussian_filter1d = sp_gaussian_filter1d
    cp_gaussian_filter = sp_gaussian_filter
    cp_affine_transform = sp_affine_transform
    print("cupy not installed. Using numpy.")

# Import helper functions
from slmsuite.holography import analysis, toolbox
from slmsuite.misc.math import REAL_TYPES
from slmsuite.misc.files import write_h5, read_h5
from slmsuite.misc.fitfunctions import gaussian2d

# List of algorithms and default parameters
# See algorithm documentation for parameter definitions.
# Tip: In general, decreasing the feedback exponent (from 1) improves
#      stability at the cost of slower convergence. The default (0.8)
#      is an empirically derived value for a reasonable tradeoff.
ALGORITHM_DEFAULTS = {
    "GS" :            {"feedback" : "computational"},    # No feedback for bare GS, but initializes var.
    "WGS-Leonardo" :  {"feedback" : "computational",
                        "feedback_exponent" : 0.8},
    "WGS-Kim" :       {"feedback" : "computational",
                        "fix_phase_efficiency" : None,
                        "fix_phase_iteration" : 10,
                        "feedback_exponent" : 0.8},
    "WGS-Nogrette" :  {"feedback" : "computational",
                        "factor":0.1}
}

# List of feedback options. See the documentation for the feedback keyword in optimize().
FEEDBACK_OPTIONS = [
    "computational",
    "computational_spot",
    "experimental",
    "experimental_spot",
    "external_spot"
]

class Hologram:
    r"""
    Phase retrieval methods applied to holography.
    See :meth:`.optimize()` to learn about the methods implemented for hologram optimization.

    Tip
    ~~~
    The Fourier domain (``"kxy"``) of an SLM with shape :attr:`slm_shape` also has the shape
    :attr:`slm_shape` under discrete Fourier transform. However, the extents of this
    domain correspond to the edges of the farfield
    (:math:`\pm\frac{\lambda}{2\Delta x}` radians, where :math:`\Delta x`
    is the SLM pixel pitch). This means that resolution of the farfield
    :math:`\pm\frac{\lambda}{2N_x\Delta x}` can be quite poor with small :math:`N_x`.
    The solution is to zero-pad the SLM nearfield
    --- artificially increasing the width :math:`N_x` and height
    :math:`N_y` even though the extent of the non-zero nearfield data remains the same ---
    and thus enhance the resolution of the farfield.
    In practice, padding is accomplished by passing a :attr:`shape` or
    :attr:`target` of appropriate shape (see constructor :meth:`.__init__()` and subclasses),
    potentially with the aid of the static helper function :meth:`.calculate_padded_shape()`.

    Note
    ~~~~
    :attr:`target`, :attr:`weights`, :attr:`phase_ff`, and :attr:`amp_ff` are all
    matrices of shape :attr:`shape`. To save memory, the matrices :attr:`phase` and :attr:`amp`
    are stored with the (smaller, but not strictly smaller) shape :attr:`slm_shape`.
    Also to save memory, :attr:`phase_ff` and :attr:`amp_ff` are set to ``None`` on construction,
    and only initialized if they need to be used. Any code additions should check for ``None``.

    Tip
    ~~~
    Due to imperfect SLM diffraction efficiency, undiffracted light will
    be present at the center of the :attr:`target`.
    This is called the zeroth order diffraction peak. To avoid this peak, consider shifting
    the data contained in :attr:`target` away from the center.

    Tip
    ~~~
    For mixed region amplitude freedom (MRAF) capabilities, set the part of the
    :attr:`target` desired as a 'noise region' to ``nan``.
    See :meth:`optimize()` for more details.
    :class:`SpotHologram` has spot-specific methods for generated noise region pattern.

    Caution
    ~~~~~~~
    By default, arguments passed to the constructor (:attr:`phase`, :attr:`amp`, ... )
    are stored directly as attributes **without copying**, where possible. These will be
    modified in place. However, :mod:`numpy` arrays passed to :mod:`cupy` will naturally
    be copied onto the GPU and arrays of incorrect :attr:`dtype` will likewise be copied
    and casted. This lack of copying is desired in many cases, such that external routines are
    accessing the same data, but the user can pass copied arrays if this behavior is undesired.

    Attributes
    ----------
    slm_shape : (int, int)
        The shape of the **near-field** device producing the hologram in the **far-field**
        in :mod:`numpy` ``(h, w)`` form. This is important to record because
        certain optimizations and calibrations depend on it. If multiple of :attr:`slm_shape`,
        :attr:`phase`, or :attr:`amp` are not ``None`` in the constructor, the shapes must agree.
        If all are ``None``, then the shape of the :attr:`target` is used instead
        (:attr:`slm_shape` == :attr:`shape`).
    phase : numpy.ndarray OR cupy.ndarray
        **Near-field** phase pattern to optimize.
        Initialized to with :meth:`random.default_rng().uniform()` by default (``None``).
        This is of shape :attr:`slm_shape`
        and (during optimization) padded to shape :attr:`shape`.
    amp : numpy.ndarray OR cupy.ndarray
        **Near-field** source amplitude pattern (i.e. image-space constraints).
        Uniform illumination is assumed by default (``None``).
        This is of shape :attr:`slm_shape`
        and (during optimization) padded to shape :attr:`shape`.
    shape : (int, int)
        The shape of the computational space in the **near-field** and **far-field**
        in :mod:`numpy` ``(h, w)`` form.
        Corresponds to the the ``"knm"`` basis in the **far-field**.
        This often differs from :attr:`slm_shape` due to padding of the **near-field**.
    target : numpy.ndarray OR cupy.ndarray
        Desired **far-field** amplitude in the ``"knm"`` basis. The goal of optimization.
        This is of shape :attr:`shape`.
    weights : numpy.ndarray OR cupy.ndarray
        The mutable **far-field** amplitude in the ``"knm"`` basis used in GS.
        Starts as :attr:`target` but may be modified by weighted feedback in WGS.
        This is of shape :attr:`shape`.
    phase_ff : numpy.ndarray OR cupy.ndarray
        Algorithm-constrained **far-field** phase in the ``"knm"`` basis.
        Stored for certain computational algorithms.
        (see :meth:`~slmsuite.holography.algorithms.Hologram.GS`).
        This is of shape :attr:`shape`.
    amp_ff : numpy.ndarray OR cupy.ndarray OR None
        **Far-field** amplitude in the ``"knm"`` basis.
        Used for comparing this, the computational result, with the :attr:`target`.
        This is of shape :attr:`shape`.
    dtype : type
        Datatype for stored **near-** and **far-field** arrays, which are **all real**.
        Some internal variables are complex. The complex numbers follow :mod:`numpy`
        `type promotion <https://numpy.org/doc/stable/reference/routines.fft.html#type-promotion>`_.
        Complex datatypes are derived from ``dtype``:

         - ``float32`` -> ``complex64`` (assumed by default)
         - ``float64`` -> ``complex128``

        ``float16`` is *not* recommended for ``dtype`` because ``complex32`` is not
        implemented by :mod:`numpy`.
    iter : int
        Tracks the current iteration number.
    method : str
        Remembers the name of the last-used optimization method. The method used for each
        iteration is stored in ``stats``.
    flags : dict
        Helper flags to store custom persistent variables for optimization.
        These flags are generally changed by passing as a ``kwarg`` to
        :meth:`~slmsuite.holography.algorithms.Hologram.optimize()`.
        Contains the following keys:

         - ``"method"`` : ``str``
            Stores the method used for optimization.
            See :meth:`~slmsuite.holography.algorithms.Hologram.optimize()`.
         - ``"fixed_phase"`` : ``bool``
            Fixes the far-field phase as mandated by certain weighted algorithms
            (see :meth:`~slmsuite.holography.algorithms.Hologram.GS()`).
         - ``"feedback"`` : ``str``
            Stores the values passed to
            :meth:`~slmsuite.holography.algorithms.Hologram.optimize()`.
         - ``"stat_groups"`` : ``list of str``
            Stores the values passed to
            :meth:`~slmsuite.holography.algorithms.Hologram.optimize()`.
         - ``"raw_stats"`` : bool
            Whether to store raw stats.
         - ``"blur_ij"`` : ``float``
            See :meth:`~slmsuite.holography.algorithms.FeedbackHologram.ijcam_to_knmslm()`.
         - Other user-defined flags.

    stats : dict
        Dictionary of useful statistics. data is stored in lists, with indices corresponding
        to each iteration. Contains:

         - ``"methods"`` : ``list of str``
            Method used for each iteration.
         - ``"flags"`` : ``dict of lists``
            Each key corresponds to a flag that was used at least once. If it is ``np.nan``
            on a given iteration, then it was undefined at that point (update functions
            keep track of all this).
         - ``"stats"`` : ``dict of dicts of lists``
            Same format as ``"flags"``, except with another layer of hierarchy corresponding
            to the source (group) of the given stats. This is to differentiate standard deviations
            computed computationally and experimentally.

        See :meth:`.update_stats()` and :meth:`.plot_stats()`.
    """

    def __init__(self, target, amp=None, phase=None, slm_shape=None, dtype=np.float32):
        r"""
        Initialize datastructures for optimization.
        When :mod:`cupy` is enabled, arrays are initialized on the GPU as :mod:`cupy` arrays:
        take care to use class methods to access these parameters instead of editing
        them directly, as :mod:`cupy` arrays behave slightly differently than numpy arrays in
        some cases.

        Parameters additional to class attributes are described below:

        Parameters
        ----------
        target : numpy.ndarray OR cupy.ndarray OR (int, int)
            Target to optimize to. The user can also pass a shape in :mod:`numpy` ``(h,
            w)`` form, and this constructor will create an empty target of all zeros.
            :meth:`.calculate_padded_shape()` can be of particular help for calculating the
            shape that will produce desired results (in terms of precision, etc).
        amp : array_like OR None
            The near-field amplitude. See :attr:`amp`. Of shape :attr:`slm_shape`.
        phase : array_like OR None
            The near-field initial phase.
            See :attr:`phase`. :attr:`phase` should only be passed if the user wants to
            precondition the optimization. Of shape :attr:`slm_shape`.
        slm_shape : (int, int) OR slmsuite.hardware.FourierSLM OR slmsuite.hardware.slms.SLM OR None
            The shape of the near-field of the SLM in :mod:`numpy` `(h, w)` form.
            Optionally, as a quality of life feature, the user can pass a
            :class:`slmsuite.hardware.FourierSLM` or
            :class:`slmsuite.hardware.slms.SLM` instead,
            and ``slm_shape`` (and ``amp`` if it is ``None``) are populated from this.
            If ``None``, tries to use the shape of ``amp`` or ``phase``, but if these
            are not present, defaults to :attr:`shape` (which is usually determined by ``target``).
        dtype : type
            See :attr:`dtype`; type to use for stored arrays.
        """
        # 1) Parse inputs
        # Parse target and create shape.
        if len(target) == 2:  # (int, int) was passed.
            self.shape = target
            target = None
        else:
            self.shape = target.shape

        # Warn the user about powers of two.
        if any(np.log2(self.shape) != np.round(np.log2(self.shape))):
            warnings.warn(
                "algorithms.py: Hologram target shape {} is not a power of 2; consider using "
                ".calculate_padded_shape() to pad to powers of 2 and speed up "
                "FFT computation. While some FFT solvers support other prime powers "
                "(3, 5, 7, ...), literature suggests that GPU support is best for powers of 2."
                "".format(self.shape)
            )

        # 1.5) Determine the shape of the SLM. We have three sources of this shape, which are
        # optional to pass, but must be self-consistent if passed:
        # a) The shape of the nearfield amplitude
        # b) The shape of the seed nearfield phase
        # c) slm_shape itself (which is set to the shape of a passed SLM, if given).
        # If no parameter is passed, these shapes are set to (nan, nan) to prepare for a
        # vote (next section).

        # Option a
        if amp is None:
            amp_shape = (np.nan, np.nan)
        else:
            amp_shape = amp.shape

        # Option b
        if phase is None:
            phase_shape = (np.nan, np.nan)
        else:
            phase_shape = phase.shape

        # Option c
        if slm_shape is None:
            slm_shape = (np.nan, np.nan)
        else:
            try:        # Check if slm_shape is a CameraSLM.
                if amp is None:
                    amp = slm_shape.slm.measured_amplitude
                slm_shape = slm_shape.slm.shape
            except:
                try:    # Check if slm_shape is an SLM
                    if amp is None:
                        amp = slm_shape.measured_amplitude
                    slm_shape = slm_shape.shape
                except: # (int, int) case
                    pass

            if len(slm_shape) != 2:
                slm_shape = (np.nan, np.nan)

        # 1.5) [cont] We now have a few options for what the shape of the SLM could be.
        # Parse these to validate consistency.
        stack = np.vstack((amp_shape, phase_shape, slm_shape))
        if np.all(np.isnan(stack)):
            self.slm_shape = self.shape
        else:
            self.slm_shape = np.around(np.nanmean(stack, axis=0)).astype(int)

            if amp is not None:
                assert np.all(self.slm_shape == np.array(amp_shape)), \
                    "algorithms.py: The shape of amplitude (via `amp` or SLM) is not equal to the " \
                    "shapes of the provided initial phase (`phase`) or SLM (via `target` or `slm_shape`)"
            if phase is not None:
                assert np.all(self.slm_shape == np.array(phase_shape)), \
                    "algorithms.py: The shape of the inital phase (`phase`) is not equal to the " \
                    "shapes of the provided amplitude (via `amp` or SLM) or SLM (via `target` or `slm_shape`)"
            if slm_shape is not None:
                assert np.all(self.slm_shape == np.array(slm_shape)), \
                    "algorithms.py: The shape of SLM (via `target` or `slm_shape`) is not equal to the " \
                    "shapes of the provided initial phase (`phase`) or amplitude (via `amp` or SLM)"

            self.slm_shape = tuple(self.slm_shape)

        # 2) Initialize variables.
        # Save the data type.
        self.dtype = dtype

        # Initialize and normalize near-field amplitude
        if amp is None:     # Uniform amplitude by default (scalar).
            self.amp = 1 / np.sqrt(np.prod(self.slm_shape))
        else:               # Otherwise, initialize and normalize.
            self.amp = cp.array(amp, dtype=dtype, copy=False)
            self.amp *= 1 / Hologram._norm(self.amp)

        # Initialize near-field phase
        self.reset_phase(phase)

        # Initialize target. reset() will handle weights.
        self._update_target(target, reset_weights=False)

        # Initialize everything else inside reset.
        self.reset(reset_phase=False, reset_flags=True)

    # Initialization helper functions.
    def reset(self, reset_phase=True, reset_flags=False):
        r"""
        Resets the hologram to an initial state. Does not restore the preconditioned ``phase``
        that may have been passed to the constructor (as this information is lost upon optimization).
        Also uses the current ``target`` rather than the ``target`` that may have been
        passed to the constructor (e.g. includes current :meth:`.refine_offset()` changes, etc).

        Parameters
        ----------
        reset_phase : bool
            Whether to additionally call :meth:`reset_phase()`.
        reset_flags : bool:
            Whether to erase the information (including passed ``kwargs``) stored in :attr:`flags`.
        """
        # Reset phase to random if needed.
        if self.phase is None or reset_phase:
            self.reset_phase()

        # Reset weights.
        self.reset_weights()

        # Reset vars.
        self.iter = 0
        self.method = ""
        if reset_flags:
            self.flags = {}
        self.stats = {"method": [], "flags": {}, "stats": {}}

        # Reset farfield storage.
        self.amp_ff = None
        self.phase_ff = None

    def reset_phase(self, phase=None):
        r"""
        Resets the hologram to a random state or to a provided phase.

        Parameters
        ----------
        phase : array_like OR None
            The near-field initial phase.
            See :attr:`phase`. :attr:`phase` should only be passed if the user wants to
            precondition the optimization. Of shape :attr:`slm_shape`.
        """
        # Reset phase to random if no phase is given.
        if phase is None:
            if cp == np:  # numpy does not support `dtype=`
                rng = np.random.default_rng()
                self.phase = rng.uniform(-np.pi, np.pi, self.slm_shape).astype(self.dtype)
            else:
                self.phase = cp.random.uniform(
                    -np.pi, np.pi, self.slm_shape, dtype=self.dtype
                )
        else:
            # Otherwise, cast as a cp.array with correct type.
            self.phase = cp.array(phase, dtype=self.dtype, copy=False)

    def reset_weights(self):
        """
        Resets the hologram weights to the :attr:`target` defaults.
        """
        # Copy from the target.
        self.weights = self.target.copy()

        # Account for MRAF by setting any noise region to zero by default.
        cp.nan_to_num(self.weights, copy=False, nan=0)

    @staticmethod
    def calculate_padded_shape(
        slm_shape,
        padding_order=1,
        square_padding=True,
        precision=np.inf,
        precision_basis="kxy",
    ):
        """
        Helper function to calculate the shape of the computational space.
        For a given base ``slm_shape``, pads to the user's requirements.
        If the user chooses multiple requirements, the largest
        dimensions for the shape are selected.
        By default, pads to the smallest square power of two that
        encapsulates the original ``slm_shape``.

        Tip
        ~~~
        See also the first tip in the constructor of :class:`Hologram` for more information
        about the importance of padding.

        Note
        ~~~~
        Under development: a parameter to pad based on available memory
        (see :meth:`_calculate_memory_constrained_shape()`).

        Parameters
        ----------
        slm_shape : (int, int) OR slmsuite.hardware.FourierSLM
            The original shape of the SLM in :mod:`numpy` `(h, w)` form. The user can pass a
            :class:`slmsuite.hardware.FourierSLM` instead, and should pass this
            when using the ``precision`` parameter.
        padding_order : int
            Scales to the ``padding_order`` th larger power of 2.
            A ``padding_order`` of zero does nothing. For instance, an SLM
            with shape ``(720, 1280)`` would yield
            ``(720, 1280)`` for ``padding_order=0``,
            ``(1024, 2048)`` for ``padding_order=1``, and
            ``(2048, 4096)`` for ``padding_order=2``.
        square_padding : bool
            If ``True``, sets the smaller shape dimension to that of the larger, yielding a square.
        precision : float
            Returns the shape that produces a computational k-space with resolution smaller
            than ``precision``. The default, infinity, requests a padded shape larger
            than zero, so ``padding_order`` will dominate.
        precision_basis : str
            Basis for the precision. Can be ``"ij"`` (camera) or ``"kxy"`` (normalized blaze).

        Returns
        -------
        (int, int)
            Shape of the computational space which satisfies the above requirements.
        """
        cameraslm = None
        if hasattr(slm_shape, "slm"):
            cameraslm = slm_shape
            slm_shape = cameraslm.slm.shape

        # Handle precision
        if np.isfinite(precision) and cameraslm is not None:
            if precision <= 0:
                raise ValueError(
                    "algorithms.py: Precision passed to calculate_padded_shape() must be positive."
                )
            dpixel = np.amin([cameraslm.slm.dx, cameraslm.slm.dy])
            fs = 1 / dpixel  # Sampling frequency

            if precision_basis == "ij":
                slm_range = np.amax(cameraslm.kxyslm_to_ijcam([fs, fs]))
                pixels = slm_range / precision
            elif precision_basis == "kxy":
                pixels = fs / precision

            # Raise to the nearest greater power of 2.
            pixels = np.power(2, int(np.ceil(np.log2(pixels))))
            precision_shape = (pixels, pixels)
        elif np.isfinite(precision):
            raise ValueError(
                "algorithms.py: Must pass a CameraSLM object under slm_shape "
                "to implement calculate_padded_shape() precision calculations!"
            )
        else:
            precision_shape = slm_shape

        # Handle padding_order
        if padding_order > 0:
            padding_shape = np.power(
                2, np.ceil(np.log2(slm_shape)) + padding_order - 1
            ).astype(int)
        else:
            padding_shape = slm_shape

        # Take the largest and square if desired.
        shape = tuple(np.amax(np.vstack((precision_shape, padding_shape)), axis=0))

        if square_padding:
            largest = np.amax(shape)
            shape = (largest, largest)

        return shape

    @staticmethod
    def _calculate_memory_constrained_shape(device=0, dtype=np.float32):
        memory = Hologram.get_mempool_limit(device=device)

        num_values = memory / np.dtype(dtype).itemsize

        # (4 real stored arrays, 2 complex runtime arrays [twice memory])
        num_values_per_array = num_values / 8

        return np.sqrt(num_values_per_array)

    # Core optimization function.
    def optimize(
        self,
        method="GS",
        maxiter=20,
        verbose=True,
        callback=None,
        feedback=None,
        stat_groups=[],
        **kwargs
    ):
        r"""
        Optimizers to solve the "phase problem": approximating the near-field phase that
        transforms a known near-field source amplitude to a desired near-field
        target amplitude.
        Supported optimization methods include:

        - Gerchberg-Saxton (GS) phase retrieval.

            ``'GS'``

              `An iterative algorithm for phase retrieval
              <http://www.u.arizona.edu/~ppoon/GerchbergandSaxton1972.pdf>`_,
              accomplished by moving back and forth between the imaging and Fourier domains,
              with amplitude corrections applied to each.
              This is implemented using fast Fourier transforms, potentially GPU-accelerated.

        - Weighted Gerchberg-Saxton (WGS) phase retrieval algorithms of various flavors.
          Improves the uniformity of GS-computed focus arrays using weighting methods and
          techniques from literature. The ``method`` keywords are:

            ``'WGS-Leonardo'``

              `The original WGS algorithm <https://doi.org/10.1364/OE.15.001913>`_.
              Weights the target amplitudes by the ratio of mean amplitude to computed
              amplitude, which amplifies weak spots while attenuating strong spots. Uses
              the following weighting function:

              .. math:: \mathcal{W} = \mathcal{W}\left(\frac{\mathcal{T}}{\mathcal{F}}\right)^p

              where :math:`\mathcal{W}`, :math:`\mathcal{T}`, and :math:`\mathcal{F}`
              are the weight amplitudes,
              target (goal) amplitudes, and
              feedback (measured) amplitudes,
              and :math:`p` is the power passed as ``"feedback_exponent"`` in
              :attr:`~slmsuite.holography.algorithms.Hologram.flags` (see ``kwargs``).
              The power :math:`p` defaults to .9 if not passed. In general, smaller
              :math:`p` will lead to slower yet more stable optimization.

            ``'WGS-Kim'``

              `Improves the convergence <https://doi.org/10.1364/OL.44.003178>`_
              of `Leonardo` by fixing the far-field phase
              strictly after a desired number of net iterations
              specified by ``"fix_phase_iteration"``
              or after exceeding a desired efficiency
              (fraction of far-field energy at the desired points)
              specified by ``"fix_phase_efficiency"``

            ``'WGS-Nogrette'``

              Weights target intensities by `a tunable gain factor <https://doi.org/10.1103/PhysRevX.4.021034>`_.

              .. math:: \mathcal{W} = \mathcal{W}/\left(1 - f\left(1 - \mathcal{F}/\mathcal{T}\right)\right)

              where :math:`f` is the gain factor passed as ``"feedback_factor"`` in
              :attr:`~slmsuite.holography.algorithms.Hologram.flags` (see ``kwargs``).
              The factor :math:`f` defaults to .1 if not passed.

              Note that while Nogrette et al compares powers, this implementation
              compares amplitudes for speed. These are identical to first order.

        - The option for `Mixed Region Amplitude Freedom (MRAF)
          <https://doi.org/10.1007/s10043-018-0456-x>`_ feedback. In standard
          iterative algorithms, the entire Fourier-domain unpatterned field is replaced with zeros.
          This is disadvantageous because a desired farfield pattern might not be especially
          compatible with a given nearfield amplitude, or otherwise. MRAF enables
          "noise regions" where some fraction of the given farfield is **not** replaced
          with zeros and instead is allowed to vary. In practice, MRAF is
          enabled by setting parts of the :attr:`target` to ``nan``; these regions act
          as the noise regions. The ``"mraf_factor"`` flag in
          :attr:`~slmsuite.holography.algorithms.Hologram.flags`
          allows the user to partially attenuate the noise regions.
          A factor of 0 fully attenuates the noise region (normal WGS behavior).
          A factor of 1 does not attenuate the noise region at all (the default).
          Middle ground is recommended, but is application-dependent as a
          tradeoff between improving pattern fidelity and maintaining pattern efficiency.

          As examples, consider two cases where MRAF can be useful:

            - **Sloping a top hat.** Suppose we want very flat amplitude on a beam.
              Requesting a sharp edge to this beam can lead to fringing effects at the
              boundary which mitigate flatness both inside and outside the beam.
              If instead a noise region is defined in a band surrounding the beam,
              the noise region will be filled with whatever slope best enables the
              desired flat beam.

            - **Mitigating diffractive orders.** Without MRAF, spot patterns with high
              crystallinity often have "ghost" diffractive orders which continue the
              pattern past the edges of requested spots. Even though these orders are attenuated
              during each phase retrieval iteration, they remain part of the best
              solution for the recovered phase. With MRAF, a noise region can help solve
              for retrieved phase which does not generate these undesired orders.

        Caution
        ~~~~~~~
        Requesting ``stat_groups`` will slow the speed of optimization due to the
        overhead of processing and saving statistics, especially in the case of
        GPU-accelerated optimization where significant time cost is incurred by
        moving these statistics to the CPU. This is especially apparent in the case
        of fully-computational holography, where this effect can slow what is otherwise
        a fully-GPU-contained loop by an order magnitude.

        Tip
        ~~~
        This function uses a parameter naming convention borrowed from
        :meth:`scipy.optimize.minimize()` and other functions in
        :mod:`scipy.optimize`. The parameters ``method``, ``maxiter``, and ``callback``
        have the same functionality as the equivalently-named parameters in
        :meth:`scipy.optimize.minimize()`.

        Parameters
        ----------
        method : str
            Optimization method to use. See the list of optimization methods above.
        maxiter : int
            Number of iterations to optimize before terminating.
        verbose : bool OR int
            Whether to display :mod:`tqdm` progress bars.
            These bars are also not displayed for ``maxiter <= 1``.
            If ``verbose`` is greater than 1, then flags are printed as a preamble.
        callback : callable OR None
            Same functionality as the equivalently-named parameter in
            :meth:`scipy.optimize.minimize()`. ``callback`` must accept a Hologram
            or Hologram subclass as the single argument. If ``callback`` returns
            ``True``, then the optimization exits. Ignored if ``None``.
        feedback : str OR None
            Type of feedback to use during optimization, for instance when weighting in ``"WGS"``.
            For direct instances of :class:`Hologram`, this can only
            be ``"computational"`` feedback. Subclasses support more types of feedback.
            Supported feedback options include the following:

            - ``"computational"`` Uses the the projected farfield pattern (transform of
              the complex nearfield) as feedback.
            - ``"experimental"`` Uses a camera contained in a passed ``cameraslm`` as feedback.
              Specific to subclasses of :class:`FeedbackHologram`.
            - ``"computational_spot"`` Takes the computational result (the projected
              farfield pattern) and integrates regions around the expected positions of
              spots in an optical focus array. More stable than ``"computational"`` for spots.
              Specific to subclasses of :class:`SpotHologram`.
            - ``"experimental_spot"`` Takes the experimental result (the image from a
              camera) and integrates regions around the expected positions of
              spots in an optical focus array. More stable than ``"experimental"`` for spots.
              Specific to subclasses of :class:`SpotHologram`.
            - ``"external_spot"`` Uses some external user-provided metric for spot
              feedback. See :attr:`external_spot_amp`.
              Specific to subclasses of :class:`SpotHologram`.

        stat_groups : list of str OR None
            Strings representing types of feedback (data gathering) upon which
            statistics should be derived. These strings correspond to valid types of
            feedback (see above). For instance, if ``"experimental"`` is passed as a
            stat group, statistics on the pixels in the experimental feedback image will
            automatically be computed and stored for each iteration of optimization.
            However, this comes with overhead (see above warning).
        **kwargs : dict, optional
            Various weight keywords and values to pass depending on the weight method.
            These are passed into :attr:`flags`. See options documented in the constructor.
        """
        # 0) Check and record method.
        methods = list(ALGORITHM_DEFAULTS.keys())
        if not method in methods:
            raise ValueError(
                "algorithms.py: unrecognized method '{}'.\n"
                "Valid methods include {}".format(method, methods)
            )
        self.method = method

        # 1) Parse flags:
        # 1.1) Set defaults if not already set.
        for flag, value in ALGORITHM_DEFAULTS[method].items():
            if not flag in self.flags:
                self.flags[flag] = value
        if not "fixed_phase" in self.flags:
            self.flags["fixed_phase"] = False

        # 1.2) Parse kwargs as flags.
        for flag in kwargs:
            self.flags[flag] = kwargs[flag]

        # 1.3) Add in non-defaulted flags, with error checks
        for group in stat_groups:
            if not (group in FEEDBACK_OPTIONS):
                raise ValueError(
                    "algorithms.py: statistics group '{}' not recognized as a feedback option.\n"
                    "Valid options: {}".format(group, FEEDBACK_OPTIONS)
                )
        self.flags["stat_groups"] = stat_groups

        if feedback is not None:
            if not (feedback in FEEDBACK_OPTIONS):
                raise ValueError(
                    "algorithms.py: feedback '{}' not recognized as a feedback option.\n"
                    "Valid options: {}".format(group, FEEDBACK_OPTIONS)
                )
            self.flags["feedback"] = feedback

        # 1.4) Print the flags if verbose.
        if verbose > 1:
            print("Optimizing with '{}' using the following method-specific flags:".format(self.method))
            pprint.pprint({
                key:value for (key, value) in self.flags.items()
                if key in ALGORITHM_DEFAULTS[method]
            })
            print("", end="", flush=True)   # Prevent tqdm conflicts.

        # 2) Prepare the iterations iterable.
        iterations = range(maxiter)

        # 2.1) Decide whether to use a tqdm progress bar. Don't use a bar for maxiter == 1.
        if verbose and maxiter > 1:
            iterations = tqdm(iterations)

        # 3) Switch between optimization methods (currently only GS- or WGS-type is supported).
        if "GS" in method:
            self.GS(iterations, callback)

    # Optimization methods (currently only GS- or WGS-type is supported).
    def GS(self, iterations, callback):
        """
        GPU-accelerated Gerchberg-Saxton (GS) iterative phase retrieval.

        Solves the "phase problem": approximates the near-field phase that
        transforms a known near-field source amplitude to a known near-field
        target amplitude.

        Caution
        ~~~~~~~
        This function should be called through :meth:`.optimize()` and not called
        directly. It is left as a public function exposed in documentation to clarify
        how the internals of :meth:`.optimize()` work.

        Note
        ~~~~
        FFTs are **not** in-place in this algorithm. In both non-:mod:`cupy` and
        :mod:`cupy` implementations, :mod:`numpy.fft` does not support in-place
        operations.  However, :mod:`scipy.fft` does in both. In the future, we may move to the scipy
        implementation. However, neither :mod:`numpy` or :mod:`scipy` ``fftshift`` support
        in-place movement (for obvious reasons). For even faster computation, algorithms should
        consider **not shifting** the FFT result, and instead shifting measurement data / etc to
        this unshifted basis. We might also implement `get_fft_plan
        <https://docs.cupy.dev/en/stable/reference/generated/cupyx.scipy.fftpack.get_fft_plan.html>`_
        for even faster FFTing.

        Parameters
        ----------
        iterations : iterable
            Number of loop iterations to run. Is an iterable to pass a :mod:`tqdm` iterable.
        callback : callable OR None
            See :meth:`.optimize()`.
        """
        # Proxy to initialize nearfield with the correct shape and correct (complex) type.
        nearfield = cp.zeros(self.shape, dtype=type(1j * self.dtype(1)))

        # Precompute MRAF helper variables.
        mraf_variables = self._mraf_helper_routines()

        # Helper variables for speeding up source phase and amplitude fixing.
        (i0, i1, i2, i3) = toolbox.unpad(self.shape, self.slm_shape)

        for _ in iterations:
            # 1) Nearfield -> farfield
            # 1.1) Fix the relevant part of the nearfield amplitude to the source amplitude.
            # Everything else is zero because power outside the SLM is assumed unreflected.
            # This is optimized for when shape is much larger than slm_shape.
            nearfield.fill(0)
            nearfield[i0:i1, i2:i3] = self.amp * cp.exp(1j * self.phase)

            # 1.2) FFT to move to the farfield.
            farfield = cp.fft.fftshift(cp.fft.fft2(cp.fft.fftshift(nearfield), norm="ortho"))

            # 2) Midloop: caching, prep
            # 2.1) Before callback(), cleanup such that it can access updated amp_ff and images.
            self._midloop_cleaning(farfield)

            # 2.2) Run step function if present and check termination conditions.
            if callback is not None:
                if callback(self):
                    break

            # 2.3) Evaluate method-specific routines, stats, etc.
            # If you want to add new functionality to GS, do so here to keep the main loop clean.
            self._GS_farfield_routines(farfield, mraf_variables)

            # 3) Farfield -> nearfield.
            nearfield = cp.fft.ifftshift(cp.fft.ifft2(cp.fft.ifftshift(farfield), norm="ortho"))

            # 3.1) Grab the phase from the complex nearfield.
            # Use arctan2() directly instead of angle() for in-place operations (out=).
            self.phase = cp.arctan2(
                nearfield.imag[i0:i1, i2:i3],
                nearfield.real[i0:i1, i2:i3],
                out=self.phase,
            )

            # 3.2) Increment iteration.
            self.iter += 1

        # Update the final far-field
        nearfield.fill(0)
        nearfield[i0:i1, i2:i3] = self.amp * cp.exp(1j * self.phase)
        farfield = cp.fft.fftshift(cp.fft.fft2(cp.fft.fftshift(nearfield), norm="ortho"))
        self.amp_ff = cp.abs(farfield)
        self.phase_ff = cp.angle(farfield)

    def _mraf_helper_routines(self):
        # MRAF helper variables
        noise_region = cp.isnan(self.target)
        mraf_enabled = bool(cp.any(noise_region))

        if not mraf_enabled:
            return {
                "noise_region":None,
                "signal_region":None,
                "mraf_enabled":False,
                "mraf_factor":None,
                "where_working":None
            }

        signal_region = cp.logical_not(noise_region)
        mraf_factor = self.flags.get("mraf_factor", None)
        if mraf_factor is not None:
            if mraf_factor < 0:
                raise ValueError("mraf_factor={} should not be negative.".format(mraf_factor))

        # `where=` functionality is needed for MRAF, but this is a undocumented/new cupy feature.
        # We test whether it is available https://github.com/cupy/cupy/pull/7281
        try:
            test = cp.arange(10)
            cp.multiply(test, test, where=test > 5)
            where_working = True
        except:
            try:
                test = cp.arange(10)
                cp.multiply(test, test, _where=test > 5)
                where_working = False
            except:
                raise Exception(
                    "MRAF not supported on this system. Arithmetic `where=` is needed. "
                    "See https://github.com/cupy/cupy/pull/7281."
                )

        return {
            "noise_region":noise_region,
            "signal_region":signal_region,
            "mraf_enabled":mraf_enabled,
            "mraf_factor":mraf_factor,
            "where_working":where_working
        }

    def _midloop_cleaning(self, farfield):
        # 2.1) Cache amp_ff for weighting (if None, will init; otherwise in-place).
        self.amp_ff = cp.abs(farfield, out=self.amp_ff)

        # 2.2) Erase images from the past loop. FUTURE: Make better and faster.
        if hasattr(self, "img_ij"):     self.img_ij = None
        if hasattr(self, "img_knm"):    self.img_knm = None

    def _GS_farfield_routines(self, farfield, mraf_variables):
        # Update statistics
        self.update_stats(self.flags["stat_groups"])

        # Weight, if desired.
        if "WGS" in self.method:
            self._update_weights()

            # Decide whether to fix phase.
            if "Kim" in self.method:
                was_not_fixed = not self.flags["fixed_phase"]

                # Enable based on efficiency.
                if self.flags["fix_phase_efficiency"] is not None:
                    stats = self.stats["stats"]
                    groups = tuple(stats.keys())

                    assert len(stats) > 0, "Must track statistics to fix phase based on efficiency!"

                    eff = stats[groups[-1]]["efficiency"][self.iter]
                    if eff > self.flags["fix_phase_efficiency"]:
                        self.flags["fixed_phase"] = True

                # Enable based on iterations.
                if was_not_fixed:
                    if self.iter >= self.flags["fix_phase_iteration"] - 1:
                        previous = self.stats["flags"]["fixed_phase"]
                        contiguous_falses = all(
                            [not previous[-1-i] for i in range(self.flags["fix_phase_iteration"])]
                        )
                        if contiguous_falses:
                            self.flags["fixed_phase"] = True

                # Save the phase if we are going from unfixed to fixed.
                if self.flags["fixed_phase"] and self.phase_ff is None or was_not_fixed:
                    self.phase_ff = cp.angle(farfield)
            else:
                self.flags["fixed_phase"] = False

        mraf_enabled = mraf_variables["mraf_enabled"]

        # Fix amplitude, potentially also fixing the phase.
        if not mraf_enabled:
            if ("fixed_phase" in self.flags and self.flags["fixed_phase"]):
                # Set the farfield to the stored phase and updated weights.
                cp.exp(1j * self.phase_ff, out=farfield)
                cp.multiply(farfield, self.weights, out=farfield)
            else:
                # Set the farfield amplitude to the updated weights.
                cp.divide(farfield, cp.abs(farfield), out=farfield)
                cp.multiply(farfield, self.weights, out=farfield)
                cp.nan_to_num(farfield, copy=False, nan=0)
        else:
            noise_region =  mraf_variables["noise_region"]
            signal_region = mraf_variables["signal_region"]
            mraf_factor =   mraf_variables["mraf_factor"]
            where_working = mraf_variables["where_working"]

            if ("fixed_phase" in self.flags and self.flags["fixed_phase"]):
                # Set the farfield to the stored phase and updated weights, in the signal region.
                if where_working:
                    cp.exp(1j * self.phase_ff, where=signal_region, out=farfield)
                    cp.multiply(farfield, self.weights, where=signal_region, out=farfield)
                    if mraf_factor is not None: cp.multiply(farfield, mraf_factor, where=noise_region, out=farfield)
                else:
                    cp.exp(1j * self.phase_ff, _where=signal_region, out=farfield)
                    cp.multiply(farfield, self.weights, _where=signal_region, out=farfield)
                    if mraf_factor is not None: cp.multiply(farfield, mraf_factor, _where=noise_region, out=farfield)
            else:
                # Set the farfield amplitude to the updated weights, in the signal region.
                if where_working:
                    cp.divide(farfield, cp.abs(farfield), where=signal_region, out=farfield)
                    cp.multiply(farfield, self.weights, where=signal_region, out=farfield)
                    if mraf_factor is not None: cp.multiply(farfield, mraf_factor, where=noise_region, out=farfield)
                else:
                    cp.divide(farfield, cp.abs(farfield), _where=signal_region, out=farfield)
                    cp.multiply(farfield, self.weights, _where=signal_region, out=farfield)
                    if mraf_factor is not None: cp.multiply(farfield, mraf_factor, _where=noise_region, out=farfield)
                cp.nan_to_num(farfield, copy=False, nan=0)

    # User interactions: Changing the target and recovering the nearfield phase and complex farfield.
    def _update_target(self, new_target, reset_weights=False, plot=False):
        """
        Change the target to something new. This method handles cleaning and normalization.

        This method is shelled by :meth:`update_target()` such that it is still accessible
        in the case that a subclass overwrites :meth:`update_target()`.

        Parameters
        ----------
        new_target : array_like OR None
            If ``None``, sets the target to zero. The ``None`` case is used internally
            by :class:`SpotHologram`.
        reset_weights : bool
            Whether to overwrite ``weights`` with ``target``.
        plot : bool
            Calls :meth:`.plot_farfield()` on :attr:`target`.
        """
        if new_target is None:
            self.target = cp.zeros(shape=self.shape, dtype=self.dtype)
        else:
            assert new_target.shape == self.shape, (
                "Target must be of appropriate shape. "
                "Initialize a new Hologram if a different shape is desired."
            )

            self.target = cp.array(new_target, dtype=self.dtype, copy=False)
            cp.abs(self.target, out=self.target)
            self.target *= 1 / Hologram._norm(self.target)

        if reset_weights:
            self.reset_weights()

        if plot:
            self.plot_farfield(self.target)

    def update_target(self, new_target, reset_weights=False, plot=False):
        """
        Change the target to something new. This method handles cleaning and normalization.

        Parameters
        ----------
        new_target : array_like OR None
            New :attr:`target` to optimize towards. Should be of shape :attr:`shape`.
            If ``None``, :attr:`target` is zeroed (used internally, but probably should not
            be used by a user).
        reset_weights : bool
            Whether to update the :attr:`weights` to this new :attr:`target`.
        plot : bool
            Calls :meth:`.plot_farfield()` on :attr:`target`.
        """
        self._update_target(new_target=new_target, reset_weights=reset_weights, plot=plot)

    def extract_phase(self):
        r"""
        Collects the current nearfield phase from the GPU with :meth:`cupy.ndarray.get()`.
        Also shifts the :math:`[-\pi, \pi]` range of :meth:`numpy.arctan2()` to :math:`[0, 2\pi]`
        for faster writing to the SLM (see :meth:`~slmsuite.hardware.slms.slm.SLM.write()`).

        Returns
        -------
        numpy.ndarray
            Current nearfield phase computed by GS.
        """
        if cp != np:
            return self.phase.get() + np.pi
        return self.phase + np.pi

    def extract_farfield(self):
        r"""
        Collects the current complex farfield from the GPU with :meth:`cupy.ndarray.get()`.

        Returns
        -------
        numpy.ndarray
            Current farfield computed by GS.
        """
        nearfield = toolbox.pad(self.amp * cp.exp(1j * self.phase), self.shape)
        farfield = cp.fft.fftshift(cp.fft.fft2(cp.fft.fftshift(nearfield), norm="ortho"))

        if cp != np:
            return farfield.get()
        return farfield

    # Weighting functions.
    def _update_weights_generic(
            self, weight_amp, feedback_amp, target_amp=None, mp=cp, nan_checks=True
        ):
        """
        Helper function to process weight feedback according to the chosen weighting method.

        Caution
        ~~~~~~~
        ``weight_amp`` **is** modified in-place.

        Parameters
        ----------
        weight_amp : numpy.ndarray OR cupy.ndarray
            A :class:`~slmsuite.holography.SpotArray` instance containing locations
            where the feedback weight should be calculated.
        feedback_amp : numpy.ndarray OR cupy.ndarray
            Resulting amplitudes corresponding to ``weight_amp``.
            Should be the same size as ``weight_amp``.
        target_amp : numpy.ndarray OR cupy.ndarray OR None
            Necessary in the case where ``target_amp`` is not uniform, such that the weighting can
            properly be applied to bring the feedback closer to the target. If ``None``, is assumed
            to be uniform. Should be the same size as ``weight_amp``.
        mp : module
            This function is used by both :mod:`cupy` and :mod:`numpy`, so we have the option
            for either. Defaults to :mod:`cupy`.
        nan_checks : bool
            Whether to enable checks to avoid division by zero or ``nan`` infiltration.

        Returns
        -------
        numpy.ndarray OR cupy.ndarray
            The updated ``weight_amp``.
        """
        assert self.method[:4] == "WGS-", "For now, assume weighting is for WGS."
        method = self.method[4:]

        # Parse feedback_amp
        if target_amp is None:  # Uniform
            feedback_corrected = mp.array(feedback_amp, copy=True, dtype=self.dtype)
        else:  # Non-uniform
            feedback_corrected = mp.array(feedback_amp, copy=True, dtype=self.dtype)
            feedback_corrected *= 1 / Hologram._norm(feedback_corrected, mp=mp)

            mp.divide(feedback_corrected, mp.array(target_amp, copy=False), out=feedback_corrected)

            if nan_checks:
                feedback_corrected[feedback_corrected == np.inf] = 1
                feedback_corrected[mp.array(target_amp) == 0] = 1

                mp.nan_to_num(feedback_corrected, copy=False, nan=1)

        # Fix feedback according to the desired method.
        if "leonardo" in method.lower() or "kim" in method.lower():
            # 1/(x^p)
            mp.power(feedback_corrected, -self.flags["feedback_exponent"], out=feedback_corrected)
        elif "nogrette" in method.lower():
            # Taylor expand 1/(1-g(1-x)) -> 1 + g(1-x) + (g(1-x))^2 ~ 1 + g(1-x)
            feedback_corrected *= -(1 / mp.nanmean(feedback_corrected))
            feedback_corrected += 1
            feedback_corrected *= -self.flags["feedback_factor"]
            feedback_corrected += 1
            mp.reciprocal(feedback_corrected, out=feedback_corrected)
        else:
            raise RuntimeError(
                "Method "
                "{}"
                " not recognized by Hologram.optimize()".format(self.method)
            )

        if nan_checks:
            feedback_corrected[feedback_corrected == np.inf] = 1

        # Update the weights.
        weight_amp *= feedback_corrected

        if nan_checks:
            mp.nan_to_num(weight_amp, copy=False, nan=.0001)
            weight_amp[weight_amp == np.inf] = 1

        # Normalize amp, as methods may have broken conservation.
        norm = Hologram._norm(weight_amp, mp=mp)
        weight_amp *= 1 / norm

        return weight_amp

    def _update_weights(self):
        """
        Change :attr:`weights` to optimize towards the :attr:`target` using feedback from
        :attr:`amp_ff`, the computed farfield amplitude. This function also updates stats.
        """
        feedback = self.flags["feedback"]

        if feedback == "computational":
            self._update_weights_generic(self.weights, self.amp_ff, self.target)

    # Statistics handling.
    @staticmethod
    def _calculate_stats(
        feedback_amp,
        target_amp,
        mp=cp,
        efficiency_compensation=True,
        total=None,
        raw=False
    ):
        """
        Helper function to analyze how close the feedback is to the target.

        Parameters
        ----------
        feedback_amp : numpy.ndarray OR cupy.ndarray
            Computational or measured result of holography.
        target_amp : numpy.ndarray OR cupy.ndarray
            Target of holography.
        mp : module
            This function is used by both :mod:`cupy` and :mod:`numpy`, so we have the option
            for either. Defaults to :mod:`cupy`.
        efficiency_compensation : bool
            Whether to scale the ``feedback`` based on the overlap with the ``target``.
            This is more accurate for images, but less accurate for SpotHolograms.
        total : float OR None
            Total power recorded by the feedback mechanism. This may differ from the
            power concentrated in ``feedback_amp ** 2`` because, for instance, power
            might exist outside spot integration regions.
            If ``None``, uses an overlap integral method to compute efficiency.
        raw : bool
            Passes the ``"raw_stats"`` flag. If ``True``, stores the
            raw feedback and raw feedback-target ratio for each pixel or spot instead of
            only the derived statistics.
        """
        # Downgrade to numpy if necessary
        if mp == np and (hasattr(feedback_amp, "get") or hasattr(target_amp, "get")):
            if hasattr(feedback_amp, "get"):
                feedback_amp = feedback_amp.get()

            if hasattr(target_amp, "get"):
                target_amp = target_amp.get()

            if total is not None:
                total = float(total)

        feedback_amp = mp.array(feedback_amp, copy=False)
        target_amp = mp.array(target_amp, copy=False)

        feedback_pwr = mp.square(feedback_amp)
        target_pwr = mp.square(target_amp)

        if total is not None:
            efficiency = float(mp.sum(feedback_pwr)) / total

        # Normalize.
        feedback_pwr_sum = mp.sum(feedback_pwr)
        feedback_pwr *= 1 / feedback_pwr_sum
        feedback_amp *= 1 / mp.sqrt(feedback_pwr_sum)

        target_pwr_sum = mp.sum(target_pwr)
        target_pwr *= 1 / target_pwr_sum
        target_amp *= 1 / mp.sqrt(target_pwr_sum)

        if total is None:
            # Efficiency overlap integral.
            efficiency = np.square(float(mp.sum(mp.multiply(target_amp, feedback_amp))))
            if efficiency_compensation:
                feedback_pwr *= 1 / efficiency

        # Make some helper lists; ignoring power where target is zero.
        mask = mp.nonzero(target_pwr)

        feedback_pwr_masked = feedback_pwr[mask]
        target_pwr_masked = target_pwr[mask]

        ratio_pwr = mp.divide(feedback_pwr_masked, target_pwr_masked)
        pwr_err = target_pwr_masked - feedback_pwr_masked

        # Compute the remaining stats.
        rmin = float(mp.amin(ratio_pwr))
        rmax = float(mp.amax(ratio_pwr))
        uniformity = 1 - (rmax - rmin) / (rmax + rmin)

        pkpk_err = pwr_err.size * float(mp.amax(pwr_err) - mp.amin(pwr_err))
        std_err = pwr_err.size * float(mp.std(pwr_err))

        final_stats = {
            "efficiency": float(efficiency),
            "uniformity": float(uniformity),
            "pkpk_err": float(pkpk_err),
            "std_err": float(std_err),
        }

        if raw:
            ratio_pwr_full = np.full_like(target_pwr, np.nan)

            if mp == np:
                final_stats["raw_pwr"] = np.square(feedback_amp)
                ratio_pwr_full[mask] = ratio_pwr
            else:
                final_stats["raw_pwr"] = np.square(feedback_amp).get()
                ratio_pwr_full[mask] = ratio_pwr.get()

            final_stats["raw_pwr_ratio"] = ratio_pwr_full

        return final_stats

    def _calculate_stats_computational(self, stats, stat_groups=[]):
        """
        Wrapped by :meth:`Hologram.update_stats()`.
        """
        if "computational" in stat_groups:
            stats["computational"] = self._calculate_stats(
                self.amp_ff,
                self.target,
                efficiency_compensation=False,
                raw="raw_stats" in self.flags and self.flags["raw_stats"]
            )

    def _update_stats_dictionary(self, stats):
        """
        Helper function to manage additions to the :attr:`stats`.

        Parameters
        ----------
        stats : dict of dicts
            Dictionary of groups, each group containing a dictionary of stats.
        """
        # Update methods
        M = len(self.stats["method"])
        diff = self.iter + 1 - M
        if diff > 0:  # Extend methods
            self.stats["method"].extend(["" for _ in range(diff)])
            M = self.iter + 1
        self.stats["method"][self.iter] = self.method  # Update method

        # Update flags
        flaglist = set(self.flags.keys()).union(set(self.stats["flags"].keys()))
        for flag in flaglist:
            # Extend flag
            if not flag in self.stats["flags"]:
                self.stats["flags"][flag] = [np.nan for _ in range(M)]
            else:
                diff = self.iter + 1 - len(self.stats["flags"][flag])
                if diff > 0:
                    self.stats["flags"][flag].extend([np.nan for _ in range(diff)])

            # Update flag
            if flag in self.flags:
                self.stats["flags"][flag][self.iter] = self.flags[flag]

        # Update stats
        grouplist = set(stats.keys()).union(set(self.stats["stats"].keys()))
        if len(grouplist) > 0:
            statlists = [set(stats[group].keys()) for group in stats.keys()]
            if len(self.stats["stats"].keys()) > 0:
                key = next(iter(self.stats["stats"]))
                statlists.append(set(self.stats["stats"][key].keys()))
            statlist = set.union(*statlists)

            for group in grouplist:
                # Check this group
                if not group in self.stats["stats"]:
                    self.stats["stats"][group] = {}

                if len(statlist) > 0:
                    for stat in statlist:
                        # Extend stat
                        if not stat in self.stats["stats"][group]:
                            self.stats["stats"][group][stat] = [
                                np.nan for _ in range(M)
                            ]
                        else:
                            diff = self.iter + 1 - len(self.stats["stats"][group][stat])
                            if diff > 0:
                                self.stats["stats"][group][stat].extend(
                                    [np.nan for _ in range(diff)]
                                )

                        # Update stat
                        if group in stats.keys():
                            if stat in stats[group].keys():
                                self.stats["stats"][group][stat][self.iter] = stats[group][
                                    stat
                                ]

    def update_stats(self, stat_groups=[]):
        """
        Calculate statistics corresponding to the desired ``stat_groups``.

        Parameters
        ----------
        stat_groups : list of str
            Which groups or types of statistics to analyze.
        """
        stats = {}

        self._calculate_stats_computational(stats, stat_groups)

        self._update_stats_dictionary(stats)

    def export_stats(self, file_path, include_state=True):
        """
        Uses :meth:`write_h5` to export the statistics hierarchy to a given h5 file.

        Parameters
        ----------
        file_path : str
            Full path to the file to read the data from.
        include_state : bool
            If ``True``, also includes all other attributes of :class:`Hologram`
            except for :attr:`dtype` (cannot pickle) and :attr:`amp_ff` (can regenerate).
            These attributes are converted to :mod:`numpy` if necessary.
            Note that the intent is **not** to produce a
            runnable :class:`Hologram` by default (as this would require pickling hardware
            interfaces), but rather to provide extra information for debugging.
        """
        # Save attributes, converting to numpy when necessary.
        if include_state:
            to_save = {
                "slm_shape" : self.slm_shape,
                "phase" : self.phase,
                "amp" : self.amp,
                "shape" : self.shape,
                "target" : self.target,
                "weights" : self.weights,
                "phase_ff" : self.phase_ff,
                "iter" : self.iter,
                "method" : self.method,
                "flags" : self.flags
            }

            for key in to_save.keys():
                if hasattr(to_save[key], "get") and not isinstance(to_save[key], dict):
                    to_save[key] = to_save[key].get()
        else:
            to_save = {}

        # Save stats.
        to_save["stats"] = self.stats

        write_h5(file_path, to_save)

    def import_stats(self, file_path, include_state=True):
        """
        Uses :meth:`write_h5` to import the statistics hierarchy from a given h5 file.

        Tip
        ~~~
        Enabling the ``"raw_stats"`` flag will export feedback data from each iteration
        instead of only derived statistics. Consider enabling this to save more detailed
        information upon export.

        Parameters
        ----------
        file_path : str
            Full path to the file to read the data from.
        include_state : bool
            If ``True``, also overwrite all other attributes of :class:`Hologram`
            except for :attr:`dtype` and :attr:`amp_ff`.
        """
        from_save = read_h5(file_path)

        # Overwrite attributes if desired.
        if include_state:
            if len(from_save.keys()) <= 1:
                raise ValueError(
                    "algorithms.py: State was not stored in file '{}'"
                    "and cannot be imported".format(file_path)
                )

            is_cupy = ["phase", "amp", "target", "weights", "phase_ff"]
            for key in from_save.keys():
                if key != "stats":
                    if key in is_cupy:
                        setattr(self, key, cp.array(from_save[key], dtype=self.dtype, copy=False))
                    else:
                        setattr(self, key, from_save[key])

        # Overwrite stats
        self.stats = from_save["stats"]

    # Visualization helper functions.
    @staticmethod
    def _compute_limits(source, epsilon=0, limit_padding=0.1):
        """
        Returns the rectangular region which crops around non-zero pixels in the
        ``source`` image. See :meth:`plot_farfield()`.
        """
        limits = []
        binary = (source > epsilon) & np.logical_not(np.isnan(source))

        # Generated limits on each axis.
        for a in [0, 1]:
            if np.sum(binary) == 0:
                limits.append((0, source.shape[1-a]-1))
            else:
                # Collapse the other axis and find the range.
                collapsed = np.where(np.any(binary, axis=a))
                limit = np.array([np.amin(collapsed), np.amax(collapsed)])

                # Add padding.
                padding = int(np.diff(limit) * limit_padding)+1
                limit += np.array([-padding, padding+1])

                # Check limits and store.
                limit = np.clip(limit, 0, source.shape[1-a]-1)
                limits.append(tuple(limit))

        return limits

    def plot_nearfield(self, title="", padded=False,
                       figsize=(8,4), cbar=False):
        """
        Plots the amplitude (left) and phase (right) of the nearfield (plane of the SLM).
        The amplitude is assumed (whether uniform, or experimentally computed) while the
        phase is the result of optimization.

        Parameters
        ----------
        title : str
            Title of the plots.
        padded : bool
            If ``True``, shows the full computational nearfield of shape :attr:`shape`.
            Otherwise, shows the region at the center of the computational space of
            size :attr:`slm_shape` corresponding to the unpadded SLM.
        figsize : tuple
            Size of the plot.
        cbar : bool
            Whether to add colorbars to the plots. Defaults to ``False``.
        """
        fig, axs = plt.subplots(1, 2, figsize=figsize)

        try:
            if isinstance(self.amp, float):
                amp = self.amp
            else:
                amp = self.amp.get()
            phase = self.phase.get()
        except:
            amp = self.amp
            phase = self.phase

        if isinstance(amp, float):
            im_amp = axs[0].imshow(
                toolbox.pad(
                    amp * np.ones(self.slm_shape),
                    self.shape if padded else self.slm_shape,
                ),
                vmin=0,
                vmax=amp,
            )
        else:
            im_amp = axs[0].imshow(
                toolbox.pad(amp, self.shape if padded else self.slm_shape),
                vmin=0,
                vmax=np.amax(amp),
            )

        im_phase = axs[1].imshow(
            toolbox.pad(np.mod(phase, 2*np.pi) / np.pi, self.shape if padded else self.slm_shape),
            vmin=0,
            vmax=2,
            interpolation="none",
            cmap="twilight",
        )

        if len(title) > 0:
            title += ": "

        axs[0].set_title(title + "Amplitude")
        axs[1].set_title(title + "Phase")

        for i,ax in enumerate(axs):
            ax.set_xlabel("SLM $x$ [pix]")
            if i==0: ax.set_ylabel("SLM $y$ [pix]")

        # Add colorbars if desired
        if cbar:
            cax = make_axes_locatable(axs[0]).append_axes('right', size='5%', pad=0.05)
            fig.colorbar(im_amp, cax=cax, orientation='vertical')
            cax = make_axes_locatable(axs[1]).append_axes('right', size='5%', pad=0.05)
            fig.colorbar(im_phase, cax=cax, orientation='vertical', format = r"%1.1f$\pi$")

        fig.tight_layout()
        plt.show()

    def plot_farfield(
            self, source=None, title="", limits=None, units="knm",
            limit_padding=0.1, figsize=(8,4), cbar=False,
        ):
        """
        Plots an overview (left) and zoom (right) view of ``source``.

        Parameters
        ----------
        source : array_like OR None
            Should have shape equal to :attr:`shape`.
            If ``None``, defaults to :attr:`amp_ff`.
        title : str
            Title of the plots. If ``"phase"`` is a substring of title, then the source
            is treated as a phase.
        limits : ((float, float), (float, float)) OR None
            :math:`x` and :math:`y` limits for the zoom plot in ``"knm"`` space.
            If None, ``limits`` are autocomputed as the smallest bounds
            that show all non-zero values (plus ``limit_padding``).
            Note that autocomputing on :attr:`target` will perform well,
            as zero values are set to actually be zero. However, doing so on
            computational or experimental outputs (e.g. :attr:`amp_ff`) will likely perform
            poorly, as values in the field deviate slightly from zero and
            artificially expand the ``limits``.
        units : str
            Far-field units for plots (see
            :func:`~slmsuite.holography.toolbox.convert_blaze_vector` for options).
            If units requiring a SLM are desired, the attribute :attr:`cameraslm` must be
            filled.
        limit_padding : float
            Fraction of the width and height to expand the limits of the zoom plot by,
            only if the passed ``limits`` is ``None`` (autocompute).
        figsize : tuple
            Size of the plot.
        cbar : bool
            Whether to add colorbars to the plots. Defaults to ``False``.

        Returns
        -------
        ((float, float), (float, float))
            Used ``limits``, which may be autocomputed. Autocomputed limits are returned
            as integers.
        """
        # Parse source.
        if source is None:
            source = self.amp_ff

            if source is None:
                source = self.extract_farfield()

            if limits is None:
                if np == cp:
                    limits = self._compute_limits(self.target, limit_padding=limit_padding)
                else:
                    limits = self._compute_limits(self.target.get(), limit_padding=limit_padding)

            if len(title) == 0:
                title = "FF Amp"

        # Interpret source and convert to numpy for plotting.
        if "phase" in title.lower():
            try:
                npsource = source.get()
            except:
                npsource = source

            npsource = np.mod(npsource, 2 * np.pi)
        else:
            try:
                npsource = cp.abs(source).get()
            except:
                npsource = np.abs(source)

        # Check units
        assert units in toolbox.BLAZE_UNITS, \
            "algorithms.py: Unit {} is not recognized as a valid blaze unit.".format(units)
        assert units != "ij", \
            "algorithms.py: 'ij' is not a valid unit for plot_farfield() because of the associated rotation."

        # Determine the bounds of the zoom region, padded by limit_padding
        if limits is None:
            limits = self._compute_limits(npsource, limit_padding=limit_padding)
        # Check the limits in case the user provided them.
        for a in [0, 1]:
            limits[a] = np.clip(limits[a], 0, npsource.shape[1-a]-1)
            if np.diff(limits[a]) == 0:
                raise ValueError("algorithms.py: clipped limit has zero length.")

        # Start making the plot
        fig, axs = plt.subplots(1, 2, figsize=figsize)

        for ax in axs:
            ax.set_facecolor("#FFEEEE")

        # Plot the full target, blurred so single pixels are visible in low res
        b = 2 * int(np.amax(self.shape) / 400) + 1  # FUTURE: fix arbitrary
        npsource_blur = cv2.GaussianBlur(npsource, (b, b), 0)
        full = axs[0].imshow(
            npsource_blur,
            vmin=0, vmax=np.nanmax(npsource_blur),
            cmap=("twilight" if "phase" in title.lower() else None)
        )
        if len(title) > 0:
            title += ": "
        axs[0].set_title(title + "Full")

        # Zoom in on our spots in a second plot
        b = 2 * int(np.diff(limits[0]) / 200) + 1  # FUTURE: fix arbitrary
        zoom_data = npsource[np.ix_(np.arange(limits[1][0], limits[1][1]),
                                    np.arange(limits[0][0], limits[0][1]))]
        zoom = axs[1].imshow(
            zoom_data,
            vmin=0, vmax=np.nanmax(zoom_data),
            extent=[limits[0][0], limits[0][1],
                    limits[1][1],limits[1][0]],
            interpolation="none" if b < 2 else "gaussian",
            cmap=("twilight" if "phase" in title.lower() else None)
        )
        axs[1].set_title(title + "Zoom", color="r")
        # Red border (to match red zoom box applied below in "full" img)
        for spine in ["top", "bottom", "right", "left"]:
            axs[1].spines[spine].set_color("r")
            axs[1].spines[spine].set_linewidth(1.5)

        # Helper function: calculate extent for the given units
        try:
            slm = self.cameraslm.slm
        except:
            slm = None
            units = "knm"

        def rebase(ax, img, to_units):
            if to_units != "knm":
                ext_nm = img.get_extent()
                ext_min = np.squeeze(toolbox.convert_blaze_vector(
                    [ext_nm[0], ext_nm[-1]],
                    from_units="knm", 
                    to_units=to_units,
                    slm=slm, 
                    shape=npsource.shape
                ))
                ext_max = np.squeeze(toolbox.convert_blaze_vector(
                    [ext_nm[1], ext_nm[2]],
                    from_units="knm", 
                    to_units=to_units,
                    slm=slm, 
                    shape=npsource.shape
                ))
                img.set_extent([ext_min[0] ,ext_max[0], ext_max[1], ext_min[1]])

        # Scale and label plots depending on units
        rebase(axs[0], full, units)
        rebase(axs[1], zoom, units)

        for i, ax in enumerate(axs):
            ax.set_xlabel(toolbox.BLAZE_LABELS[units][0])
            if i == 0: ax.set_ylabel(toolbox.BLAZE_LABELS[units][1])

        # If cam_points is defined (i.e. is a FeedbackHologram or subclass),
        # plot a yellow rectangle for the extents of the camera
        if hasattr(self, "cam_points") and self.cam_points is not None:
            # Check to see if the camera extends outside of knm space.
            plot_slm_fov = (
                np.any(self.cam_points[0, :4] < 0) or
                np.any(self.cam_points[1, :4] < 0) or
                np.any(self.cam_points[0, :4] >= npsource.shape[1]) or
                np.any(self.cam_points[1, :4] >= npsource.shape[1])
            )

            # If so, plot a labeled green rectangle to show the extents of knm space.
            if plot_slm_fov:
                extent = full.get_extent()
                pix_width = (np.diff(extent[0:2])[0]) / npsource.shape[1]
                rect = plt.Rectangle(
                    np.array(extent[::2]) - pix_width/2,
                    np.diff(extent[0:2])[0],
                    np.diff(extent[2:])[0],
                    ec="g",
                    fc="none",
                )
                axs[0].add_patch(rect)
                axs[0].annotate(
                    "SLM FoV",
                    (np.mean(extent[:2]), np.max(extent[2:])),
                    c="g", size="small", ha="center", va="top"
                )

            # Convert cam_points to knm.
            if units == "knm":
                cam_points = self.cam_points
            else:
                cam_points = toolbox.convert_blaze_vector(
                    self.cam_points,
                    from_units="knm", 
                    to_units=units,
                    slm=slm, 
                    shape=npsource.shape
                )

            # Plot the labeled yellow rectangle representing the camera.
            axs[0].plot(
                cam_points[0],
                cam_points[1],
                c="y",
            )
            axs[0].annotate(
                "Camera FoV",
                (np.mean(cam_points[0, :4]), np.max(cam_points[1, :4])),
                c="y", size="small", ha="center", va="top"
            )

            # Determine sensible limits of the field of view.
            if plot_slm_fov:
                dx = np.max(cam_points[0]) - np.min(cam_points[0])
                dy = np.max(cam_points[1]) - np.min(cam_points[1])
            else:
                dx = dy = 0

            ext = full.get_extent()
            axs[0].set_xlim([
                min(ext[0], np.min(cam_points[0]) - dx/10),
                max(ext[1], np.max(cam_points[0]) + dx/10)
            ])
            axs[0].set_ylim([
                max(ext[2], np.max(cam_points[1]) + dy/10),
                min(ext[3], np.min(cam_points[1]) - dy/10)
            ])

        # Bonus: Plot a red rectangle to show the extents of the zoom region
        if np.diff(limits[0]) > 0 and np.diff(limits[1]) > 0:
            extent = zoom.get_extent()
            pix_width = (np.diff(extent[0:2])[0]) / np.diff(limits[0])
            rect = plt.Rectangle(
                tuple((np.array(extent[::2]) - pix_width/2).astype(float)),
                float(np.diff(extent[0:2])[0]),
                float(np.diff(extent[2:])[0]),
                ec="r",
                fc="none",
            )
            axs[0].add_patch(rect)
            axs[0].annotate(
                "Zoom",
                (np.mean(extent[:2]), np.min(extent[2:])),
                c="r", size="small", ha="center", va="bottom"
            )

        # Add colorbar if desired
        if cbar:
            cax = make_axes_locatable(axs[1]).append_axes('right', size='5%', pad=0.05)
            fig.colorbar(zoom, cax=cax, orientation='vertical')

        plt.tight_layout()
        plt.show()

        return limits

    def plot_stats(self, stats_dict=None, stat_groups=[], ylim=None):
        """
        Plots the statistics contained in the given dictionary.

        Parameters
        ----------
        stats_dict : dict OR None
            Stats to plot in dictionary form. If ``None``, defaults to :attr:`stats`.
        stat_groups : list of str OR None
            Which
        ylim : (int, int) OR None
            Allows the user to pass in desired y limits.
            If ``None``, the default y limits are used.
        """
        if stats_dict is None:
            stats_dict = self.stats

        _, ax = plt.subplots(1, 1, figsize=(6,4))

        stats = ["efficiency", "uniformity", "pkpk_err", "std_err"]
        markers = ["o", "o", "s", "D"]
        legendstats = ["inefficiency", "nonuniformity", "pkpk_err", "std_err"]

        niter = np.arange(0, len(stats_dict["method"]))

        if stat_groups is None or len(stat_groups) == 0:
            stat_keys = list(stats_dict["stats"].keys())
        else:
            stat_keys = [str(group) for group in stat_groups]
        dummylines_modes = []

        for ls_num, stat_key in enumerate(stat_keys):
            stat_group = stats_dict["stats"][stat_key]

            for i in range(len(stats)):
                # Invert the stats if it is efficiency or uniformity.
                y = stat_group[stats[i]]
                if i < 2:
                    y = 1 - np.array(y)

                color = "C%d"%ls_num
                line = ax.scatter(niter, y, marker=markers[i], ec=color,
                                  fc="None" if i >= 1 else color)
                ax.plot(niter, y, c=color, lw=0.5)

                if i == 0:  # Remember the solid lines for the legend.
                    line = ax.plot([],[], c=color)[0]
                    dummylines_modes.append(line)

        # Make the linestyle legend.
        # Inspired from https://stackoverflow.com/a/46214879
        dummylines_keys = []
        for i in range(len(stats)):
            dummylines_keys.append(ax.scatter([], [], marker=markers[i], ec="k",
                                              fc = "None" if i >= 1 else "k"))

        ax.set_xlabel('Iteration')
        ax.set_ylabel('Relative Metrics')
        ax.set_title(self.__class__.__name__ + ' Statistics')
        ax.set_yscale("log")
        plt.grid()
        try:    # This fails under all nan or other conditions. Fail elegantly.
            plt.tight_layout()
        except:
            pass
        if ylim is not None:
            ax.set_ylim(ylim)

        # Shade fixed_phase. FUTURE: A more general method could be written
        if "fixed_phase" in stats_dict["flags"] and any(stats_dict["flags"]["fixed_phase"]):
            fp = np.concatenate((
                stats_dict["flags"]["fixed_phase"],
                [stats_dict["flags"]["fixed_phase"][-1]]
            )) | np.concatenate((
                [stats_dict["flags"]["fixed_phase"][0]],
                stats_dict["flags"]["fixed_phase"]
            ))
            niter_fp = np.arange(0, len(stats_dict["method"]) + 1)

            ylim = ax.get_ylim()
            poly = ax.fill_between(niter_fp - .5, ylim[0], ylim[1], where=fp,
                                   alpha=0.1, color='b', zorder=-np.inf)
            ax.set_ylim(ylim)

            dummylines_keys.append(poly)
            legendstats.append("fixed_phase")

        # Make the color/linestyle legend.
        plt.legend(dummylines_modes + dummylines_keys, stat_keys + legendstats, loc="lower left")

        ax.set_xlim([-.75, len(stats_dict["method"]) - .25])

        plt.show()

        return ax

    # Other helper functions.
    @staticmethod
    def set_mempool_limit(device=0, size=None, fraction=None):
        """
        Helper function to set the `cupy memory pool size
        <https://docs.cupy.dev/en/stable/reference/generated/cupy.cuda.MemoryPool.html#cupy.cuda.MemoryPool>`_.

        Parameters
        ----------
        device : int
            Which GPU to set the limit on. Passed to :meth:`cupy.cuda.Device()`.
        size : int
            Desired number of bytes in the pool. Passed to :meth:`cupy.cuda.MemoryPool.set_limit()`.
        fraction : float
            Fraction of available memory to use. Passed to :meth:`cupy.cuda.MemoryPool.set_limit()`.
        """
        if cp == np:
            raise ValueError("algorithms.py: Cannot set mempool for numpy. Need cupy.")

        mempool = cp.get_default_memory_pool()

        with cp.cuda.Device(device):
            mempool.set_limit(size=size, fraction=fraction)

            print(
                "cupy memory pool limit set to {:.2f} GB...".format(
                    mempool.get_limit() / (1024.0 ** 3)
                )
            )

    @staticmethod
    def get_mempool_limit(device=0):
        """
        Helper function to get the `cupy memory pool size
        <https://docs.cupy.dev/en/stable/reference/generated/cupy.cuda.MemoryPool.html#cupy.cuda.MemoryPool>`_.

        Parameters
        ----------
        device : int
            Which GPU to set the limit on. Passed to :meth:`cupy.cuda.Device()`.

        Returns
        -------
        int
            Current memory pool limit in bytes
        """

        if cp == np:
            raise ValueError("algorithms.py: Cannot get mempool for numpy. Need cupy.")

        mempool = cp.get_default_memory_pool()

        with cp.cuda.Device(device):
            return mempool.get_limit()

    @staticmethod
    def _norm(matrix, mp=cp):
        r"""
        Computes the root of the sum of squares of the given ``matrix``. Implements:

        .. math:: \sqrt{\iint |\vec{E}|^2}

        Parameters
        ----------
        matrix : numpy.ndarray OR cupy.ndarray
            Data, potentially complex.
        mp : module
            This function is used by both :mod:`cupy` and :mod:`numpy`, so we have the option
            for either. Defaults to :mod:`cupy`.

        Returns
        -------
        float
            The result.
        """
        if mp.iscomplexobj(matrix):
            return mp.sqrt(mp.nansum(mp.square(mp.abs(matrix))))
        else:
            return mp.sqrt(mp.nansum(mp.square(matrix)))


class FeedbackHologram(Hologram):
    """
    Experimental holography aided by camera feedback.
    Contains mechanisms for hologram positioning and camera feedback aided by a
    :class:`~slmsuite.hardware.cameraslms.FourierSLM`.

    Attributes
    ----------
    cameraslm : slmsuite.hardware.cameraslms.FourierSLM OR None
        A hologram with experimental feedback needs access to an SLM and camera.
        If None, no feedback is applied (mostly defaults to :class:`Hologram`).
    cam_shape : (int, int)
        Shape of the camera in the meaning of :meth:`numpy.shape()`.
    cam_points : numpy.ndarray
        Array containing points corresponding to the corners of the camera in the SLM's k-space.
        First point is repeated at the end for easy plotting.
    target_ij :  array_like OR None
        Amplitude target in the ``"ij"`` (camera) basis. Of same ``shape`` as the camera in
        :attr:`cameraslm`.  Counterpart to :attr:`target` which is in the ``"knm"``
        (computational k-space) basis.
    img_ij, img_knm
        Cached **amplitude** feedback image in the
        ``"ij"`` (raw camera) basis or
        ``"knm"`` (transformed to computational k-space) basis.
        Measured with :meth:`.measure()`.
    """

    def __init__(self, shape, target_ij=None, cameraslm=None, **kwargs):
        """
        Initializes a hologram with camera feedback.

        Parameters
        ----------
        shape : (int, int)
            Computational shape of the SLM in :mod:`numpy` `(h, w)` form. See :meth:`.Hologram.__init__()`.
        target_ij : array_like OR None
            See :attr:`target_ij`. Should only be ``None`` if the :attr:`target`
            will be generated by other means (see :class:`SpotHologram`), so the
            user should generally provide an array.
        cameraslm : slmsuite.hardware.cameraslms.FourierSLM OR slmsuite.hardware.slms.SLM OR None
            Provides access to experimental feedback.
            If an :class:`slmsuite.hardware.slms.SLM` is passed, this is set to `None`,
            but the information contained in the SLM is passed to the superclass :class:`.Hologram`.
            See :attr:`cameraslm`.
        kwargs
            See :meth:`Hologram.__init__`.
        """
        # Use the Hologram constructor to initialize self.target with proper shape,
        # pass other arguments (esp. slm_shape).
        self.cameraslm = cameraslm
        if self.cameraslm is not None:
            # Determine camera size in SLM-space.
            try:
                amp = self.cameraslm.slm.measured_amplitude
                slm_shape = self.cameraslm.slm.shape
            except:
                # See if an SLM was passed.
                try:
                    amp = self.cameraslm.measured_amplitude
                    slm_shape = self.cameraslm.shape

                    # We don't have access to all the calibration stuff, so don't
                    # confuse the rest of the init/etc.
                    self.cameraslm = None
                except:
                    raise ValueError("Expected a CameraSLM or SLM to be passed to cameraslm.")

        else:
            amp = kwargs.pop("amp", None)
            slm_shape = None

        if not "slm_shape" in kwargs:
            kwargs["slm_shape"] = slm_shape

        super().__init__(target=shape, amp=amp, **kwargs)

        self.img_ij = None
        self.img_knm = None
        if target_ij is None:
            self.target_ij = None
        else:
            self.target_ij = target_ij.astype(self.dtype)

        if (
            self.cameraslm is not None
            and self.cameraslm.fourier_calibration is not None
        ):
            # Generate a list of the corners of the camera, for plotting.
            cam_shape = self.cameraslm.cam.shape

            ll = [0, 0]
            lr = [0, cam_shape[0] - 1]
            ur = [cam_shape[1] - 1, cam_shape[0] - 1]
            ul = [cam_shape[1] - 1, 0]

            points_ij = toolbox.format_2vectors(np.vstack((ll, lr, ur, ul, ll)).T)
            points_kxy = self.cameraslm.ijcam_to_kxyslm(points_ij)
            self.cam_points = toolbox.convert_blaze_vector(
                points_kxy, "kxy", "knm", slm=self.cameraslm.slm, shape=self.shape
            )
            self.cam_shape = cam_shape

            # Transform the target, if it is provided.
            if target_ij is not None:
                self.update_target(target_ij, reset_weights=True)

        else:
            self.cam_points = None
            self.cam_shape = None

    def ijcam_to_knmslm(self, img, out=None, blur_ij=None, order=3):
        """
        Convert an image in the camera domain to computational SLM k-space using, in part, the
        affine transformation stored in a cameraslm's Fourier calibration.

        Note
        ~~~~
        This includes two transformations:

         - The affine transformation ``"ij"`` -> ``"kxy"`` (camera pixels to normalized k-space).
         - The scaling ``"kxy"`` -> ``"knm"`` (normalized k-space to computational k-space pixels).

        Parameters
        ----------
        img : numpy.ndarray OR cupy.ndarray
            Image to transform. This should be the same shape as images returned by the camera.
        out : numpy.ndarray OR cupy.ndarray OR None
            If ``out`` is not ``None``, this array will be used to write the memory in-place.
        blur_ij : int OR None
            Applies a ``blur_ij`` pixel-width Gaussian blur to ``img``.
            If ``None``, defaults to the ``"blur_ij"`` flag if present; otherwise zero.
        order : int
            Order of interpolation used for transformation. Defaults to 3 (cubic).

        Returns
        -------
        numpy.ndarray OR cupy.ndarray
            Image transformed into ``"knm"`` space.
        """
        assert self.cameraslm is not None
        assert self.cameraslm.fourier_calibration is not None

        # First transformation.
        conversion = (
            toolbox.convert_blaze_vector(
                (1, 1), "knm", "kxy", slm=self.cameraslm.slm, shape=self.shape
            ) -
            toolbox.convert_blaze_vector(
                (0, 0), "knm", "kxy", slm=self.cameraslm.slm, shape=self.shape
            )
        )
        M1 = np.diag(np.squeeze(conversion))
        b1 = np.matmul(M1, -toolbox.format_2vectors(np.flip(np.squeeze(self.shape)) / 2))

        # Second transformation.
        M2 = self.cameraslm.fourier_calibration["M"]
        b2 = self.cameraslm.fourier_calibration["b"] - np.matmul(
            M2, self.cameraslm.fourier_calibration["a"]
        )

        # Composite transformation (along with xy -> yx).
        M = cp.array(np.flip(np.flip(np.matmul(M2, M1), axis=0), axis=1))
        b = cp.array(np.flip(np.matmul(M2, b1) + b2))

        # See if the user wants to blur.
        if blur_ij is None:
            if "blur_ij" in self.flags:
                blur_ij = self.flags["blur_ij"]
            else:
                blur_ij = 0

        # FUTURE: use cp_gaussian_filter (faster?); was having trouble with cp_gaussian_filter.
        if blur_ij > 0:
            img = sp_gaussian_filter(img, (blur_ij, blur_ij), output=img, truncate=2)

        cp_img = cp.array(img, dtype=self.dtype)
        cp.abs(cp_img, out=cp_img)

        # Perform affine.
        target = cp_affine_transform(
            input=cp_img,
            matrix=M,
            offset=b,
            output_shape=self.shape,
            order=order,
            output=out,
            mode="constant",
            cval=0,
        )

        # Filter the image. FUTURE: fix.
        # target = cp_gaussian_filter1d(target, blur, axis=0, output=target, truncate=2)
        # target = cp_gaussian_filter1d(target, blur, axis=1, output=target, truncate=2)

        target = cp.abs(target, out=target)
        norm = Hologram._norm(target)
        target *= 1 / norm

        assert norm != 0, "FeedbackHologram.ijcam_to_knmslm(): target_ij is out of range of knm space. Check transformations."

        return target

    def update_target(self, new_target, reset_weights=False, plot=False):
        # Transformation order of zero to prevent nan-blurring in MRAF cases.
        self.ijcam_to_knmslm(new_target, out=self.target, order=0)

        if reset_weights:
            self.reset_weights()

        if plot:
            self.plot_farfield(self.target)

    def measure(self, basis="ij"):
        """
        Method to request a measurement to occur. If :attr:`img_ij` is ``None``,
        then a new image will be grabbed from the camera (this is done automatically in
        algorithms).

        Parameters
        ----------
        basis : str
            The cached image to be sure to fill with new data.
            Can be ``"ij"`` or ``"knm"``.

             - If ``"knm"``, then :attr:`img_ij` and :attr:`img_knm` are filled.
             - If ``"ij"``, then :attr:`img_ij` is filled, and :attr:`img_knm` is ignored.

            This is useful to avoid (expensive) transformation from the ``"ij"`` to the
            ``"knm"`` basis if :attr:`img_knm` is not needed.
        """
        if self.img_ij is None:
            self.cameraslm.slm.write(self.extract_phase(), settle=True)
            self.cameraslm.cam.flush()
            self.img_ij = np.array(self.cameraslm.cam.get_image(), copy=False, dtype=self.dtype)

            if basis == "knm":  # Compute the knm basis image.
                self.img_knm = self.ijcam_to_knmslm(self.img_ij, out=self.img_knm)
                cp.sqrt(self.img_knm, out=self.img_knm)
            else:  # The old image is outdated, erase it. FUTURE: memory concerns?
                self.img_knm = None

            self.img_ij = np.sqrt(
                self.img_ij
            )  # Don't load to the GPU if not necessary.
        elif basis == "knm":
            if self.img_knm is None:
                self.img_knm = self.ijcam_to_knmslm(np.square(self.img_ij), out=self.img_knm)
                cp.sqrt(self.img_knm, out=self.img_knm)

    def refine_offset(self, img, basis="kxy"):
        """
        **(NotImplemented)**
        Hones the position of the produced image to the desired target image to compensate for
        Fourier calibration imperfections. Works either by moving the desired camera
        target to align where the image ended up (``basis="ij"``) or by moving
        the :math:`k`-space image to target the desired camera target
        (``basis="knm"``/``basis="kxy"``). This should be run at the user's request
        inbetween :meth:`optimize` iterations.

        Parameters
        ----------
        img : numpy.ndarray
            Image measured by the camera.
        basis : str
            The correction can be in any of the following bases:
            - ``"ij"`` changes the pixel that the spot is expected at,
            - ``"kxy"`` or ``"knm"`` changes the k-vector which the SLM targets.
            Defaults to ``"kxy"`` if ``None``.

        Returns
        -------
        numpy.ndarray
            Euclidean pixel error in the ``"ij"`` basis for each spot.
        """

        raise NotImplementedError()

    def _update_weights(self):
        """
        Change :attr:`weights` to optimize towards the :attr:`target` using feedback from
        :attr:`amp_ff`, the computed farfield amplitude. This function also updates stats.
        """
        feedback = self.flags["feedback"]

        if feedback == "computational":
            self._update_weights_generic(self.weights, self.amp_ff, self.target)
        elif feedback == "experimental":
            self.measure("knm")  # Make sure data is there.
            self._update_weights_generic(self.weights, self.img_knm, self.target)

    def _calculate_stats_experimental(self, stats, stat_groups=[]):
        """
        Wrapped by :meth:`FeedbackHologram.update_stats()`.
        """
        if "experimental_knm" in stat_groups:
            self.measure("knm")  # Make sure data is there.

            stats["experimental_knm"] = self._calculate_stats(
                self.img_knm,
                self.target,
                efficiency_compensation=True,
                raw="raw_stats" in self.flags and self.flags["raw_stats"]
            )
        if "experimental_ij" in stat_groups or "experimental" in stat_groups:
            self.measure("ij")  # Make sure data is there.

            stats["experimental_ij"] = self._calculate_stats(
                self.img_ij.astype(self.dtype),
                self.target_ij,
                mp=np,
                efficiency_compensation=True,
                raw="raw_stats" in self.flags and self.flags["raw_stats"]
            )

    def update_stats(self, stat_groups=[]):
        """
        Calculate statistics corresponding to the desired ``stat_groups``.

        Parameters
        ----------
        stat_groups : list of str
            Which groups or types of statistics to analyze.
        """
        stats = {}

        self._calculate_stats_computational(stats, stat_groups)
        self._calculate_stats_experimental(stats, stat_groups)

        self._update_stats_dictionary(stats)


class SpotHologram(FeedbackHologram):
    """
    Holography optimized for the generation of optical focal arrays.

    Is a subclass of :class:`FeedbackHologram`, but falls back to non-camera-feedback
    routines if :attr:`cameraslm` is not passed.

    Tip
    ~~~
    Quality of life features to generate noise regions for mixed region amplitude
    freedom (MRAF) algorithms are supported. Specifically, set ``null_region``
    parameters to help specify where the noise region is not.

    Attributes
    ----------
    spot_knm, spot_kxy, spot_ij : array_like of float OR None
        Stored vectors with shape ``(2, N)`` in the style of
        :meth:`~slmsuite.holography.toolbox.format_2vectors()`.
        These vectors are floats.
        The subscript refers to the basis of the vectors, the transformations between
        which are autocomputed.
        If necessary transformations do not exist, :attr:`spot_ij` is set to ``None``.
    spot_knm_rounded : array_like of int
        :attr:`spot_knm` rounded to nearest integers (indices).
        These vectors are integers.
        This is necessary because
        GS algorithms operate on a pixel grid, and the target for each spot in a
        :class:`SpotHologram` is a single pixel (index).
    spot_kxy_rounded, spot_ij_rounded : array_like of float
        Once :attr:`spot_knm_rounded` is rounded, the original :attr:`spot_kxy`
        and :attr:`spot_ij` are no longer accurate. Transformations are again used
        to backcompute the positions in the ``"ij"`` and ``"kxy"`` bases corresponding
        to the true computational location of a given spot.
        These vectors are floats.
    spot_amp : array_like of float
        The target amplitude for each spot.
        Must have length corresponding to the number of spots.
        For instance, the user can request dimmer or brighter spots.
    external_spot_amp : array_like of float
        When using ``"external_spot"`` feedback or the ``"external_spot"`` stat group,
        the user must supply external data. This data is transferred through this
        attribute. For iterative feedback, have the ``callback()`` function set
        :attr:`external_spot_amp` dynamically. By default, this variable is set to even
        distribution of amplitude.
    spot_integration_width_knm : int
        For spot-specific feedback methods, better SNR is achieved when integrating over
        many farfield pixels. This variable stores the width of the integration region
        in ``"knm"`` (farfield) space.
    spot_integration_width_ij : int
        For spot-specific feedback methods, better SNR is achieved when integrating over
        many camera pixels. This variable stores the width of the integration region
        in ``"ij"`` (camera) space.
    null_knm : array_like of float OR None
        In addition to points where power is desired, :class:`SpotHologram` is equipped
        with quality of life features to select points where power is undesired. These
        points are stored in :attr:`null_knm` with shape ``(2, M)`` in the style of
        :meth:`~slmsuite.holography.toolbox.format_2vectors()`. A region around these
        points is set to zero (null) and not allowed to participate in the noise region.
    null_radius_knm : float
        The radius in ``"knm"`` space around the points :attr:`null_knm` to zero or null
        (prevent from participating in the ``nan`` noise region).
        This is useful to prevent power being deflected to very high orders,
        which are unlikely to be properly represented in practice on a physical SLM.
    null_region_knm : array_like of bool OR ``None``
        Array of shape :attr:`shape`. Where ``True``, sets the background to zero
        instead of nan. If ``None``, has no effect.
    subpixel_beamradius_knm : float
        The radius in knm space corresponding to the beamradius of the Gaussian spot
        which is targeted when ``subpixel`` features are enabled.
        In the future, a non-Gaussian kernel might be used instead.
        This radius is computed based upon the stored amplitude in the SLM (if passed).
        This is an experimental feature and should be used with caution.
    """

    def __init__(
        self,
        shape,
        spot_vectors,
        basis="knm",
        spot_amp=None,
        cameraslm=None,
        null_vectors=None,
        null_radius=None,
        null_region=None,
        null_region_radius_frac=None,
        subpixel=False,
        **kwargs
    ):
        """
        Initializes a :class:`SpotHologram` targeting given spots at ``spot_vectors``.

        Parameters
        ----------
        shape : (int, int)
            Computational shape of the SLM. See :meth:`.Hologram.__init__()`.
        spot_vectors : array_like
            Spot position vectors with shape ``(2, N)`` in the style of
            :meth:`~slmsuite.holography.toolbox.format_2vectors()`.
        basis : str
            The spots can be in any of the following bases:

            - ``"ij"`` for camera coordinates (pixels),
            - ``"kxy"`` for centered normalized SLM k-space (radians).
            - ``"knm"`` for computational SLM k-space (pixels).

            Defaults to ``"knm"`` if ``None``.
        spot_amp : array_like OR None
            The amplitude to target for each spot.
            See :attr:`SpotHologram.spot_amp`.
            If ``None``, all spots are assumed to have the same amplitude.
            Normalization is performed automatically; the user is not required to
            normalize.
        cameraslm : slmsuite.hardware.cameraslms.FourierSLM OR None
            If the ``"ij"`` basis is chosen, and/or if the user wants to make use of camera
            feedback, a cameraslm must be provided.
        null_vectors : array_like OR None
            Null position vectors with shape ``(2, N)`` in the style of
            :meth:`~slmsuite.holography.toolbox.format_2vectors()`.
            MRAF methods are forced zero around these points.
        null_radius : float OR None
            Radius to null around in the given ``basis``.
            Note that basis conversions are imperfect for anisotropic basis
            transformations. The radius will always be set to be circular in ``"knm"``
            space, and will attempt to match to the closest circle
            to the (potentially elliptical) projection into ``"knm"`` from the given ``basis``.
        null_region : array_like OR None
            Array of shape :attr:`shape`. Where ``True``, sets the background to zero
            instead of nan. If ``None``, has no effect.
        null_region_radius_frac : float OR None
            Helper function to set the ``null_region`` to zero for Fourier space radius fractions above
            ``null_region_radius_frac``. This is useful to prevent power being deflected
            to very high orders, which are unlikely to be properly represented in
            practice on a physical SLM.
        subpixel : bool
            If enabled, the :attr:`target` is set to a series of Gaussian spots with
            radius :attr:`subpixel_beamradius_knm` instead of a series of single pixel
            spots (the default for :class:`SpotHologram`). The major benefit here is:

            - **Greater resolution with limited padding.** With 2-3 orders of padding,
              the farfield has sufficient resolution to render a Gaussian positioned
              **inbetween** farfield pixels, allowing for greater resolutions without
              having to pad further. This is especially important when operating at the
              memory limits of a system.

            Defaults to ``False``. This is an experimental feature and should be used
            with caution. Currently, there are issues with the initial phase causing
            some spots to be permanently attenuated.
        **kwargs
            Passed to :meth:`.FeedbackHologram.__init__()`.
        """
        # Parse vectors.
        vectors = toolbox.format_2vectors(spot_vectors)

        if spot_amp is not None:
            assert np.shape(vectors)[1] == len(spot_amp.ravel()), \
                "spot_amp must have the same length as the provided spots."

        # Parse null_vectors
        if null_vectors is not None:
            null_vectors = toolbox.format_2vectors(null_vectors)
            assert np.all(np.shape(null_vectors) == np.shape(null_vectors)), \
                "spot_amp must have the same length as the provided spots."
        else:
            self.null_knm = None
            self.null_radius_knm = None
        self.null_region_knm = None

        # Interpret vectors depending upon the basis.
        if (basis is None or basis == "knm"):  # Computational Fourier space of SLM.
            self.spot_knm = vectors

            if cameraslm is not None:
                self.spot_kxy = toolbox.convert_blaze_vector(
                    self.spot_knm, "knm", "kxy", cameraslm.slm, shape
                )

                if cameraslm.fourier_calibration is not None:
                    self.spot_ij = cameraslm.kxyslm_to_ijcam(self.spot_kxy)
                else:
                    self.spot_ij = None
            else:
                self.spot_kxy = None
                self.spot_ij = None

            # Handle null parameters.
            self.null_knm = null_vectors
            self.null_radius_knm = null_radius
            self.null_region_knm = null_region
        elif basis == "kxy":                    # Normalized units.
            assert cameraslm is not None, "We need a cameraslm to interpret kxy."

            self.spot_kxy = vectors

            if hasattr(cameraslm, "fourier_calibration"):
                if cameraslm.fourier_calibration is not None:
                    self.spot_ij = cameraslm.kxyslm_to_ijcam(vectors)
                    # This is okay for non-feedback GS, so we don't error.
            else:
                self.spot_ij = None

            self.spot_knm = toolbox.convert_blaze_vector(
                self.spot_kxy, "kxy", "knm", cameraslm.slm, shape
            )
        elif basis == "ij":                     # Pixel on the camera.
            assert cameraslm is not None, "We need an cameraslm to interpret ij."
            assert cameraslm.fourier_calibration is not None, (
                "We need an cameraslm with "
                "fourier-calibrated kxyslm_to_ijcam and ijcam_to_kxyslm transforms "
                "to interpret ij."
            )

            self.spot_ij = vectors
            self.spot_kxy = cameraslm.ijcam_to_kxyslm(vectors)
            self.spot_knm = toolbox.convert_blaze_vector(
                vectors, "ij", "knm", cameraslm, shape
            )
        else:
            raise Exception("algorithms.py: Unrecognized basis for spots '{}'.".format(basis))

        # Handle null conversions in the ij or kxy cases.
        if basis == "ij" or basis == "kxy":
            if null_vectors is not None:
                # Convert the null vectors.
                self.null_knm = toolbox.convert_blaze_vector(
                    null_vectors, basis, "knm", cameraslm, shape
                )

                # Convert the null radius.
                if null_radius is not None:
                    self.null_radius_knm = toolbox.convert_blaze_radius(
                        null_radius, basis, "knm", cameraslm, shape
                    )
                else:
                    self.null_radius_knm = None
            else:
                self.null_knm = None
                self.null_radius_knm = None

            self.null_region_knm = null_region

        # Generate point spread functions (psf) for the knm and ij bases
        if cameraslm is not None:
            psf_kxy = cameraslm.slm.spot_radius_kxy()
            psf_knm = toolbox.convert_blaze_radius(psf_kxy, "kxy", "knm", cameraslm.slm, shape)
            psf_ij = toolbox.convert_blaze_radius(psf_kxy, "kxy", "ij", cameraslm, shape)
        else:
            psf_knm = 0
            psf_ij = np.nan

        if np.isnan(psf_knm):   psf_knm = 0
        if np.isnan(psf_ij):    psf_ij = 0

        if subpixel:
            warnings.warn(
                "algorithms.py: subpixel spot sampling is an experimental feature "
                "and should be used with caution."
            )
            if psf_knm > .5:
                self.subpixel_beamradius_knm = psf_knm
            else:
                raise ValueError(
                    "algorithms.py: nearfield amplitude is not sufficiently padded to have "
                    "appreciable size in the farfield. Consider padding more to use subpixel "
                    "features."
                )
        else:
            self.subpixel_beamradius_knm = None

        # Use semi-arbitrary values to determine integration widths. The default width is:
        #  - six times the psf,
        #  - but then clipped to be:
        #    + larger than 3 and
        #    + smaller than the minimum inf-norm distance between spots divided by 1.5
        #      (divided by 1 would correspond to the largest non-overlapping integration
        #      regions; 1.5 gives comfortable padding)
        #  - and finally forced to be an odd integer.
        min_psf = 3

        dist_knm = np.max([toolbox.smallest_distance(self.spot_knm) / 1.5, min_psf])
        self.spot_integration_width_knm = np.clip(6 * psf_knm, min_psf, dist_knm)
        self.spot_integration_width_knm = int(2 * np.floor(self.spot_integration_width_knm / 2) + 1)

        if self.spot_ij is not None:
            dist_ij = np.max([toolbox.smallest_distance(self.spot_ij) / 1.5, min_psf])
            self.spot_integration_width_ij = np.clip(6 * psf_ij, 3, dist_ij)
            self.spot_integration_width_ij =  int(2 * np.floor(self.spot_integration_width_ij / 2) + 1)
        else:
            self.spot_integration_width_ij = None

        # Check to make sure spots are within relevant camera and SLM shapes.
        if (
            np.any(self.spot_knm[0] < self.spot_integration_width_knm / 2) or
            np.any(self.spot_knm[1] < self.spot_integration_width_knm / 2) or
            np.any(self.spot_knm[0] >= shape[1] - self.spot_integration_width_knm / 2) or
            np.any(self.spot_knm[1] >= shape[0] - self.spot_integration_width_knm / 2)
        ):
            raise ValueError(
                "Spots outside SLM computational space bounds!\nSpots:\n{}\nBounds: {}".format(
                    self.spot_knm, shape
                )
            )

        if self.spot_ij is not None:
            cam_shape = cameraslm.cam.shape

            if (
                np.any(self.spot_ij[0] < self.spot_integration_width_ij / 2) or
                np.any(self.spot_ij[1] < self.spot_integration_width_ij / 2) or
                np.any(self.spot_ij[0] >= cam_shape[1] - self.spot_integration_width_ij / 2) or
                np.any(self.spot_ij[1] >= cam_shape[0] - self.spot_integration_width_ij / 2)
            ):
                raise ValueError(
                    "Spots outside camera bounds!\nSpots:\n{}\nBounds: {}".format(
                        self.spot_ij, cam_shape
                    )
                )

        # Parse spot_amp.
        if spot_amp is None:
            self.spot_amp = np.full(len(vectors[0]), 1.0 / np.sqrt(len(vectors[0])))
        else:
            self.spot_amp = np.ravel(spot_amp)

        # Set the external amp variable to be perfect by default.
        self.external_spot_amp = np.copy(self.spot_amp)

        # Decide the null_radius (if necessary)
        if self.null_knm is not None:
            if self.null_radius_knm is None:
                all_spots = np.hstack((self.null_knm, self.spot_knm))
                self.null_radius_knm = toolbox.smallest_distance(all_spots) / 4

            self.null_radius_knm = int(np.ceil(self.null_radius_knm))

        # Initialize target/etc.
        super().__init__(shape, target_ij=None, cameraslm=cameraslm, **kwargs)

        # Parse null_region after __init__
        if basis == "ij" and null_region is not None:
            # Transformation order of zero to prevent nan-blurring in MRAF cases.
            self.null_region_knm = self.ijcam_to_knmslm(null_region, out=self.null_region_knm, order=0) != 0

        # If we have an input for null_region_radius_frac, then force the null region to
        # exclude higher order k-vectors according to the desired exclusion fraction.
        if null_region_radius_frac is not None:
            # Build up the null region pattern if we have not already done the transform above.
            if self.null_region_knm is None:
                self.null_region_knm = cp.zeros(self.shape, dtype=bool)

            # Make a circle, outside of which the null_region is active.
            xl = cp.linspace(-1, 1, self.null_region_knm.shape[0])
            yl = cp.linspace(-1, 1, self.null_region_knm.shape[1])
            (xg, yg) = cp.meshgrid(xl, yl)
            mask = cp.square(xg) + cp.square(yg) > null_region_radius_frac ** 2
            self.null_region_knm[mask] = True

        # Fill the target with data.
        self.update_target(reset_weights=True)

    def __len__(self):
        """
        Overloads len() to return the number of spots in this :class:`SpotHologram`.

        Returns
        -------
        int
            The length of :attr:`spot_amp`.
        """
        return self.spot_knm.shape[1]

    @staticmethod
    def make_rectangular_array(
        shape,
        array_shape,
        array_pitch,
        array_center=None,
        basis="knm",
        orientation_check=False,
        **kwargs
    ):
        """
        Helper function to initialize a rectangular 2D array of spots, with certain size and pitch.

        Note
        ~~~~
        The array can be in SLM k-space coordinates or in camera pixel coordinates, depending upon
        the choice of ``basis``. For the ``"ij"`` basis, ``cameraslm`` must be included as one
        of the ``kwargs``. See :meth:`__init__()` for more ``basis`` information.

        Important
        ~~~~~~~~~
        Spot positions will be rounded to the grid of computational k-space ``"knm"``,
        to create the target image (of finite size) that algorithms optimize towards.
        Choose ``array_pitch`` and ``array_center`` carefully to avoid undesired pitch
        non-uniformity caused by this rounding.

        Parameters
        ----------
        shape : (int, int)
            Computational shape of the SLM in :mod:`numpy` `(h, w)` form. See :meth:`.SpotHologram.__init__()`.
        array_shape : (int, int) OR int
            The size of the rectangular array in number of spots ``(NX, NY)``.
            If a scalar N is given, assume ``(N, N)``.
        array_pitch : (float, float) OR float
            The spacing between spots in the x and y directions ``(pitchx, pitchy)``.
            If a single pitch is given, assume ``(pitch, pitch)``.
        array_center : (float, float) OR None
            The shift of the center of the spot array from the zeroth order.
            Uses ``(x, y)`` form in the chosen basis.
            If ``None``, defaults to the position of the zeroth order, converted into the
            relevant basis:

             - If ``"knm"``, this is ``(shape[1], shape[0])/2``.
             - If ``"kxy"``, this is ``(0,0)``.
             - If ``"ij"``, this is the pixel position of the zeroth order on the
               camera (calculated via Fourier calibration).

        basis : str
            See :meth:`__init__()`.
        orientation_check : bool
            Whether to delete the last two points to check for parity.
        **kwargs
            Any other arguments are passed to :meth:`__init__()`.
        """
        # Parse size and pitch.
        if isinstance(array_shape, REAL_TYPES):
            array_shape = (int(array_shape), int(array_shape))
        if isinstance(array_pitch, REAL_TYPES):
            array_pitch = (array_pitch, array_pitch)

        # Determine array_center default.
        if array_center is None:
            if basis == "knm":
                array_center = (shape[1] / 2.0, shape[0] / 2.0)
            elif basis == "kxy":
                array_center = (0, 0)
            elif basis == "ij":
                assert "cameraslm" in kwargs, "We need an cameraslm to interpret ij."
                cameraslm = kwargs["cameraslm"]
                assert cameraslm is not None, "We need an cameraslm to interpret ij."
                assert cameraslm.fourier_calibration is not None, (
                    "We need an cameraslm with "
                    "fourier-calibrated kxyslm_to_ijcam and ijcam_to_kxyslm transforms "
                    "to interpret ij."
                )

                array_center = toolbox.convert_blaze_vector(
                    (0, 0), "kxy", "ij", cameraslm
                )

        # Make the grid edges.
        x_edge = (np.arange(array_shape[0]) - (array_shape[0] - 1) / 2.0)
        x_edge = x_edge * array_pitch[0] + array_center[0]
        y_edge = (np.arange(array_shape[1]) - (array_shape[1] - 1) / 2.0)
        y_edge = y_edge * array_pitch[1] + array_center[1]

        # Make the grid lists.
        x_grid, y_grid = np.meshgrid(x_edge, y_edge, sparse=False, indexing="xy")
        x_list, y_list = x_grid.ravel(), y_grid.ravel()

        # Delete the last two points if desired and valid.
        if orientation_check and len(x_list) > 2:
            x_list = x_list[:-2]
            y_list = y_list[:-2]

        vectors = np.vstack((x_list, y_list))

        # Return a new SpotHologram.
        return SpotHologram(shape, vectors, basis=basis, spot_amp=None, **kwargs)

    def _update_target_spots(self, reset_weights=False, plot=False):
        """
        Wrapped by :meth:`SpotHologram.update_target()`.
        """
        # Round the spot points to the nearest integer coordinates in knm space.
        if self.subpixel_beamradius_knm is None:
            self.spot_knm_rounded = np.around(self.spot_knm).astype(int)
        else:
            # Don't round if we're doing subpixel stuff.
            self.spot_knm_rounded = self.spot_knm

        # Convert these to the other coordinate systems if possible.
        if self.cameraslm is not None:
            self.spot_kxy_rounded = toolbox.convert_blaze_vector(
                self.spot_knm_rounded,
                "knm",
                "kxy",
                self.cameraslm.slm,
                self.shape,
            )

            if self.cameraslm.fourier_calibration is not None:
                self.spot_ij_rounded = self.cameraslm.kxyslm_to_ijcam(
                    self.spot_kxy_rounded
                )
            else:
                self.spot_ij_rounded = None
        else:
            self.spot_kxy_rounded = None
            self.spot_ij_rounded = None

        # Erase previous target in-place.
        if self.null_knm is None:
            self.target.fill(0)
        else:
            # By default, everywhere is "amplitude free", denoted by nan.
            self.target.fill(np.nan)

            # Now we start setting areas where null is desired. First, zero the blanket region.
            if self.null_region_knm is not None:
                self.target[self.null_region_knm] = 0

            # Second, zero the regions around the "null points".
            if self.null_knm is not None:
                all_spots = np.hstack((self.null_knm, self.spot_knm))
                w = int(2*self.null_radius_knm + 1)

                for ii in range(all_spots.shape[1]):
                    toolbox.imprint(
                        self.target,
                        (np.around(all_spots[0, ii]), w, np.around(all_spots[1, ii]), w),
                        0,
                        centered=True,
                        circular=True
                    )

        # Set all the target pixels to the appropriate amplitude.
        if self.subpixel_beamradius_knm is None:
            self.target[
                self.spot_knm_rounded[1, :], self.spot_knm_rounded[0, :]
            ] = self.spot_amp
        else:   # Otherwise, make a target consisting of imprinted gaussians (subpixel enabled)
            grid = np.meshgrid(np.arange(self.target.shape[1]), np.arange(self.target.shape[0]))

            for spot_idx in range(len(self)):
                toolbox.imprint(
                    matrix=self.target,
                    window=(
                        self.spot_knm[0, spot_idx], 4*np.ceil(self.subpixel_beamradius_knm)+1,
                        self.spot_knm[1, spot_idx], 4*np.ceil(self.subpixel_beamradius_knm)+1
                    ),
                    function=gaussian2d,
                    grid=grid,
                    imprint_operation="replace",
                    centered=True,
                    circular=True,
                    clip=True,                      # End of imprint parameters
                    x0=self.spot_knm[0, spot_idx],  # Start of gaussian2d parameters
                    y0=self.spot_knm[1, spot_idx],
                    a=self.spot_amp[spot_idx],
                    c=0,
                    wx=self.subpixel_beamradius_knm,
                    wy=self.subpixel_beamradius_knm,
                )

        self.target /= Hologram._norm(self.target)

        if reset_weights:
            self.reset_weights()

        if plot:
            self.plot_farfield(self.target)

    def update_target(self, reset_weights=False, plot=False):
        """
        From the spot locations stored in :attr:`spot_knm`, update the target pattern.

        Note
        ~~~~
        If there's a cameraslm, updates the :attr:`spot_ij_rounded` attribute
        corresponding to where pixels in the k-space where actually placed (due to rounding
        to integers, stored in :attr:`spot_knm_rounded`), rather the
        idealized floats :attr:`spot_knm`.

        Note
        ~~~~
        The :attr:`target` and :attr:`weights` matrices are modified in-place for speed,
        unlike :class:`.Hologram` or :class:`.FeedbackHologram` which make new matrices.
        This is because spot positions are expected to be corrected using :meth:`correct_spots()`.

        Parameters
        ----------
        reset_weights : bool
            Whether to rest the :attr:`weights` to this new :attr:`target`.
        plot : bool
            Whether to enable debug plotting to see the positions of the spots relative to the
            shape of the camera and slm.
        """
        self._update_target_spots(reset_weights=reset_weights, plot=plot)

    def refine_offset(self, img=None, basis="kxy", force_affine=True, plot=False):
        """
        Hones the positions of the produced spots toward the desired targets to compensate for
        Fourier calibration imperfections. Works either by moving camera integration
        regions to the positions where the spots ended up (``basis="ij"``) or by moving
        the :math:`k`-space targets to target the desired camera pixels
        (``basis="knm"``/``basis="kxy"``). This should be run at the user's request
        inbetween :meth:`optimize` iterations.

        Parameters
        ----------
        img : numpy.ndarray OR None
            Image measured by the camera. If ``None``, defaults to :attr:`img_ij` via :meth:`measure()`.
        basis : str
            The correction can be in any of the following bases:

            - ``"ij"`` changes the pixel that the spot is expected at,
            - ``"kxy"``, ``"knm"`` changes the k-vector which the SLM targets.

            Defaults to ``"kxy"``. If basis is set to ``None``, no correction is applied
            to the data in the :class:`SpotHologram`.
        force_affine : bool
            Whether to force the offset refinement to behave as an affine transformation
            between the original and refined coordinate system. This helps to tame
            outliers. Defaults to ``True``.

        plot : bool
            Enables debug plots.

        Returns
        -------
        numpy.ndarray
            Spot shift in the ``"ij"`` basis for each spot.
        """
        # If no image was provided, get one from cache.
        if img is None:
            img = self.measure(basis="ij")

        # Take regions around each point from the given image.
        regions = analysis.take(
            img, self.spot_ij, self.spot_integration_width_ij, centered=True, integrate=False
        )

        # Fast version; have to iterate for accuracy.
        shift_vectors = analysis.image_positions(regions)
        shift_vectors = np.clip(
            shift_vectors,
            -self.spot_integration_width_ij/4,
            self.spot_integration_width_ij/4
        )

        # Store the shift vector before we force_affine.
        sv1 = self.spot_ij + shift_vectors

        if force_affine:
            affine = analysis.fit_affine(self.spot_ij, self.spot_ij + shift_vectors, plot=plot)
            shift_vectors = (np.matmul(affine["M"], self.spot_ij) + affine["b"]) - self.spot_ij

        # Record the shift vector after we force_affine.
        sv2 = self.spot_ij + shift_vectors

        # Plot the above if desired.
        if plot:
            mask = analysis.take(
                img, self.spot_ij, self.spot_integration_width_ij,
                centered=True, integrate=False, return_mask=True
            )

            plt.figure(figsize=(12, 12))
            plt.imshow(img * mask)
            plt.scatter(sv1[0,:], sv1[1,:], s=200, fc="none", ec="r")
            plt.scatter(sv2[0,:], sv2[1,:], s=300, fc="none", ec="b")
            plt.show()

        # Handle the feedback applied from this refinement.
        if basis is not None:
            if (basis == "kxy" or basis == "knm"):
                # Modify k-space targets. Don't modify any camera spots.
                self.spot_kxy = self.spot_kxy - (
                    self.cameraslm.ijcam_to_kxyslm(shift_vectors) -
                    self.cameraslm.ijcam_to_kxyslm((0,0))
                )
                self.spot_knm = toolbox.convert_blaze_vector(
                    self.spot_kxy, "kxy", "knm", self.cameraslm.slm, self.shape
                )
                self.update_target(reset_weights=True)
                self.reset_phase()
            elif basis == "ij":
                # Modify camera targets. Don't modify any k-vectors.
                self.spot_ij = self.spot_ij - shift_vectors
            else:
                raise Exception("Unrecognized basis '{}'.".format(basis))

        return shift_vectors

    def _update_weights(self):
        """
        Change :attr:`weights` to optimize towards the :attr:`target` using feedback from
        :attr:`amp_ff`, the computed farfield amplitude. This function also updates stats.
        """
        feedback = self.flags["feedback"]

        # If we're doing subpixel stuff, we can't use computational feedback, upgrade to computational_spot.
        if self.subpixel_beamradius_knm is not None and feedback == "computational":
            feedback = self.flags["feedback"] = "computational_spot"

        # Weighting strategy depends on the chosen feedback method.
        if feedback == "computational":
            # Pixel-by-pixel weighting
            self._update_weights_generic(self.weights, self.amp_ff, self.target, nan_checks=True)
        else:
            # Integrate a window around each spot, with feedback from respective sources.
            if feedback == "computational_spot":
                amp_feedback = cp.sqrt(analysis.take(
                    cp.square(self.amp_ff),
                    self.spot_knm_rounded,
                    self.spot_integration_width_knm,
                    centered=True,
                    integrate=True,
                    mp=cp
                ))
            elif feedback == "experimental_spot":
                self.measure(basis="ij")

                amp_feedback = np.sqrt(analysis.take(
                    np.square(np.array(self.img_ij, copy=False, dtype=self.dtype)),
                    self.spot_ij,
                    self.spot_integration_width_ij,
                    centered=True,
                    integrate=True
                ))
            elif feedback == "external_spot":
                amp_feedback = self.external_spot_amp
            else:
                raise ValueError("algorithms.py: Feedback '{}' not recognized.".format(feedback))

            if self.subpixel_beamradius_knm is None:
                # Default mode: no subpixel stuff. We update single pixels.
                self.weights[self.spot_knm_rounded[1, :], self.spot_knm_rounded[0, :]] = (
                    self._update_weights_generic(
                        self.weights[self.spot_knm_rounded[1, :], self.spot_knm_rounded[0, :]],
                        cp.array(amp_feedback, copy=False, dtype=self.dtype),
                        self.spot_amp,
                        nan_checks=True
                    )
                )
            else:
                # Complex mode: subpixel stuff. Update Gaussian patterns.
                if hasattr(amp_feedback, "get"):
                    amp_feedback = amp_feedback.get()

                # Figure out the multiplication factors on a dummy array.
                dummy_weights = (
                    self._update_weights_generic(
                        np.ones(len(self)),
                        amp_feedback,
                        self.spot_amp,
                        mp=np
                    )
                )

                # Update each Gaussian with each respective multiplication factor.
                for spot_idx in range(len(self)):
                    window = toolbox.window_slice(
                        window=(
                            self.spot_knm[0, spot_idx], 4*np.ceil(self.subpixel_beamradius_knm)+1,
                            self.spot_knm[1, spot_idx], 4*np.ceil(self.subpixel_beamradius_knm)+1
                        ),
                        shape=None,
                        centered=True,
                        circular=True
                    )
                    self.weights[window] *= dummy_weights[spot_idx]

    def _calculate_stats_spots(self, stats, stat_groups=[]):
        """
        Wrapped by :meth:`SpotHologram.update_stats()`.
        """

        if "computational_spot" in stat_groups:
            if self.shape == self.slm_shape:
                # Spot size is one pixel wide: no integration required.
                stats["computational_spot"] = self._calculate_stats(
                    self.amp_ff[self.spot_knm_rounded[1, :], self.spot_knm_rounded[0, :]],
                    self.spot_amp,
                    efficiency_compensation=False,
                    total=cp.sum(cp.square(self.amp_ff)),
                    raw="raw_stats" in self.flags and self.flags["raw_stats"]
                )
            else:
                # Spot size is wider than a pixel: integrate a window around each spot
                if cp != np:
                    pwr_ff = cp.square(self.amp_ff)
                    pwr_feedback = analysis.take(
                        pwr_ff,
                        self.spot_knm,
                        self.spot_integration_width_knm,
                        centered=True,
                        integrate=True,
                        mp=cp
                    )

                    stats["computational_spot"] = self._calculate_stats(
                        cp.sqrt(pwr_feedback),
                        self.spot_amp,
                        mp=cp,
                        efficiency_compensation=False,
                        total=cp.sum(pwr_ff),
                        raw="raw_stats" in self.flags and self.flags["raw_stats"]
                    )
                else:
                    pwr_ff = np.square(self.amp_ff)
                    pwr_feedback = analysis.take(
                        pwr_ff,
                        self.spot_knm,
                        self.spot_integration_width_knm,
                        centered=True,
                        integrate=True
                    )

                    stats["computational_spot"] = self._calculate_stats(
                        np.sqrt(pwr_feedback),
                        self.spot_amp,
                        mp=np,
                        efficiency_compensation=False,
                        total=np.sum(pwr_ff),
                        raw="raw_stats" in self.flags and self.flags["raw_stats"]
                    )

        if "experimental_spot" in stat_groups:
            self.measure(basis="ij")

            pwr_img = np.square(self.img_ij)

            pwr_feedback = analysis.take(
                pwr_img,
                self.spot_ij,
                self.spot_integration_width_ij,
                centered=True,
                integrate=True
            )

            stats["experimental_spot"] = self._calculate_stats(
                np.sqrt(pwr_feedback),
                self.spot_amp,
                mp=np,
                efficiency_compensation=False,
                total=np.sum(pwr_img),
                raw="raw_stats" in self.flags and self.flags["raw_stats"]
            )

        if "external_spot" in stat_groups:
            pwr_feedback = np.square(np.array(self.external_spot_amp, copy=False, dtype=self.dtype))
            stats["external_spot"] = self._calculate_stats(
                np.sqrt(pwr_feedback),
                self.spot_amp,
                mp=np,
                efficiency_compensation=False,
                total=np.sum(pwr_feedback),
                raw="raw_stats" in self.flags and self.flags["raw_stats"]
            )

    def update_stats(self, stat_groups=[]):
        """
        Calculate statistics corresponding to the desired ``stat_groups``.

        Parameters
        ----------
        stat_groups : list of str
            Which groups or types of statistics to analyze.
        """
        stats = {}

        self._calculate_stats_computational(stats, stat_groups)
        self._calculate_stats_experimental(stats, stat_groups)
        self._calculate_stats_spots(stats, stat_groups)

        self._update_stats_dictionary(stats)
