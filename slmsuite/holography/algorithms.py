"""
GPU-accelerated holography algorithms.

This module is currently focused on Gerchberg-Saxton (GS) iterative Fourier transform
phase retrieval algorithms [1]_ via the :class:`~slmsuite.holography.algorithms.Hologram` class;
however, support for complex holography and other algorithms (e.g. gradient descent algorithms [2]_)
is also planned. Additionally, so-called Weighted Gerchberg-Saxton (WGS) algorithms for hologram
generation with or without closed-loop camera feedback are supported, especially for
the generation of optical focus arrays [3]_, a subset of general image formation.

Tip
~~~~~~~~
This module makes use of the GPU-accelerated computing library :mod:`cupy` [4]_.
If :mod:`cupy` is not supported, then :mod:`numpy` is used as a fallback, though 
CPU alone is significantly slower. Using :mod:`cupy` is highly encouraged.

References
----------
.. [1] R. W. Gerchberg and W. O. Saxton, "A Practical Algorithm for Determination
       of Phase from Image and Diffraction Plane Pictures," Optik 35, (1972).
.. [2] J. R. Fienup, "Phase retrieval algorithms: a comparison," Appl. Opt. 21, (1982).
.. [3] D. Kim, et al., "Large-scale uniform optical focus array generation with a
       phase spatial light modulator," Opt. Lett. 44, (2019).
.. [4] https://github.com/cupy/cupy/
"""
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm

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

class Hologram:
    r"""
    Phase retrieval methods applied to holography.
    See :meth:`.optimize()` for available methods for hologram optimization.

    Tip
    ~~~
    The Fourier domain (kxy) of an SLM with shape :attr:`slm_shape` also has the shape
    :attr:`slm_shape` under FFT. However, the extents of this domain correspond to the edges
    of the farfield (:math:`\pm\frac{\lambda}{2\Delta x}` radians, where :math:`\Delta x`
    is the SLM pixel pitch). This means that resolution of the
    relevant region at the center of the farfield can be quite poor.
    The solution is to pad the SLM shape --- increasing the width and height even
    though the extents remain the same --- and thus increase the resolution of the farfield.
    In practice, padding is accomplished by passing a :attr:`shape` or
    :attr:`target` of appropriate shape (see constructor :meth:`.__init__()` and subclasses),
    potentially with the aid of the static helper function :meth:`.calculate_padding()`.

    Note
    ~~~~
    :attr:`target`, :attr:`weights`, :attr:`phase_ff`, and :attr:`amp_ff` are all
    matrices of shape :attr:`shape`. To save memory, the matrices :attr:`phase` and :attr:`amp`
    are stored with the (smaller, but not strictly smaller) shape :attr:`slm_shape`.
    Also to save memory, :attr:`phase_ff` and :attr:`amp_ff` are set to ``None`` on construction,
    and only initialized if they need to be used. Any additions should check for ``None``.

    Tip
    ~~~
    Due to SLM inefficiency, undiffracted light will be present at the center of the :attr:`target`.
    This is called the zeroth order diffraction peak. To avoid this peak, consider shifting
    the data contained in :attr:`target` away from the center.

    Attributes
    ----------
    slm_shape : (int, int) OR None
        The shape of the device producing the hologram. This is important to record because
        certain optimizations and calibrations depend on it. If multiple of :attr:`slm_shape`,
        :attr:`phase`, or :attr:`amp` are not ``None``, the shapes must agree. If all are
        ``None``, then the shape of the :attr:`target` is used instead
        (:attr:`slm_shape` == :attr:`shape`).
    phase : numpy.ndarray OR cupy.ndarray OR None
        Near-field phase pattern to optimize.
        Initialized to with :meth:`random.default_rng().uniform()` by default (``None``).
    amp : numpy.ndarray OR cupy.ndarray OR None
        Near-field source amplitude pattern (i.e. image-space constraints).
        Uniform illumination is assumed by default (``None``).
    shape : (int, int)
        The shape of the computational space.
        This often differs from :attr:`slm_shape` due to padding.
    target : numpy.ndarray OR cupy.ndarray
        Desired far-field amplitude. The goal of optimization.
    weights : numpy.ndarray OR cupy.ndarray
        The mutable far-field amplitude used in GS.
        Starts as `target` but may be modified by weighted feedback in WGS.
    phase_ff : numpy.ndarray OR cupy.ndarray
        Algorithm-constrained far-field phase. Stored for certain computational algorithms.
        (see :meth:`~slmsuite.holography.algorithms.Hologram.GS`).
    amp_ff : numpy.ndarray OR cupy.ndarray
        Far-field (Fourier-space) amplitude. Used for comparing this, the computational
        result, with the :attr:`target`.
    dtype : type
        Datatype for stored near- and far-field arrays, which are all real.
        Follows :mod`numpy` type promotion [5]_. Complex datatypes are derived from ``dtype``:

         - ``float32`` -> ``complex64`` (assumed by default)
         - ``float64`` -> ``complex128``

        ``float16`` is *not* recommended for ``dtype`` because ``complex32`` is not
        implemented by :mod`numpy`.
    N : int
        Tracks the current iteration number.
    method : str
        Remembers the name of the last-used optimization method. The method used for each
        iteration is stored in ``stats``.
    flags : dict
        Helper flags to store custom persistent variables for optimization. Contains the 
        following keys:

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
            Same format as ``"flags"``, except with another layer of heirarchy corresponding
            to the source of the given stats. This is to differentiate standard deviations
            computed computationally and experimentally.

        See :meth:`.update_stats()` and :meth:`.plot_stats()`.

    References
    ----------
    .. [5] https://numpy.org/doc/stable/reference/routines.fft.html#type-promotion
    """
    def __init__(self, target, amp=None, phase=None, slm_shape=None, dtype=np.float32):
        r"""
        Initialize datastructures for optimization.
        Parameters additional to class attributes are described below:

        target : numpy.ndarray OR cupy.ndarray OR (int, int)
            Target to optimize to. The user can also pass a shape, and this constructor
            will create an empty target of all zeros.
            :meth:`.calculate_padding()` can be of particular help for calculating the 
            shape that will produce desired results (in terms of precision, etc).
        amp, phase : array_like OR None
            See :attr:`amp` and :attr:`phase`. :attr:`phase` should only be passed
            if the user wants to precondition the optimization. Of shape :attr:`slm_shape`.
        slm_shape : (int, int) OR slmsuite.hardware.FourierSLM OR None
            The original shape of the SLM. The user can pass a
            :class:`slmsuite.hardware.FourierSLM` instead.
            If ``None``, set to :attr:`shape`.
        dtype : type
            See :attr:`dtype`; type to use for stored arrays.
        """
        # Parse target and create shape.
        if len(target) == 2:    # (int, int) was passed.
            self.shape = target
            target = None
        else:
            self.shape = target.shape

        # Warn the user about powers of two.
        if any(np.log2(self.shape) != np.round(np.log2(self.shape))):
            print(  "Warning: Hologram target shape {} is not a power of 2; consider using " \
                    ".calculate_padding() to pad to powers of 2 and speed up " \
                    "FFT computation.".format(self.shape))

        # Initialize storage vars
        self.dtype = dtype
        self.N = 0
        self.method = ''
        self.flags = {'fixed_phase':False, 'stat_groups':[]}

        # Initialize statistics dictionary
        self.stats = {'method':[], 'flags':{}, 'stats':{}}

        # Determine the shape of the SLM
        if amp is None:     amp_shape = (np.nan, np.nan)
        else:               amp_shape = amp.shape

        if phase is None:   phase_shape = (np.nan, np.nan)
        else:               phase_shape = phase.shape

        if slm_shape is None:   slm_shape = (np.nan, np.nan)
        else:
            try:    # Check if slm_shape is a CameraSLM.
                slm_shape = slm_shape.slm.shape
            except:
                pass


        stack = np.vstack((amp_shape, phase_shape, slm_shape))

        if np.all(np.isnan(stack)):
            self.slm_shape = self.shape
        else:
            self.slm_shape = np.nanmean(stack, axis=0).astype(np.int)

            if amp is not None:         assert np.all(self.slm_shape == np.array(amp_shape))
            if phase is not None:       assert np.all(self.slm_shape == np.array(phase_shape))
            if slm_shape is not None:   assert np.all(self.slm_shape == np.array(slm_shape))

            self.slm_shape = tuple(self.slm_shape)

        # Initialize and normalize near-field amplitude
        if amp is None:     # Uniform amplitude by default (scalar)
            self.amp = 1/np.sqrt(np.prod(self.slm_shape))
        else:               # Otherwise, initialize and normalize
            self.amp = cp.array(amp, dtype=dtype)
            self.amp *= 1/Hologram.norm(self.amp)

        # Initialize near-field phase
        if phase is None:   # Random near-field phase by default
            if cp == np:    # numpy does not support dtype=
                rng = np.random.default_rng()
                self.phase = rng.uniform(-np.pi, np.pi, self.slm_shape).astype(dtype)
            else:
                # rng = cp.random.default_rng()
                self.phase = cp.random.uniform(-np.pi, np.pi, self.slm_shape, dtype=dtype)
        else:               # Initialize
            self.phase = cp.array(phase, dtype=dtype)

        # Initialize target and weights.
        self._update_target(target, reset_weights=True)

        # Initialize SLM farfield data variables.
        self.amp_ff = None
        self.phase_ff = None

    # Initialization helper
    @staticmethod
    def calculate_padding(slm_shape, padding_order=1, square_padding=True,
                            precision=np.inf, basis="kxy"):
        """
        Helper function to calculate the shape of the computational space.
        For a given shape, pads to the user's requirements. If the user chooses
        multiple requirements, the largest is selected.

        Future: Add a setting to make pad based on available memory.

        Parameters
        ----------
        slm_shape : (int, int) OR slmsuite.hardware.FourierSLM
            The original shape of the SLM. The user can pass a
            :class:`slmsuite.hardware.FourierSLM` instead, and should pass this
            when using the ``precision`` parameter.
        padding_order : int
            Scales to the ``padding_order``th closest greater power of 2.
            A ``padding_order`` of zero does nothing.
        square_padding : bool
            Returns a square shape using the largest dimension.
        precision : float
            Returns the shape that produces a computational k-space with resolution smaller
            than ``precision``.
        basis : str
            Basis for the precision. Can be ``"ij"`` (camera) or ``"kxy"`` (normalized blaze).

        Returns
        -------
        (int, int)
            Shape of the computational space which satisfies the above requirements.
        """
        try:
            cameraslm = slm_shape
            slm_shape = cameraslm.slm.shape
        except:
            cameraslm = None

        if np.isfinite(precision) and cameraslm is not None:
            dpixel = np.amin([cameraslm.slm.dx, cameraslm.slm.dy])
            fs = 1/dpixel   # Sampling frequency

            if basis == "ij":
                slm_range = np.amax(cameraslm.kxyslm_to_ijcam([fs, fs]))
                pixels = slm_range / precision
            elif basis == "kxy":
                pixels = fs / precision

            # Raise to the nearest greater power of 2.
            pixels = np.power(2, int(np.ceil(np.log2(pixels))))
            precision_shape = (pixels, pixels)
        else:
            precision_shape = slm_shape

        if padding_order > 0:
            padding_shape = (
                np.power(2, np.ceil(np.log2(slm_shape)) + (padding_order-1)).astype(np.int)
            )
        else:
            padding_shape = slm_shape

        shape = tuple(np.amax(np.vstack((precision_shape, padding_shape)), axis=0))

        if square_padding:
            largest = np.amax(shape)
            shape = (largest, largest)

        return shape

    # Core optimization function.
    def optimize(self, method="GS", maxiter=20, verbose=True,
                 callback=None, feedback='', stat_groups=[], **kwargs):
        """
        Optimizers to solve the "phase problem": approximating the near-field phase that
        transforms a known near-field source amplitude to a desired near-field
        target amplitude.
        Supported optimization methods include:

        - Gerchberg-Saxton (GS) phase retrieval.

            ``'GS'`` [1]_
              Implemented using fast Fourier transforms, potentially GPU-accelerated.
        
        - Weighted Gerchberg-Saxton (WGS) phase retrieval algorithms of various flavors.
          Improves the uniformity of GS-computed focus arrays using weighting methods and
          techniques from literature. The ``method`` keywords are:
        
            ``'WGS-Leonardo'`` [6]_
              Original WGS algorithm. Weights the target
              amplitudes by the ratio of mean amplitude to computed amplitude, which
              amplifies weak spots while attenuating strong spots.
            ``'WGS-Kim'`` [3]_
              Improves the convergence of `Leonardo` by fixing the far-field
              phase after a desired number of iterations or after achieving a desired
              efficiency (fraction of far-field energy at the desired points).
            ``'WGS-Nogrette'`` [7]_
              Weights target intensities by a tunable gain factor.

        Note
        ~~~~
        This function uses parameter naming convention borrowed from
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
        verbose : bool
            Whether to display :mod:`tqdm` progress bars.
            These bars are also not displayed for ``maxiter <= 1``.
        callback : callable OR None
            Same functionality as the equivalently-named parameter in
            :meth:`scipy.optimize.minimize()`. ``callback`` must accept a Hologram
            or Hologram subclass as the single argument. If ``callback`` returns
            ``True``, then the optimization exits. Ignored if ``None``.
        feedback : str or None
            Type of feedback requested. For non-subclasses of :class:`Hologram`, this can only
            be ``"computational"`` feedback.
            When using WGS, defaults to ``"computational"`` if empty.
        **kwargs : dict, optional
            Various weight keywords and values to pass depending on the weight method.
            See method keywords below. These are passed into :attr:`flags`.

        References
        ----------
        .. [6] R. Di Leonardo, F. Ianni, and G. Ruocco, "Computer generation of
               optimal holograms for optical trap arrays," Opt. Express 15, (2007).
        .. [7] F. Nogrette et al., "Single-Atom Trapping in Holographic 2D Arrays
               of Microtraps with Arbitrary Geometries" Phys. Rev. X 4, (2014).
        """

        # Check and record method.
        methods = [ "GS",
                    "WGS-Leonardo",
                    "WGS-Kim",
                    "WGS-Nogrette"]

        assert method in methods, \
            "Unrecognized method {}. Valid methods inlcude [{}]".format(method, methods)

        self.method = method

        # Handle flags.
        for flag in kwargs:
            self.flags[flag] = kwargs[flag]

        self.flags['feedback'] = feedback
        self.flags['stat_groups'] = stat_groups

        # Iterations to process.
        iterations = range(maxiter)

        # Decide whether to use a tqdm progress bar. Don't use a bar for N=1.
        if verbose and maxiter > 1:
            iterations = tqdm(iterations)

        # Switch between methods
        if  "GS" in method:
            if "WGS" in method:
                if len(self.flags['feedback']) == 0:
                    self.flags['feedback'] = 'computational'

            self.GS(iterations, callback)

    # Optimization methods (currently only GS-type is supported)
    def GS(self, iterations, callback):
        """
        GPU-accelerated Gerchberg-Saxton (GS) iterative phase retrieval.

        Solves the "phase problem": approximates the near-field phase that
        transforms a known near-field source amplitude to a known near-field
        target amplitude.

        Caution
        ~~~~~~~
        This function should be called through :meth:`.optimize()` and not called directly.

        Note
        ~~~~
        FFTs are **not** in-place in this algorithm. In both non-:mod:`cupy` and 
        :mod:`cupy` implementations, :mod:`numpy.fft` does not support in-place
        operations.  However, :mod:`scipy.fft` does in both. In the future, we may move to the scipy
        implementation. However, neither :mod:`numpy` or :mod:`scipy` ``fftshift`` support
        in-place movement (for obvious reasons). For even faster computation, algorithms should
        consider **not shifting** the FTT result, and instead shifting measurement data / etc to
        this unshifted basis.

        Parameters
        ----------
        iterations : iterable
            Number of loop iterations to run. Is an iterable to pass a :mod:`tqdm` iterable.
        callback : callable OR None
            See :meth:`.optimize()`.
        """
        # Future: in-place FFT
        # Future: rename nearfield and farfield to both be "complex" to avoid hogging memory.

        # Proxy to initialize nearfield with the correct shape and (complex) type.
        nearfield = cp.exp(1j * self.target)

        # Helper variables for speeding up source phase and amplitude fixing.
        (i0, i1, i2, i3) = toolbox.unpad(self.shape, self.slm_shape)

        for _ in iterations:
            # Fix the relevant part of the nearfield amplitude to the source amplitude.
            # Everything else is zero because power outside the SLM is assumed unreflected.
            # This is optimized for when shape is much larger than slm_shape.
            nearfield.fill(0)
            nearfield[i0:i1, i2:i3] = self.amp * cp.exp(1j * self.phase)
            farfield = cp.fft.fftshift(cp.fft.fft2(nearfield, norm='ortho'))

            # Calculate amp_ff, if needed.
            if 'computational' in self.flags['feedback'] or \
                any('computational' in group for group in self.flags['stat_groups']):
                # Calculate amp_ff for weighting (if None, will init; otherwise in-place).
                self.amp_ff = cp.abs(farfield, out=self.amp_ff)

            # Erase irrelevant images from the past loop.
            if hasattr(self, 'img_ij'):
                self.img_ij = None

            # Weight, if desired. This function also updates stats.
            if "WGS" in self.method:
                self._update_weights()

                # Calculate amp_ff again, as _update_weights may have modified it.
                # This is to reduce memory use at the slight cost of performance.
                if 'computational' in self.flags['feedback'] or \
                    any('computational' in group for group in self.flags['stat_groups']):
                    # Calculate amp_ff for weighting (if None, will init; otherwise in-place).
                    self.amp_ff = cp.abs(farfield, out=self.amp_ff)
            
            self.update_stats(self.flags['stat_groups'])

            # Run step function and check termination conditions.
            if callback is not None and callback(self):
                break

            # Fix amplitude, potentially also fixing the phase.
            if 'fixed_phase' in self.flags and \
                self.flags['fixed_phase'] and \
                self.phase_ff is not None:
                # Set the farfield to the stored phase and updated weights.
                cp.exp(1j * self.phase_ff, out=farfield)
                cp.multiply(farfield, self.weights, out=farfield)

                # Future: check this potentially-optimized method
                # farfield.fill(0)
                # mask = self.weights != 0
                # cp.exp(1j * self.phase_ff[mask], out=farfield[mask])
                # cp.multiply(farfield[mask], self.weights[mask], out=farfield[mask])
            else:
                # Set the farfield amplitude to the updated weights.
                cp.divide(farfield, cp.abs(farfield), out=farfield)
                cp.multiply(farfield, self.weights, out=farfield)
                cp.nan_to_num(farfield, copy=False, nan=0)

                # Future: check this potentially-optimized method
                # farfield.fill(0)
                # mask = self.weights != 0
                # cp.divide(farfield[mask], cp.abs(farfield[mask]), \
                #         out=farfield[mask], where=farfield!=0)
                # cp.multiply(farfield[mask], self.weights[mask], out=farfield[mask])

            # Move to real space.
            nearfield = cp.fft.ifft2(cp.fft.ifftshift(farfield), norm='ortho')
            cp.arctan2(nearfield.imag[i0:i1, i2:i3], nearfield.real[i0:i1, i2:i3], out=self.phase)

            # Increment iterations.
            self.N += 1

        # Update the final far-field
        nearfield.fill(0)
        nearfield[i0:i1, i2:i3] = self.amp * cp.exp(1j * self.phase)
        farfield = cp.fft.fftshift(cp.fft.fft2(nearfield, norm='ortho'))
        cp.abs(farfield, out=self.amp_ff)
        self.phase_ff = cp.angle(farfield)

    # User interactions: Changing the target and recovering the phase.
    def _update_target(self, new_target, reset_weights=False):
        """
        Change the target to something new. This method handles cleaning and normalization.

        new_target : numpy.ndarray OR cupy.ndarray OR None
            If ``None``, sets the target to zero.
        reset_weights : bool
            Whether to overwrite ``weights`` with ``target``.
        """
        if new_target is None:
            self.target = cp.zeros(shape=self.shape, dtype=self.dtype)
        else:
            assert new_target.shape == self.shape, "Target must be of appropriate shape. "\
                "Initialize a new Hologram if a different shape is desired."

            self.target = cp.abs(cp.array(new_target, dtype=self.dtype))
            self.target *= 1/Hologram.norm(self.target)

        if reset_weights:
            self.weights = cp.copy(self.target)
    def update_target(self, new_target, reset_weights=False):
        """
        Allows the user to change the target to something new.
        Cleaning and normalization is handled.

        Parameters
        ----------
        new_target : array_like OR None
            New :attr:`target` to optimize towards. Should be of shape :attr:`shape`.
            If ``None``, :attr:`target` is zeroed (used internally, but probably should not
            be used by a user).
        reset_weights : bool
            Whether to update the :attr:`weights` to this new :attr:`target`.
        """
        self._update_target(new_target=new_target, reset_weights=reset_weights)
    def extract_phase(self):
        r"""
        Collects the current near-field phase from the GPU with :meth:`cupy.ndarray.get()`.
        Also shifts the :math:`[-\pi, \pi]` range of :meth:`numpy.arctan2()` to :math:`[0, 2\pi]`
        for faster writing to the SLM (see :meth:`~slmsuite.hardware.slms.slm.SLM.write()`).

        Returns
        -------
        numpy.ndarray
            Current near-field phase computed by GS.
        """
        if cp != np:
            return self.phase.get() + np.pi
        return self.phase + np.pi

    # Weighting
    @staticmethod
    def _update_weights_generic(weight_amp, feedback_amp, target_amp=None, method=None):
        """
        Helper function to process weight feedback according to the chosen weighting method.

        Caution
        ~~~~~~~~
        ``weight_amp`` *is* modified in-place and ``feedback_amp`` *may be* modified in-place.

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
        method : str OR None
            Weighting method, see the method descriptions in :meth:`optimize()`.

        Returns
        ----------
        numpy.ndarray OR cupy.ndarray
            The updated ``weight_amp``.
        """

        if target_amp is None:  # Uniform
            feedback_corrected = feedback_amp
        else:                   # Non-uniform
            feedback_corrected = feedback_amp
            norm = Hologram.norm(feedback_amp)
            feedback_corrected *= 1/norm

            cp.divide(  feedback_corrected,
                        target_amp,
                        out=feedback_corrected)

            cp.nan_to_num(feedback_corrected, copy=False, nan=1)

        # TODO: fix
        method='Kim'
        factor = .1
        power = .7

        if method == 'Leonardo' or method == 'Kim':
            # Leonardo uses amp.
            cp.power(feedback_corrected, -power, out=feedback_corrected)
            weight_amp *= feedback_corrected
        elif method == 'Nogrette':
            # TODO: paper uses pwr, not amp. Remove?
            # Taylor expand 1/(1-g(1-x)) -> 1 + g(1-x) + (g(1-x))^2 ~ 1 + g(1-x)
            feedback_corrected *= -(1 / cp.nanmean(feedback_corrected))
            feedback_corrected += 1
            feedback_corrected *= -factor
            feedback_corrected += 1
            cp.reciprocal(feedback_corrected, out=feedback_corrected)

            weight_amp *= feedback_corrected
        else:
            raise RuntimeError("Method ""{}"" not recognized by Hologram.optimize()"\
                .format(method))

        cp.nan_to_num(weight_amp, copy=False, nan=0)

        # Normalize amp power, as methods may have broken power conservation.
        norm = Hologram.norm(weight_amp)
        weight_amp *= 1/norm

        return weight_amp
    def _update_weights(self):
        """
        Change :attr:`weights` to optimize towards the :attr:`target` using feedback from
        :attr:`amp_ff`, the computed farfield amplitude. This function also updates stats.
        """
        feedback = self.flags['feedback']

        if feedback == "computational":
            self._update_weights_generic(self.weights, self.amp_ff, self.target, method=self.method)

    # Statistics handling
    @staticmethod
    def _calculate_stats(feedback_amp, target_amp, mp=cp, efficiency_compensation=True, total=None):
        """
        Helper function to analyze how close the feedback is to the target.

        feedback_amp : numpy.ndarray OR cupy.ndarray
            Computational or measured result of holography.
        target_amp : numpy.ndarray OR cupy.ndarray
            Target of holography.
        mp : module
            This function is used by both :mod:`cupy` and :mod:`numpy`, so we have the option
            of either. Defaults to :mod:`cupy`.
        efficiency_compensation : bool
            Whether to scale the ``feedback`` based on the overlap with the ``target``.
            This is more accurate for images, but less accurate for SpotHolograms.
        """
        # Downgrade to numpy if necessary
        if isinstance(feedback_amp, np.ndarray) or isinstance(target_amp, np.ndarray):
            if not isinstance(feedback_amp, np.ndarray):
                feedback_amp = feedback_amp.get()
                
            if not isinstance(target_amp, np.ndarray):
                target_amp = target_amp.get()

            if total is not None:
                total = float(total)
            
            mp = np

        feedback_pwr = mp.square(feedback_amp)
        target_pwr = mp.square(target_amp)

        if total is not None:
            efficiency = float(mp.sum(feedback_pwr)) / total

        # Normalize.
        feedback_pwr_sum = (mp.sum(feedback_pwr))
        feedback_pwr *= 1/feedback_pwr_sum
        feedback_amp *= 1/mp.sqrt(feedback_pwr_sum)

        target_pwr_sum = (mp.sum(target_pwr))
        target_pwr *= 1/target_pwr_sum
        target_amp *= 1/mp.sqrt(target_pwr_sum)


        if total is None:
            # Efficiency overlap integral.
            efficiency = np.square(float(mp.sum(mp.multiply(target_amp, feedback_amp))))
            if efficiency_compensation:
                feedback_pwr *= 1/efficiency

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

        pkpk_err = float(mp.amax(pwr_err) - mp.amin(pwr_err))
        std_err = float(mp.std(pwr_err))

        return {"efficiency":efficiency, "uniformity":uniformity, \
                "pkpk_err":pkpk_err, "std_err":std_err}
    def _calculate_stats_computational(self, stats, stat_groups=[]):
        if 'computational' in stat_groups:
            stats['computational'] = self._calculate_stats( self.amp_ff, self.target,
                                                            efficiency_compensation=False)
    def _update_stats_dictionary(self, stats):
        """
        Helper function to manage additions to the :attr:`stats`.

        Parameters
        ----------
        stats : dict of dicts
            Dictionary of groups, each group containing a dictionary of stats.
        """
        # Update methods
        M = len(self.stats['method'])
        difference = self.N + 1 - M
        if difference > 0:                          # Extend methods
            self.stats['method'].extend(['' for _ in range(difference)])
            M = self.N + 1
        self.stats['method'][self.N] = self.method  # Update method

        # Update flags
        flaglist = set(self.flags.keys()).union(set(self.stats['flags'].keys()))
        for flag in flaglist:
            # Extend flag
            if not flag in self.stats['flags']:
                self.stats['flags'][flag] = [np.nan for _ in range(M)]
            else:
                difference = self.N + 1 - len(self.stats['flags'][flag])
                if difference > 0:
                    self.stats['flags'][flag].extend([np.nan for _ in range(difference)])

            # Update flag
            if flag in self.flags:
                self.stats['flags'][flag][self.N] = self.flags[flag]

        # Update stats
        grouplist = set(stats.keys()).union(set(self.stats['stats'].keys()))
        if len(grouplist) > 0:
            statlists = [set(stats[group].keys()) for group in stats.keys()]
            if len(self.stats['stats'].keys()) > 0:
                key = next(iter(self.stats['stats']))
                statlists.append(set(self.stats['stats'][key].keys()))
            statlist = set.union(*statlists)

            for group in grouplist:
                # Check this group
                if not group in self.stats['stats']:
                    self.stats['stats'][group] = {}

                if len(statlist) > 0:
                    for stat in statlist:
                        # Extend stat
                        if not stat in self.stats['stats'][group]:
                            self.stats['stats'][group][stat] = [np.nan for _ in range(M)]
                        else:
                            difference = self.N + 1 - len(self.stats['stats'][group][stat])
                            if difference > 0:
                                self.stats['stats'][group][stat].extend([np.nan for _ in range(difference)])

                        # Update stat
                        if group in stats.keys():
                            if stat in stats[group].keys():
                                self.stats['stats'][group][stat][self.N] = stats[group][stat]
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

    # Visualization
    def plot_nearfield(self, title='', padded=False):
        """
        Plots the amplitude (left) and phase (right) of the nearfield (plane of the SLM).
        The amplitude is assumed (whether uniform, or experimentally computed) while the 
        phase is the result of optimization.

        Parameters
        ----------
        title : str
            Title of the plots.
        padded : bool
            If ``True``, shows the full computational space of shape :attr:`shape`.
            Otherwise, shows the region at the center of the computational space of
            size :attr:`slm_shape`.
        """
        _, axs = plt.subplots(1, 2, constrained_layout=True, figsize=(12,6))

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
            axs[0].imshow(toolbox.pad(amp * np.ones(self.slm_shape),
                self.shape if padded else self.slm_shape), vmin=0, vmax=amp)
        else:
            axs[0].imshow(toolbox.pad(amp,
                self.shape if padded else self.slm_shape), vmin=0, vmax=np.amax(amp))

        axs[1].imshow(toolbox.pad(phase, self.shape if padded else self.slm_shape), 
            vmin=-np.pi, vmax=np.pi, interpolation='none')
        
        if len(title) > 0:
            title += ': '

        axs[0].set_title(title + 'Amplitude')
        axs[1].set_title(title + 'Phase')

        plt.show()
    def plot_farfield(self, source=None, limits=None, limit_padding=.2, title=''):
        """
        Plots an overview (left) and zoom (right) view of ``source``.

        Parameters
        ----------
        source : array_like OR None
            Should have shape equal to :attr:`shape`.
            If ``None``, defaults to :attr:`target`.
        limits : ((float, float), (float, float)) OR None
            :math:`x` and :math:`y` limits for the zoom plot.
            If None, ``limits`` are autocomputed as the smallest bounds
            that show all non-zero values (plus ``limit_padding``).
            Note that autocomputing on :attr:`target` will perform well,
            as zero values are set to actually be zero. However, doing so on 
            computational or experimental outputs (e.g. :attr:`amp_ff`) will likely perform
            poorly, as values deviate slightly from zero and artificially expand the ``limits``.
        limit_padding : float
            Fraction of the width and height to expand the limits by, only if
            the passed ``limits`` is ``None`` (autocompute).
        title : str
            Title of the plots.

        Returns
        -------
        ((float, float), (float, float))
            Used ``limits``, which may be autocomputed. If autocomputed, the result will
            be integers.
        """
        if source is None:
            source = self.target

            if len(title) == 0:
                title = 'Target'

        try:
            npsource = cp.abs(source).get()
        except:
            npsource = np.abs(source)

        _, axs = plt.subplots(1, 2, constrained_layout=True, figsize=(12,6))

        if limits == None:
            # Determine the bounds of the zoom region, padded by 20%
            limits = []
            binary = npsource > 0

            for a in [0,1]:
                collapsed = np.where(np.any(binary, axis=a))  # Collapse the other axis
                limit = np.array([np.amin(collapsed), np.amax(collapsed)])

                padding = int(np.diff(limit) * limit_padding)
                limit += np.array([-padding, padding])

                limit = np.clip(limit, 0, self.shape[a])

                limits.append(limit)

        # Plot the full target, blurred so single pixels are visible in low res
        b = 2*int(max(self.shape)/500) + 1  # Future: fix arbitrary
        axs[0].imshow(cv2.GaussianBlur(npsource, (b, b), 0))

        # Plot a red rectangle to show the extents of the zoom region
        rect = plt.Rectangle([limits[0][0], limits[1][0]],
            np.diff(limits[0])[0], np.diff(limits[1])[0], ec='r', fc='none')
        axs[0].add_patch(rect)

        # If cam_points is defined (i.e. is a FeedbackSLM), plot a yellow rectangle for the extents of the camera
        try:
            axs[0].plot(self.shape[1]/2. + self.cam_points[0],
                        self.shape[0]/2. + self.cam_points[1], c='y')
        except:
            pass

        if len(title) > 0:
            title += ': '

        axs[0].set_title(title + 'Full')

        # Zoom in on our spots
        axs[1].imshow(npsource)
        axs[1].set_xlim(limits[0])
        axs[1].set_ylim(np.flip(limits[1]))
        axs[1].set_title(title + 'Zoom')

        plt.show()

        return limits
    def plot_stats(self, stats_dict=None):
        """
        Plots the statistics contained in the given dictionary.

        Parameters
        ----------
        stats_dict : dict OR None
            If ``None``, defaults to :attr:`stats`.
        """
        if stats_dict is None:
            stats_dict = self.stats

        _, ax = plt.subplots(1,1)

        # solid, densely dashed, sparesely dotted, densely dotted
        linestyles = ['solid', (0, (5, 1)), (0, (1, 2)), (0, (1, 1))]
        stats = ['efficiency', 'uniformity', 'pkpk_err', 'std_err']
        legendstats = ['inefficiency', 'nonuniformity', 'pkpk_err', 'std_err']
        niter = np.arange(0, len(stats_dict['method']))

        stat_keys = stats_dict['stats'].keys()
        assert len(stat_keys) <= 10, "Not enough default colors to describe all modes."

        lines = []
        color_num = 0

        for stat_key in stat_keys:
            stat_group = stats_dict['stats'][stat_key]

            color = 'C'+str(color_num)
            color_num += 1

            for i in range(len(stats)):
                # Invert the stats if it is efficiency or uniformity.
                y = stat_group[stats[i]]
                if i < 2:
                    y = 1-np.array(y)

                line = ax.semilogy(niter, y, c=color, ls=linestyles[i])[0]

                if i == 0:  # Remember the solid lines for the legend.
                    lines.append(line)

        # Make the linestyle legend.
        # Inspired from https://stackoverflow.com/a/46214879
        dummy_lines = []
        for i in range(len(stats)):
            dummy_lines.append(ax.plot([],[], c="black", ls=linestyles[i])[0])
        legend1 = plt.legend(dummy_lines, legendstats, loc='center right')

        # Make the color legend.
        plt.legend(lines, stat_keys, loc='center left')

        # Add the linestyle legend back in and show.
        ax.add_artist(legend1)
        plt.show()

    # Other helper functions
    @staticmethod
    def set_mempool(device=0, size=None, fraction=None):
        """
        Helper function to set the cupy memory pool size. See [8]_.

        References
        --------

        .. [8] https://docs.cupy.dev/en/stable/reference/generated/cupy.cuda.MemoryPool.html#cupy.cuda.MemoryPool

        Parameters
        ----------
        device : int
            Which GPU to set the limit on. Passed to :meth:`cupy.cuda.Device()`.
        size : int
            Desired number of bytes in the pool. Passed to :meth:`cupy.cuda.MemoryPool.set_limit()`.
        fraction : float
            Fraction of availible memory to use. Passed to :meth:`cupy.cuda.MemoryPool.set_limit()`.
        """
        mempool = cp.get_default_memory_pool()

        with cp.cuda.Device(device):
            mempool.set_limit(size=size, fraction=fraction)

            print("cupy memory pool limit set to {} GB...".format(mempool.get_limit() / (1024. ** 3)))
    @staticmethod
    def norm(matrix):
        r"""
        Computes the root of the sum of squares of the given ``matrix``. Implements:
        
        .. math:: \sqrt{\iint |\vec{E}|^2}

        Parameters
        ----------
        matrix : numpy.ndarray OR cupy.ndarray
            Data, potentially complex.

        Returns
        -------
        float
            The result.
        """
        if cp.iscomplexobj(matrix):
            return (cp.sqrt(cp.sum(cp.square(cp.abs(matrix)))))
            # return cp.sqrt(cp.sum(cp.multiply(matrix, cp.conj(matrix))))
        else:
            return (cp.sqrt(cp.sum(cp.square(matrix))))

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
        Target in the ``"ij"`` basis. Of same ``shape`` as the camera in :attr:`cameraslm`.
        Counterpart to :attr:`target` which is in the ``"knm"`` basis.
    img_ij, img_knm
        Cached feedback image in the ``"ij"`` (raw) basis or ``"knm"`` (transformed) basis.
        Measured with :meth:`.measure()`.
    """
    def __init__(self, shape, target_ij=None, cameraslm=None, **kwargs):
        """
        Parameters
        --------
        shape : (int, int)
            Computational shape of the SLM. See :meth:`.Hologram.__init__()`.
        target_ij : array_like OR None
            See :attr:`target_ij`. Should only be ``None`` if the :attr:`target`
            will be generated by other means (see :class:`SpotHologram`), so the
            user should generally provide an array.
        cameraslm : slmsuite.hardware.cameraslms.FourierSLM OR None
            See :attr:`cameraslm`.
        """
        # Use the Hologram construtor to initialize self.target with proper shape,
        # pass other arguments (esp. slm_shape).
        self.cameraslm = cameraslm
        if self.cameraslm is not None:
            # Determine camera size in SLM-space.
            amp = self.cameraslm.slm.measured_amplitude
            slm_shape = self.cameraslm.slm.shape
        else:
            amp = None
            slm_shape = None

        if not 'slm_shape' in kwargs:
            kwargs['slm_shape'] = slm_shape

        super().__init__(target=shape, amp=amp, **kwargs)

        self.img_ij = None
        self.img_knm = None
        if target_ij is None:
            self.target_ij = None
        else:
            self.target_ij = target_ij.astype(self.dtype)

        if self.cameraslm is not None and self.cameraslm.fourier_calibration is not None:
            # Transform the target, if it is provided.
            if target_ij is not None:
                self.update_target(target_ij, reset_weights=True)

            cam_shape = self.cameraslm.cam.shape

            ll = [0, 0]
            lr = [0, cam_shape[0]-1]
            ur = [cam_shape[1]-1, cam_shape[0]-1]
            ul = [cam_shape[1]-1, 0]
            
            points_ij = toolbox.clean_2vectors(np.vstack((ll, lr, ur, ul, ll)).T)
            points_kxy = self.cameraslm.ijcam_to_kxyslm(points_ij)
            self.cam_points = toolbox.convert_blaze_vector(points_kxy, "kxy", "knm", 
                                                slm=self.cameraslm.slm, shape=self.shape)
            self.cam_shape = cam_shape
        else:
            self.cam_points = None
            self.cam_shape = None

    def ijcam_to_knmslm(self, img, output=None, blur_ij=None):
        """
        Convert an image in the camera domain to computational SLM k-space using, in part, the
        affine transformation stored in a cameraslm's Fourier calibration.

        Note
        ~~~~~~~~
        This includes two transformations:

         - The affine transformation ``"ij"`` -> ``"kxy"`` (camera pixels to normalized k-space).
         - The scaling ``"kxy"`` -> ``"knm"`` (normalized k-space to computational k-space pixels).

        Note
        ~~~~~~~~
        Future optimizations might include default blurring of the ``image`` or ``target``,
        along with different interpolation ``order`` (see :meth:`scipy.ndimage.affine_transform()`).

        Parameters
        ----------
        img : numpy.ndarray OR cupy.ndarray
            Image to transform. This should be the same shape as images returned by the camera.
        output : numpy.ndarray OR cupy.ndarray OR None
            If ``output`` is not ``None``, this array will be used to write the memory in-place.

        Returns
        ----------
        numpy.ndarray OR cupy.ndarray
            Image transformed into ``"knm"`` space.
        """
        assert self.cameraslm is not None
        assert self.cameraslm.fourier_calibration is not None

        # First transformation.
        conversion = toolbox.convert_blaze_vector([1, 1], "knm", "kxy", 
            slm=self.cameraslm.slm, shape=self.shape)
        M1 = np.diag(np.squeeze(conversion))
        b1 = -toolbox.clean_2vectors(np.flip(np.squeeze(self.shape))/2)

        # Second transformation.
        M2 = self.cameraslm.fourier_calibration["M"]
        b2 = self.cameraslm.fourier_calibration["b"] - \
            np.matmul(M2, self.cameraslm.fourier_calibration["a"])
        
        # Composite transformation (along with xy -> yx).
        M = cp.array(np.matmul(M2, M1).T)
        b = cp.array(np.flip(np.matmul(np.matmul(M2, M1), b1) + b2))

        # See if the user wants to blur.
        if blur_ij is None:
            if "blur_ij" in self.flags:
                blur_ij = self.flags["blur_ij"]
            else:
                blur_ij = 0
        
        # Future: use cp_gaussian_filter; was having trouble with this.
        if blur_ij > 0:
            img = sp_gaussian_filter(img, (blur_ij, blur_ij), output=img, truncate=2)

        cp_img = cp.array(img.astype(self.dtype))
        cp.abs(cp_img, out=cp_img)

        # Perform affine.
        target = cp_affine_transform(   input=cp_img,
                                        matrix=M,
                                        offset=b,
                                        output_shape=self.shape,
                                        output=output,
                                        mode="constant",
                                        cval=0)

        # Filter the image. Future: fix.
        # target = cp_gaussian_filter1d(target, blur, axis=0, output=target, truncate=2)
        # target = cp_gaussian_filter1d(target, blur, axis=1, output=target, truncate=2)

        target = cp.abs(target, out=target)
        target *= 1/Hologram.norm(target)

        return target
    def update_target(self, new_target, reset_weights=False):
        self.ijcam_to_knmslm(new_target, output=self.target)
        
        if reset_weights:
            cp.copyto(self.weights, self.target)

    def measure(self, basis="ij"):
        """
        Method to request a measurement to occur. If :attr:`img_ij` is ``None``,
        then a new image will be grabbed from the camera.

        Parameters
        ----------
        basis : str
            The basis to be sure to fill with data. Can be ``"ij"`` or ``"knm"``.
        """
        if self.img_ij is None:
            self.cameraslm.slm.write(self.extract_phase())
            self.cameraslm.cam.flush()
            self.img_ij = self.cameraslm.cam.get_image()

            if basis == "knm":  # Compute the knm basis image.
                self.img_knm = self.ijcam_to_knmslm(self.img_ij, output=self.img_knm)
                cp.sqrt(self.img_knm, out=self.img_knm)
            else:               # The old image is outdated, erase it. Future: memory concerns?
                self.img_knm = None

            self.img_ij = np.sqrt(self.img_ij)   # Don't load to the GPU if not neccesary.
        elif basis == "knm":
            if self.img_knm is None:
                self.ijcam_to_knmslm(np.square(self.img_ij), output=self.img_knm)
                cp.sqrt(self.img_knm, out=self.img_knm)
    def _update_weights(self):
        """
        Change :attr:`weights` to optimize towards the :attr:`target` using feedback from
        :attr:`amp_ff`, the computed farfield amplitude. This function also updates stats.
        """
        feedback = self.flags['feedback']

        if feedback == "computational":
            self._update_weights_generic(self.weights, self.amp_ff, self.target, method=self.method)
        elif feedback == "experimental":
            self.measure("knm")  # Make sure data is there.
            self._update_weights_generic(self.weights, self.img_knm, self.target, method=self.method)

    def _calculate_stats_experimental(self, stats, stat_groups=[]):
        if 'experimental_knm' in stat_groups:
            self.measure("knm")  # Make sure data is there.
            stats['experimental_knm'] = self._calculate_stats(self.img_knm, self.target,
                                                            efficiency_compensation=True)
        if 'experimental_ij' in stat_groups or 'experimental' in stat_groups:
            self.measure("ij")  # Make sure data is there.
            stats['experimental_ij'] = self._calculate_stats(self.img_ij.astype(self.dtype), 
                                                            self.target_ij, mp=np,
                                                            efficiency_compensation=True)
    def update_stats(self, stat_groups=[]):
        """
        Calculate statistics corresponding to the desired ``stat_groups``.

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

    Attributes
    ----------
    spot_knm, spot_kxy, spot_ij : array_like OR None
        Stored vectors with shape ``(2, N)`` in the style of 
        :meth:`~slmsuite.holography.toolbox.clean_2vectors()`.
        The subscript refers to the basis of the vectors, the transformations between
        which are autocomputed.
        If necessary transformations do not exist, :attr:`spot_ij` is set to ``None``.
    spot_knm_rounded : array_like
        :attr:`spot_knm` rounded to nearest integers (indices). This is necessary because 
        GS algorithms operate on a pixel grid, and the target for each spot in a 
        :class:`SpotHologram` is a single pixel (index).
    spot_kxy_rounded, spot_ij_rounded : array_like
        Once :attr:`spot_knm_rounded` is rounded, the original :attr:`spot_kxy` 
        and :attr:`spot_ij` are no longer accurate. Transformations are again used
        to backcompute the positons in the ``"ij"`` and ``"kxy"`` bases corresponding
        to the true computational location of a given spot.
    spot_amp : array_like
        The target amplitude for each spot.
        Must have length corresponding to the number of spots.
        For instance, the user can request dimmer or brighter spots.
    """

    def __init__(self, shape, spot_vectors, basis="knm", spot_amp=None, cameraslm=None, **kwargs):
        """
        Initializes a :class:`SpotHologram` targeting given spots at ``spot_vectors``.

        Parameters
        ----------
        shape : (int, int)
            Computational shape of the SLM. See :meth:`.Hologram.__init__()`.
        spot_vectors : array_like
            Spot position vectors with shape ``(2, N)`` in the style of 
            :meth:`~slmsuite.holography.toolbox.clean_2vectors()`.
        basis : str
            The spots can be in any of the following bases:

            - ``"ij"`` for camera coordinates (pixels),
            - ``"kxy"`` for centered normalized SLM k-space (radians).
            - ``"knm"`` for centered computational SLM k-space (pixels).
            
            Defaults to ``"knm"`` if ``None``.
        spot_amp : array_like OR None
            See :attr:`SpotHologram.spot_amp`.
            If ``None``, all spots are assumed to have the same amplitude.
            Normalization is performed automatically; the user is not required to normalize.
        cameraslm : slmsuite.hardware.cameraslms.FourierSLM OR None
            If the ``"ij"`` basis is chosen, and/or if the user wants to make use of camera
            feedback, a cameraslm must be provided.
        **kwargs
            Passed to :meth:`.FeedbackHologram.__init__()`.
        """
        vectors = toolbox.clean_2vectors(spot_vectors)

        if spot_amp is not None:
            assert np.shape(vectors)[1] == len(spot_amp.ravel()), \
                "spot_amp must have the same length as the provided spots."

        # Handle the basis.
        if basis is None or basis == "knm": # Computational Fourier space of SLM, zero-centered
            self.spot_knm = vectors

            if cameraslm is not None:
                self.spot_kxy = toolbox.convert_blaze_vector(
                    self.spot_knm, "knm", "kxy", cameraslm.slm, shape)

                if cameraslm.fourier_calibration is not None:
                    self.spot_ij = cameraslm.kxyslm_to_ijcam(self.spot_kxy)
                else:
                    self.spot_ij = None
            else:
                self.spot_kxy = None
                self.spot_ij = None
        elif basis == "kxy":                # Normalized units
            assert cameraslm is not None, "We need an cameraslm to interpret ij."

            self.spot_kxy = vectors

            try:
                self.spot_ij = cameraslm.kxyslm_to_ijcam(vectors)
            except:     # This is okay for non-feedback GS, so we don't error.
                self.spot_ij = None

            self.spot_knm = toolbox.convert_blaze_vector(
                self.spot_kxy, "kxy", "knm", cameraslm.slm, shape)
        elif basis == "ij":                     # Pixel on the camera
            assert cameraslm is not None, "We need an cameraslm to interpret ij."
            assert cameraslm.fourier_calibration is not None, "We need an cameraslm with " \
                    "fourier-calibrated kxyslm_to_ijcam and ijcam_to_kxyslm transforms " \
                    "to interpret ij."

            self.spot_ij = vectors
            self.spot_kxy = cameraslm.ijcam_to_kxyslm(vectors)
            self.spot_knm = toolbox.convert_blaze_vector(
                self.spot_kxy, "kxy", "knm", cameraslm.slm, shape)
        else:
            raise Exception("Unrecognized basis '{}'.".format(basis))

        # Check to make sure spots are within relevant camera and SLM shapes.
        if  np.any(np.abs(self.spot_knm[0]) > shape[1]/2.) or \
            np.any(np.abs(self.spot_knm[1]) > shape[0]/2.):
           raise ValueError("Spots outside SLM computational space bounds!")

        if self.spot_ij is not None:
            cam_shape = cameraslm.cam.shape
            # TODO: psf?
            if  np.any(self.spot_ij[0] < 0) or np.any(self.spot_ij[0] >= cam_shape[1]) or \
                np.any(self.spot_ij[1] < 0) or np.any(self.spot_ij[1] >= cam_shape[0]):
                raise ValueError("Spots outside camera bounds!")

        # Parse spot_amp.
        if spot_amp is None:
            self.spot_amp = np.ones_like(vectors[0])
        else:
            self.spot_amp = spot_amp.ravel()

        # Initialize target/etc.
        super().__init__(shape, target_ij=None, cameraslm=cameraslm, **kwargs)

        # Fill the target with data.
        self.update_target(reset_weights=True)

    @staticmethod
    def make_rectangular_array( shape, array_shape, array_pitch, array_center=(0,0),
                                basis="knm", parity_check=False, **kwargs):
        """
        Helper function to initialize a rectangular 2D array of spots, with certain size and pitch.

        Note
        ~~~~~~~~
        The array can be in SLM k-space coordinates or in camera pixel coordinates, depending upon
        the choice of ``basis``. For the ``"ij"`` basis, ``cameraslm`` must be included as one
        of the ``kwargs``. See :meth:`__init__()` for more ``basis`` information.

        Parameters
        ----------
        shape : (int, int)
            Computational shape of the SLM. See :meth:`.Hologram.__init__()`.
        array_shape : (int, int) OR int
            The size of the rectangular array in number of spots (NX, NY).
            If a single N is given, assume (N, N).
        array_pitch : (float, float) OR float
            The spacing between spots in the x and y directions in kxy coordinates.
            If a single pitch is given, assume (pitch, pitch).
        array_center : (float, float)
            The shift of the center of the spot array from the zeroth order in
            kxy coordinates. (kx, ky) form.
        basis : str
            See :meth:`__init__()`.
        parity_check : bool
            Whether to delete the last two points to check for parity.
        **kwargs
            Any other arguments are passed to :meth:`__init__()`.
        """
        # Parse size and pitch.
        if isinstance(array_shape, (int, float)):
            array_shape = (int(array_shape), int(array_shape))
        if isinstance(array_pitch, (int, float)):
            array_pitch = (array_pitch, array_pitch)

        # Make the grid edges.
        x_edge = (np.arange(array_shape[0]) - (array_shape[0]-1)/2) * array_pitch[0] \
                    + array_center[0]
        y_edge = (np.arange(array_shape[1]) - (array_shape[1]-1)/2) * array_pitch[1] \
                    + array_center[1]

        # Make the grid lists.
        x_grid, y_grid = np.meshgrid(x_edge, y_edge, sparse=False, indexing='xy')
        x_list, y_list = x_grid.ravel(), y_grid.ravel()

        # Delete the last two points if desired and valid.
        if parity_check and len(x_list) > 2:
            x_list = x_list[:-2]
            y_list = y_list[:-2]

        vectors = np.vstack((x_list, y_list))

        # Return a new SpotHologram.
        return SpotHologram(shape, vectors, basis=basis, spot_amp=None, **kwargs)

    def _update_target_spots(self, reset_weights=False, plot=False):
        """
        Wrapped by :meth:`SpotHologram.update_target()`.
        """
        # Erase previous target in-place. Future: Optimize speed if positions haven't shifted?
        self.target.fill(0)

        shape = toolbox.clean_2vectors(self.shape).astype(np.float)

        self.spot_knm_rounded = np.ceil(shape/2 + self.spot_knm.astype(np.float)).astype(np.int)

        if self.cameraslm is not None:
            self.spot_kxy_rounded = toolbox.convert_blaze_vector(
                self.spot_knm_rounded - shape/2, "knm", "kxy", self.cameraslm.slm, self.shape)

            if self.cameraslm.fourier_calibration is not None:
                self.spot_ij_rounded = self.cameraslm.kxyslm_to_ijcam(self.spot_kxy_rounded)
            else:
                self.spot_ij_rounded = None
        else:
            self.spot_kxy_rounded = None
            self.spot_ij_rounded = None

        self.target[self.spot_knm_rounded[1, :], self.spot_knm_rounded[0, :]] = self.spot_amp
        self.target /= Hologram.norm(self.target)

        if reset_weights:
            cp.copyto(self.weights, self.target)

        if plot:
            self.plot_target()
    def update_target(self, reset_weights=False, plot=False):
        """
        From the spot locations stored in :attr:`spot_knm`, update the target pattern.

        Note
        ~~~~~~~~
        If there's a cameraslm, updates the :attr:`spot_ij_rounded` attribute
        corresponding to where pixels in the k-space where actually placed (due to rounding
        to integers, stored in :attr:`spot_knm_rounded`), rather the
        idealized floats :attr:`spot_knm`.

        Note
        ~~~~~~~~
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

    # TODO: add this.
    def correct_spots(self, img, basis="kxy"):
        """
        Fourier calibration is rarely perfect; this function hones the positions of the spots to
        the desired targets.

        Parameters
        ----------
        img : numpy.ndarray
            Image measured by the camera.
        basis : str
            The correction can be in any of the following bases:
            - ``"ij"`` changes the pixel that the spot is expected at,
            - ``"kxy"``, ``"knm"`` changes the k-vector which the SLM targets.
            Defaults to ``"kxy"`` if ``None``.

        Returns
        ----------
        numpy.ndarray
            Euclidian pixel error in the ``"ij"`` basis for each spot.
        """
        psf = 11 # TODO
        blur = 2*int(psf/8)+1

        regions = analysis.take(img, self.spot_ij, psf, centered=True, integrate=False)

        # Filter the images, but not the stack.
        sp_gaussian_filter1d(regions, blur, axis=1, output=regions)
        sp_gaussian_filter1d(regions, blur, axis=2, output=regions)

        shift_x = np.argmax(np.amax(regions, axis=1, keepdims=True), axis=2) - (psf-1)/2
        shift_y = np.argmax(np.amax(regions, axis=2, keepdims=True), axis=1) - (psf-1)/2
        shift_vector = np.vstack(shift_x, shift_y)
        shift_error = np.sqrt(np.square(shift_x) + np.square(shift_y))

        if basis is None or basis == "kxy" or basis == "knm":   # Don't modify any camera spots.
            self.spot_kxy = self.spot_kxy_rounded - self.cameraslm.ijcam_to_kxyslm(shift_vector)
            self.spot_knm = toolbox.convert_blaze_vector(
                self.spot_kxy, "kxy", "knm", self.cameraslm.slm, self.shape)
            self.update_target()
        elif basis == "ij":     # Don't modify any k-vectors.
            self.spot_ij = self.spot_ij - shift_vector
        else:
            raise Exception("Unrecognized basis '{}'.".format(basis))

        return shift_error
    def _update_weights(self):
        """
        Change :attr:`weights` to optimize towards the :attr:`target` using feedback from
        :attr:`amp_ff`, the computed farfield amplitude. This function also updates stats.

        Parameters
        ----------
        """
        feedback = self.flags['feedback']

        if feedback == "computational":
            self._update_weights_generic(self.weights, self.amp_ff, self.target, method=self.method)
        elif feedback == "experimental-spot":
            self.measure(basis="ij")

            psf = 3 # TODO: Fix
            feedback = analysis.take(self.img_ij, self.spot_ij, psf, centered=True, integrate=True)

            self._update_weights_generic(
                self.weights[self.spot_knm_rounded[1,:], self.spot_knm_rounded[0,:]],
                feedback, self.spot_amp, method=self.method)

    def _calculate_stats_spots(self, stats, stat_groups=[]):
        if 'computational_spot' in stat_groups:
            total = cp.sum(cp.square(self.amp_ff))
            stats['computational_spot'] = self._calculate_stats(
                self.amp_ff[self.spot_knm_rounded[1, :], self.spot_knm_rounded[0, :]],
                self.spot_amp, efficiency_compensation=False, total=total)
        if 'experimental_spot' in stat_groups:
            self.measure(basis="ij")

            psf = 3 # TODO: Fix
            feedback = analysis.take(  self.img_ij, self.spot_ij, psf,
                                            centered=True, integrate=True)

            total = cp.sum(self.img_ij)

            stats['experimental_spot'] = self._calculate_stats(
                np.sqrt(feedback), self.spot_amp, mp=np,
                efficiency_compensation=False, total=total)
    def update_stats(self, stat_groups=[]):
        """
        Calculate statistics corresponding to the desired ``stat_groups``.

        stat_groups : list of str
            Which groups or types of statistics to analyze.
        """
        stats = {}

        self._calculate_stats_computational(stats, stat_groups)
        self._calculate_stats_experimental(stats, stat_groups)
        self._calculate_stats_spots(stats, stat_groups)

        self._update_stats_dictionary(stats)
