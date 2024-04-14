from slmsuite.holography.algorithms._header import *
from slmsuite.holography.algorithms._stats import _HologramStats


class Hologram(_HologramStats):
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
        (see :meth:`~slmsuite.holography.algorithms.Hologram.optimize_gs`).
        This is of shape :attr:`shape`.
    amp_ff : numpy.ndarray OR cupy.ndarray OR None
        **Far-field** amplitude in the ``"knm"`` basis.
        Used for comparing this, the computational result, with the :attr:`target`.
        This is of shape :attr:`shape`.
    dtype : type
        Datatype for stored **near-** and **far-field** arrays, which are **all real**.
        The default is ``float32``.
    dtype_complex : type
        Some internal variables are complex. The complex numbers follow :mod:`numpy`
        `type promotion <https://numpy.org/doc/stable/reference/routines.fft.html#type-promotion>`_.
        Complex datatypes are derived from :attr:`dtype`:

         - ``float32`` -> ``complex64`` (default :attr:`dtype`)
         - ``float64`` -> ``complex128``

        ``float16`` is *not* recommended for :attr:`dtype` because ``complex32`` is not
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
            (see :meth:`~slmsuite.holography.algorithms.Hologram.optimize_gs()`).
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
            try:  # Check if slm_shape is a CameraSLM.
                if amp is None:
                    amp = slm_shape.slm._get_source_amplitude()
                    amp_shape = amp.shape
                slm_shape = slm_shape.slm.shape
            except:
                try:  # Check if slm_shape is an SLM
                    if amp is None:
                        amp = slm_shape._get_source_amplitude()
                        amp_shape = amp.shape
                    slm_shape = slm_shape.shape

                except:  # (int, int) case
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
                assert np.all(self.slm_shape == np.array(amp_shape)), (
                    "algorithms.py: The shape of amplitude (via `amp` or SLM) is not equal to the "
                    "shapes of the provided initial phase (`phase`) or SLM (via `target` or `slm_shape`)"
                )
            if phase is not None:
                assert np.all(self.slm_shape == np.array(phase_shape)), (
                    "algorithms.py: The shape of the inital phase (`phase`) is not equal to the "
                    "shapes of the provided amplitude (via `amp` or SLM) or SLM (via `target` or `slm_shape`)"
                )
            if slm_shape is not None:
                assert np.all(self.slm_shape == np.array(slm_shape)), (
                    "algorithms.py: The shape of SLM (via `target` or `slm_shape`) is not equal to the "
                    "shapes of the provided initial phase (`phase`) or amplitude (via `amp` or SLM)"
                )

            self.slm_shape = tuple(self.slm_shape)

        # 2) Initialize variables.
        # Save the data type.
        if dtype(0).nbytes == 4:
            self.dtype = np.float32
            self.dtype_complex = np.complex64
        elif dtype(0).nbytes == 8:
            self.dtype = np.float64
            self.dtype_complex = np.complex128
        else:
            raise ValueError(f"Data type {dtype} not supported.")

        # # Create a "pinned memory" array for optimized stats transfers off of the GPU.
        # self._stats_pinned = cp_zeros_pinned((5,))

        # Initialize and normalize near-field amplitude
        if amp is None:  # Uniform amplitude by default (scalar).
            self.amp = 1 / np.sqrt(np.prod(self.slm_shape))
        else:               # Otherwise, initialize and normalize.
            self.amp = cp.array(amp, dtype=self.dtype, copy=False)
            self.amp *= 1 / Hologram._norm(self.amp)

        # Initialize near-field phase
        self.reset_phase(phase)

        # Initialize target. reset() will handle weights.
        self._update_target(target, reset_weights=False)

        # Initialize everything else inside reset.
        self.reset(reset_phase=False, reset_flags=True)

        # Custom GPU kernels for speedy weighting.
        self._update_weights_generic_cuda_kernel = None
        if np != cp:
            try:
                self._update_weights_generic_cuda_kernel = cp.RawKernel(
                    CUDA_KERNELS,
                    'update_weights_generic'
                )
            except:
                pass

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
                self.phase = cp.random.uniform(-np.pi, np.pi, self.slm_shape, dtype=self.dtype)
        else:
            # Otherwise, cast as a cp.array with correct type.
            self.phase = cp.array(phase, dtype=self.dtype, copy=False)

    def reset_weights(self):
        """
        Resets the hologram weights to the :attr:`target` defaults.
        """
        # Copy from the target.
        self.weights = self.target.copy()

        if hasattr(self, "zero_weights"):
            self.zero_weights *= 0

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
            :class:`slmsuite.hardware.FourierSLM` or :class:`slmsuite.hardware.SLM` instead,
            and should pass this when using the ``precision`` parameter.
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
        # If slm_shape is actually a FourierSLM
        if hasattr(slm_shape, "slm"):
            cameraslm = slm_shape
            slm_shape = cameraslm.slm.shape
        # If slm_shape is actually a SLM
        elif hasattr(slm_shape, "shape"):
            cameraslm = lambda: 0
            cameraslm.slm = slm_shape
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
            padding_shape = np.power(2, np.ceil(np.log2(slm_shape)) + padding_order - 1).astype(int)
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
            if new_target.shape != self.shape:
                raise ValueError(
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
            Current nearfield phase of the optimization.
        """
        if cp != np:
            return self.phase.get() + np.pi
        return self.phase + np.pi

    def extract_farfield(self, affine=None, get=True):
        r"""
        Collects the current complex farfield from the GPU with :meth:`cupy.ndarray.get()`.

        Parameters
        ----------
        affine : dict
            Affine transformation to apply to far-field data (in the form of
            :attr:`~slmsuite.hardware.cameraslms.FourierSLM.fourier_calibration`).
        get : bool
            Whether or not to convert the cupy array to a numpy array if cupy is used.
            This is ignored if numpy is used.

        Returns
        -------
        numpy.ndarray
            Current farfield of the optimization.
        """
        nearfield = toolbox.pad(self.amp * cp.exp(1j * self.phase), self.shape)
        farfield = cp.fft.fftshift(cp.fft.fft2(cp.fft.fftshift(nearfield), norm="ortho"))
        self.amp_ff = cp.abs(farfield)
        self.phase_ff = cp.angle(farfield)

        if cp != np:
            if affine is not None:
                cp_affine_transform(
                    input=farfield,
                    matrix=affine["M"],
                    offset=affine["b"],
                    output_shape=self.shape,
                    order=3,
                    output=farfield,
                    mode="constant",
                    cval=0,
                )
            return farfield.get() if get else farfield
        else:
            if affine is not None:
                sp_affine_transform(
                    input=farfield,
                    matrix=affine["M"],
                    offset=affine["b"],
                    output_shape=self.shape,
                    order=3,
                    output=farfield,
                    mode="constant",
                    cval=0,
                )
            return farfield

    # Core optimization function.
    def optimize(
        self,
        method="GS",
        maxiter=20,
        verbose=True,
        callback=None,
        feedback=None,
        stat_groups=[],
        **kwargs,
    ):
        r"""
        Optimizers to solve the "phase problem": approximating the near-field phase that
        transforms a known near-field source amplitude to a desired far-field
        target amplitude.
        Supported optimization methods include:

        - Gerchberg-Saxton (GS) phase retrieval.

            - ``'GS'``

              `An iterative algorithm for phase retrieval
              <http://www.u.arizona.edu/~ppoon/GerchbergandSaxton1972.pdf>`_,
              accomplished by moving back and forth between the imaging and Fourier domains,
              with amplitude corrections applied to each.
              This is usually implemented using fast discrete Fourier transforms,
              potentially GPU-accelerated.

        - Weighted Gerchberg-Saxton (WGS) phase retrieval algorithms of various flavors.
          Improves the uniformity of GS-computed focus arrays using weighting methods and
          techniques from literature. The ``method`` keywords are:

            - ``'WGS-Leonardo'``

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

            - ``'WGS-Kim'``

              `Improves the convergence <https://doi.org/10.1364/OL.44.003178>`_
              of ``WGS-Leonardo`` by fixing the far-field phase
              strictly after a desired number of net iterations
              specified by ``"fix_phase_iteration"``
              or after exceeding a desired efficiency
              (fraction of far-field energy at the desired points)
              specified by ``"fix_phase_efficiency"``

            - ``'WGS-Nogrette'``

              Weights target intensities by `a tunable gain factor <https://doi.org/10.1103/PhysRevX.4.021034>`_.

              .. math:: \mathcal{W} = \mathcal{W}/\left(1 - f\left(1 - \mathcal{F}/\mathcal{T}\right)\right)

              where :math:`f` is the gain factor passed as ``"feedback_factor"`` in
              :attr:`~slmsuite.holography.algorithms.Hologram.flags` (see ``kwargs``).
              The factor :math:`f` defaults to .1 if not passed.

              Note that while Nogrette et al compares powers, this implementation
              compares amplitudes for speed. These are identical to first order.

            - ``'WGS-Wu'``

              `exponential <https://doi.org/10.1364/OE.413723>`_

              .. math:: \mathcal{W} = \mathcal{W}\exp\left( f (\mathcal{T} - \mathcal{F}) \right)

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
            print(
                "Optimizing with '{}' using the following method-specific flags:".format(
                    self.method
                )
            )
            pprint.pprint(
                {
                    key: value
                    for (key, value) in self.flags.items()
                    if key in ALGORITHM_DEFAULTS[method]
                }
            )
            print("", end="", flush=True)  # Prevent tqdm conflicts.

        # 2) Prepare the iterations iterable.
        iterations = range(maxiter)

        # 2.1) Decide whether to use a tqdm progress bar. Don't use a bar for maxiter == 1.
        if verbose and maxiter > 1:
            iterations = tqdm(iterations)

        # 3) Switch between optimization methods (currently only GS- or WGS-type is supported).
        if "GS" in method:
            self.optimize_gs(iterations, callback)

    # Optimization methods (currently only GS- or WGS-type is supported).
    def optimize_gs(self, iterations, callback):
        """
        GPU-accelerated Gerchberg-Saxton (GS) iterative phase retrieval.

        Solves the "phase problem": approximates the near-field phase that
        transforms a known near-field source amplitude to a desired far-field
        target amplitude.

        Caution
        ~~~~~~~
        This function should be called through :meth:`.optimize()` and not called
        directly. It is left as a public function exposed in documentation to clarify
        how the internals of :meth:`.optimize()` work.

        Note
        ~~~~
        Default FFTs are **not** in-place in this algorithm. In both non-:mod:`cupy` and
        :mod:`cupy` implementations, :mod:`numpy.fft` does not support in-place
        operations.  However, :mod:`scipy.fft` does in both. In the future, we may move to the scipy
        implementation. However, neither :mod:`numpy` or :mod:`scipy` ``fftshift`` support
        in-place movement (for obvious reasons). For even faster computation, algorithms should
        consider **not shifting** the FFT result, and instead shifting measurement data / etc to
        this unshifted basis. We might also implement `get_fft_plan
        <https://docs.cupy.dev/en/stable/reference/generated/cupyx.scipy.fftpack.get_fft_plan.html>`_
        for even faster FFTing. However, in practice, speed is limited by other
        peripherals (especially feedback and stats) rather than FFT speed or memory.

        Parameters
        ----------
        iterations : iterable
            Number of loop iterations to run. Is an iterable to pass a :mod:`tqdm` iterable.
        callback : callable OR None
            See :meth:`.optimize()`.
        """
        # Proxy to initialize nearfield with the correct shape and correct (complex) type.
        nearfield = cp.zeros(self.shape, dtype=self.dtype_complex)
        self.farfield = cp.zeros(self.target.shape, dtype=self.dtype_complex)    # Use target.shape for FreeSpotHologram cases.

        # Precompute MRAF helper variables.
        mraf_variables = self._mraf_helper_routines()

        # Helper variables for speeding up source phase and amplitude fixing.
        (i0, i1, i2, i3) = toolbox.unpad(self.shape, self.slm_shape)

        for _ in iterations:
            # (A) Nearfield -> farfield
            # Fix the relevant part of the nearfield amplitude to the source amplitude.
            # Everything else is zero because power outside the SLM is assumed unreflected.
            # This is optimized for when shape is much larger than slm_shape.
            nearfield.fill(0)
            nearfield[i0:i1, i2:i3] = self.amp * cp.exp(1j * self.phase)

            # FFT to move to the farfield.
            self.farfield = self._nearfield2farfield(nearfield, farfield_out=self.farfield)

            # (B) Midloop caching and prep
            # Before callback(), cleanup such that it can access updated amp_ff and images.
            self._midloop_cleaning(self.farfield)

            # Run step function if present and check termination conditions.
            if callback is not None:
                if callback(self):
                    break

            # Evaluate method-specific routines, stats, etc.
            # If you want to add new functionality to GS, do so here to keep the main loop clean.
            self._gs_farfield_routines(self.farfield, mraf_variables)

            # (C) Farfield -> nearfield.
            nearfield = self._farfield2nearfield(self.farfield, nearfield_out=nearfield)

            # Grab the phase from the complex nearfield.
            # Use arctan2() directly instead of angle() for in-place operations (out=).
            self.phase = cp.arctan2(
                nearfield.imag[i0:i1, i2:i3],
                nearfield.real[i0:i1, i2:i3],
                out=self.phase,
            )

            # Increment iteration.
            self.iter += 1

        # Update the final far-field
        nearfield.fill(0)
        nearfield[i0:i1, i2:i3] = self.amp * cp.exp(1j * self.phase)
        self.farfield = self._nearfield2farfield(nearfield, farfield_out=self.farfield)
        self.amp_ff = cp.abs(self.farfield)
        self.phase_ff = cp.angle(self.farfield)

    def _nearfield2farfield(self, nearfield, farfield_out=None):
        """TODO"""
        return cp.fft.fftshift(cp.fft.fft2(cp.fft.fftshift(nearfield), norm="ortho"))

    def _farfield2nearfield(self, farfield, nearfield_out=None):
        """TODO"""
        return cp.fft.ifftshift(cp.fft.ifft2(cp.fft.ifftshift(farfield), norm="ortho"))

    def _mraf_helper_routines(self):
        # MRAF helper variables
        noise_region = cp.isnan(self.target)
        mraf_enabled = bool(cp.any(noise_region))

        if not mraf_enabled:
            return {
                "mraf_enabled":False,
                "where_working":None,
                "signal_region":None,
                "noise_region":None,
                "zero_region":None,
            }

        zero_region = cp.abs(self.target) == 0
        Z = int(cp.sum(zero_region))
        if Z > 0 and not hasattr(self, "zero_weights"):
            self.zero_weights = cp.zeros((Z,), dtype=self.dtype_complex)

        signal_region = cp.logical_not(cp.logical_or(noise_region, zero_region))
        mraf_factor = self.flags.get("mraf_factor", None)
        # if mraf_factor is not None:
        #     if mraf_factor < 0:
        #         raise ValueError("mraf_factor={} should not be negative.".format(mraf_factor))

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
            "mraf_enabled":mraf_enabled,
            "where_working":where_working,
            "signal_region":signal_region,
            "noise_region":noise_region,
            "zero_region":zero_region,
        }

    def _midloop_cleaning(self, farfield):
        # 2.1) Cache amp_ff for weighting (if None, will init; otherwise in-place).
        self.amp_ff = cp.abs(farfield, out=self.amp_ff)

        # 2.2) Erase images from the past loop. FUTURE: Make better and faster.
        if hasattr(self, "img_ij"):
            self.img_ij = None
        if hasattr(self, "img_knm"):
            self.img_knm = None

    def _gs_farfield_routines(self, farfield, mraf_variables):
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
                            [not previous[-1 - i] for i in range(self.flags["fix_phase_iteration"])]
                        )
                        if contiguous_falses:
                            self.flags["fixed_phase"] = True

                # Save the phase if we are going from unfixed to fixed.
                if self.flags["fixed_phase"] and self.phase_ff is None or was_not_fixed:
                    self.phase_ff = cp.arctan2(farfield.imag, farfield.real, out=self.phase_ff)
            else:
                self.flags["fixed_phase"] = False

        mraf_enabled = mraf_variables["mraf_enabled"]

        # Fix amplitude, potentially also fixing the phase.
        if not mraf_enabled:
            # if ("fixed_phase" in self.flags and self.flags["fixed_phase"]):
            #     # Set the farfield to the stored phase and updated weights.
            #     cp.exp(1j * self.phase_ff, out=farfield)
            #     cp.multiply(farfield, self.weights, out=farfield)
            # else:
            #     # Set the farfield amplitude to the updated weights.
            #     cp.divide(farfield, cp.abs(farfield), out=farfield)
            #     cp.multiply(farfield, self.weights, out=farfield)
            #     cp.nan_to_num(farfield, copy=False, nan=0)

            if not ("fixed_phase" in self.flags and self.flags["fixed_phase"]):
                self.phase_ff = cp.arctan2(farfield.imag, farfield.real, out=self.phase_ff)

            cp.exp(1j * self.phase_ff, out=farfield)
            cp.multiply(farfield, self.weights, out=farfield)
        else:   # Mixed region amplitude freedom (MRAF) case.
            zero_region =   mraf_variables["zero_region"]
            noise_region =  mraf_variables["noise_region"]
            signal_region = mraf_variables["signal_region"]
            mraf_factor =   self.flags.get("mraf_factor", None)
            where_working = mraf_variables["where_working"]

            if hasattr(self, "zero_weights"):
                self.zero_weights -= self.flags.get("zero_factor", 1) * np.abs(farfield[zero_region]) * farfield[zero_region]
                farfield[zero_region] = self.zero_weights

            # # Handle signal and noise regions.
            # if ("fixed_phase" in self.flags and self.flags["fixed_phase"]):
            #     # Set the farfield to the stored phase and updated weights, in the signal region.
            #     if where_working:
            #         cp.exp(1j * self.phase_ff, where=signal_region, out=farfield)
            #         cp.multiply(farfield, self.weights, where=signal_region, out=farfield)
            #         if mraf_factor is not None: cp.multiply(farfield, mraf_factor, where=noise_region, out=farfield)
            #     else:
            #         cp.exp(1j * self.phase_ff, _where=signal_region, out=farfield)
            #         cp.multiply(farfield, self.weights, _where=signal_region, out=farfield)
            #         if mraf_factor is not None: cp.multiply(farfield, mraf_factor, _where=noise_region, out=farfield)
            # else:
            #     # Set the farfield amplitude to the updated weights, in the signal region.
            #     if where_working:
            #         cp.divide(farfield, cp.abs(farfield), where=signal_region, out=farfield)
            #         cp.multiply(farfield, self.weights, where=signal_region, out=farfield)
            #         if mraf_factor is not None: cp.multiply(farfield, mraf_factor, where=noise_region, out=farfield)
            #     else:
            #         cp.divide(farfield, cp.abs(farfield), _where=signal_region, out=farfield)
            #         cp.multiply(farfield, self.weights, _where=signal_region, out=farfield)
            #         if mraf_factor is not None: cp.multiply(farfield, mraf_factor, _where=noise_region, out=farfield)
            #     cp.nan_to_num(farfield, copy=False, nan=0)

            if not ("fixed_phase" in self.flags and self.flags["fixed_phase"]):
                self.phase_ff = cp.arctan2(farfield.imag, farfield.real, out=self.phase_ff)

            if where_working:
                cp.exp(1j * self.phase_ff, where=signal_region, out=farfield)
                cp.multiply(farfield, self.weights, where=signal_region, out=farfield)
                if mraf_factor is not None: cp.multiply(farfield, mraf_factor, where=noise_region, out=farfield)
            else:
                cp.exp(1j * self.phase_ff, _where=signal_region, out=farfield)
                cp.multiply(farfield, self.weights, _where=signal_region, out=farfield)
                if mraf_factor is not None: cp.multiply(farfield, mraf_factor, _where=noise_region, out=farfield)

    # Weighting functions.
    def _update_weights_generic(
            self, weight_amp, feedback_amp, target_amp=None, xp=cp, nan_checks=True
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
            Measured or result amplitudes corresponding to ``weight_amp``.
            Should be the same size as ``weight_amp``.
        target_amp : numpy.ndarray OR cupy.ndarray OR None
            Necessary in the case where ``target_amp`` is not uniform, such that the weighting can
            properly be applied to bring the feedback closer to the target. If ``None``, is assumed
            to be uniform. Should be the same size as ``weight_amp``.
        xp : module
            This function is used by both :mod:`cupy` and :mod:`numpy`, so we have the option
            for either. Defaults to :mod:`cupy`.
        nan_checks : bool
            Whether to enable checks to avoid division by zero or ``nan`` infiltration.

        Returns
        -------
        numpy.ndarray OR cupy.ndarray
            The updated ``weight_amp``.
        """
        if self._update_weights_generic_cuda_kernel is None or xp == np:
            return self._update_weights_generic_cupy(weight_amp, feedback_amp, target_amp, xp, nan_checks)
        else:
            return self._update_weights_generic_cuda(weight_amp, feedback_amp, target_amp)


    def _update_weights_generic_cupy(
            self, weight_amp, feedback_amp, target_amp=None, xp=cp, nan_checks=True
        ):
        assert self.method[:4] == "WGS-", "For now, assume weighting is for WGS."
        method = self.method[4:].lower()

        # Parse feedback_amp
        if target_amp is None:  # Uniform
            feedback_corrected = xp.array(feedback_amp, copy=True, dtype=self.dtype)
        else:  # Non-uniform
            feedback_corrected = xp.array(feedback_amp, copy=True, dtype=self.dtype)
            feedback_corrected *= 1 / Hologram._norm(feedback_corrected, xp=xp)

            xp.divide(feedback_corrected, xp.array(target_amp, copy=False), out=feedback_corrected)

            if nan_checks:
                feedback_corrected[feedback_corrected == np.inf] = 1
                feedback_corrected[xp.array(target_amp, copy=False) == 0] = 1

                xp.nan_to_num(feedback_corrected, copy=False, nan=1)

        # Fix feedback according to the desired method.
        if "leonardo" in method or "kim" in method:
            # 1/(x^p)
            xp.power(feedback_corrected, -self.flags["feedback_exponent"], out=feedback_corrected)
        elif "nogrette" in method.lower():
            # Taylor expand 1/(1-g(1-x)) -> 1 + g(1-x) + (g(1-x))^2 ~ 1 + g(1-x)
            feedback_corrected *= -(1 / xp.nanmean(feedback_corrected))
            feedback_corrected += 1
            feedback_corrected *= -self.flags["feedback_factor"]
            feedback_corrected += 1
            xp.reciprocal(feedback_corrected, out=feedback_corrected)
        # elif "wu" in method:
        #     feedback = np.exp(self.flags["feedback_exponent"] * (target - feedback))
        # elif "tanh" in method:
        #     feedback = self.flags["feedback_factor"] * np.tanh(self.flags["feedback_exponent"] * (target - feedback))
        else:
            raise RuntimeError(
                "Method " "{}" " not recognized by Hologram.optimize()".format(self.method)
            )

        if nan_checks:
            feedback_corrected[feedback_corrected == np.inf] = 1

        # Update the weights.
        weight_amp *= feedback_corrected

        if nan_checks:
            xp.nan_to_num(weight_amp, copy=False, nan=0.0001)
            # weight_amp[weight_amp == np.inf] = 1

        # Normalize amp, as methods may have broken conservation.
        weight_amp *= (1 / Hologram._norm(weight_amp, xp=xp))

        return weight_amp

    def _update_weights_generic_cuda(self, weight_amp, feedback_amp, target_amp=None):
        method = ALGORITHM_INDEX[self.method]

        if target_amp is None:  # Uniform
            feedback_norm = 0
            target_amp = 0
        else:
            feedback_norm = Hologram._norm(feedback_amp, xp=cp)

        N = weight_amp.size
        method = ALGORITHM_INDEX[self.method]

        threads_per_block = int(
            self._update_weights_generic_cuda_kernel.max_threads_per_block
        )
        blocks = N // threads_per_block

        # Call the RawKernel.
        self._update_weights_generic_cuda_kernel(
            (blocks,),
            (threads_per_block,),
            (
                weight_amp,
                feedback_amp,
                target_amp,
                N,
                method,
                feedback_norm,
                self.flags.pop("feedback_exponent", 0),     # TODO: fix
                self.flags.pop("feedback_factor", 0)
            )
        )

        weight_amp *= (1 / Hologram._norm(weight_amp, xp=cp))

        return weight_amp

    def _update_weights(self):
        """
        Change :attr:`weights` to optimize towards the :attr:`target` using feedback from
        :attr:`amp_ff`, the computed farfield amplitude.
        """
        feedback = self.flags["feedback"]

        if feedback == "computational":
            self._update_weights_generic(self.weights, self.amp_ff, self.target)

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
                    mempool.get_limit() / (1024.0**3)
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
    def _norm(matrix, xp=cp):
        r"""
        Computes the root of the sum of squares of the given ``matrix``. Implements

        .. math:: \sqrt{\iint |\vec{E}|^2}.

        Parameters
        ----------
        matrix : numpy.ndarray OR cupy.ndarray
            Data, potentially complex.
        xp : module
            This function is used by both :mod:`cupy` and :mod:`numpy`, so we have the option
            for either. Defaults to :mod:`cupy`.

        Returns
        -------
        float
            The result.
        """
        if xp.iscomplexobj(matrix):
            return xp.sqrt(xp.nansum(xp.square(xp.abs(matrix))))
        else:
            return xp.sqrt(xp.nansum(xp.square(matrix)))
