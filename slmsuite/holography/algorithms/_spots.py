from slmsuite.holography.algorithms._header import *
from slmsuite.holography.algorithms._hologram import Hologram
from slmsuite.holography.algorithms._feedback import FeedbackHologram


class _AbstractSpotHologram(FeedbackHologram):
    """
    Abstract class to eventally handle :meth:`SpotHologram.refine_offset()`
    and other shared methods for :class:`SpotHologram` and :class:`CompressedSpotHologram`.
    There are many parts of :class:`SpotHologram` with repetition and bloat that
    can be simplified with more modern features from other parts of :mod:`slmsuite`.
    """
    pass
    # def update_spots(self, source_basis="kxy"):
    #     pass


class CompressedSpotHologram(_AbstractSpotHologram):
    """
    Holography optimized for the generation of optical focal arrays, making use of
    kernels not bound to the grid of a discrete Fourier transform.

    Is a subclass of :class:`FeedbackHologram`, but falls back to non-camera-feedback
    routines if :attr:`cameraslm` is not passed.

    Attributes
    ----------
    spot_zernike : numpy.ndarray OR cupy.ndarray of float
        Spot position vectors with shape ``(D, N)``.

        Important
        ~~~~~~~~~
        When :mod:`cupy` is enabled, these spots are hosted on the GPU as ``cupy.ndarray``.
        Modifications or updates to the data should take this type into account.
    zernike_basis : numpy.ndarray
        The ind
    spot_kxy : numpy.ndarray of float
        Spot position vectors in the normalized basis with shape ``(2, N)`` or ``(3, N)``.
    spot_ij : array_like of float OR None
        Lateral spot position vectors in the camera basis with shape ``(2, N)``.
    external_spot_amp : array_like of float
        When using ``"external_spot"`` feedback or the ``"external_spot"`` stat group,
        the user must supply external data. This data is transferred through this
        attribute. For iterative feedback, have the ``callback()`` function set
        :attr:`external_spot_amp` dynamically. By default, this variable is set to even
        distribution of amplitude.
    spot_integration_width_ij : int
        For spot-specific feedback methods, better SNR is achieved when integrating over
        many camera pixels. This variable stores the width of the integration region
        in ``"ij"`` (camera) space.
    cuda : bool
        Whether the custom CUDA kernel is used for optimization (option 2).
    """
    def __init__(
        self,
        spot_vectors,
        basis="kxy",
        spot_amp=None,
        cameraslm=None,
        **kwargs
    ):
        r"""
        Initializes a :class:`CompressedSpotHologram` targeting given spots at ``spot_vectors``.
        This class makes use of so-called 'compressed' methods. Instead of
        effectively evaluating a blazing nearfield-farfield transformation kernel
        :math:`\phi(x,y,k_x,k_y) = k_xx + k_yy` at every
        point in the grid of a discrete Fourier transform, this class 'compresses' the
        effort to evaluate such kernels **only when the target farfield amplitude is
        non-zero.**

        For focal arrays, the majority of the DFT grid is zero, so only
        processing the non-zero points can be faster. Importantly, each kernel is
        no longer bound to a grid, so :math:`k_x, k_y` can be defined as free floating
        points. More importantly, we are not restricted to the linear shearing that the
        standard Fourier kernel imposes. Rather, quadratic focusing terms or more
        complicated summations of Zernike polynomials can be employed to focus or
        correct for aberration.

        Important
        ---------
        This class supports three different options for generating compressed spot
        arrays.

        #.  First, a :mod:`numpy`/:mod:`cupy` method is implemented.
            This method makes ample use of memory and is inefficient for very large spot arrays.
            This method works by playing tricks to vectorize operations onto a cached kernel.
            To update the cached kernel with new spot positions in :attr:`spot_kxy`,
            set :attr:`_cupy_kernel' to ``None``.
            More general or complicated functionality requires full use of the GPU
            and thus the following methods require :mod:`cupy`.

        #.  A custom CUDA kernel loaded into :mod:`cupy` focussing on arrays of
            spots, **each spot having an individualized Zernike calibration**,
            which of course includes steering in the :math:`x`, :math:`y`, and :math:`z`
            directions with the 2nd, 1st, and 4th Zernike polynomials, respectively (tilt and focus).
            Saving and transporting a set of Zernike polynomials up to a large order would
            consume unacceptable  amounts of memory and memory bandwidth.
            Instead, this kernel dynamically constructs all the zernike polynomials in the
            given basis locally on the GPU, using this data to
            apply the nearfield-farfield transformation before returning only the result
            of the transformation. Such computation requires only the data of the input
            and output, along with efficient 'compressed' information defining the
            construction of the polynomial kernels.
            This kernel bases itself upon :attr:`spot_zernike`, which is moved to the
            GPU at runtime as a ``cupy.ndarray``. This list can be updated

        The chosen option is selected dynamically. Option 2 is the highest preference.
        If the kernel fails to load or :mod:`cupy` is unavailable, the option will downgrade
        to option 1. The choice is stored in :attr:`cuda`.

        Parameters
        ----------
        spot_vectors : array_like
            Spot position vectors with shape ``(D, N)``,
            where ``D`` is the dimension of the parameters of each spot.
        basis : str OR array_like of int
            The spots can be in any of the following bases:

            -   ``"kxy"`` for centered normalized SLM :math:`k`-space (radians),
                for dimension ``D`` of 2 and 3. This is the default.

            -   ``"ij"`` for camera coordinates (pixels),
                for dimension ``D`` of 2 and 3.

            -   ``"zernike"`` for applying Zernike terms to each spot (radians),
                for dimension ``D`` equal to the length of ``zernike_basis``.
                The provided coefficients are multiplied directly on the the normalized
                Zernike polynomials on the unit disk.
                See :meth:`~slmsuite.holography.toolbox.phase.zernike_sum()`.

                Important
                ~~~~~~~~~
                The assumed Zernike basis depends on the dimensionality of the provided spots:

                -   If ``D == 2``, then the basis is assumed to be ``[2,1]``
                    corresponding to the :math:`x = Z_2 = Z_1^1`
                    and :math:`y = Z_1 = Z_1^{-1}` tilt terms.

                -   If ``D == 3``, then the basis is assumed to be ``[2,1,4]``
                    corresponding to the previous, with the addition of the
                    :math:`Z_4 = Z_2^0` focus term.

                -   If ``D > 3``, then the basis is assumed to be ``[1,...,D]``.
                    The piston (Zernike index 0) term is ignored as this constant phase is
                    not relevant.

                See the next option to customize the Zernike basis beyond these defaults.

            -   ``array_like of int`` for applying a custom Zernike basis.
                List of ``D`` indices corresponding to Zernike polynomials using ANSI indexing.
                See :meth:`~slmsuite.holography.toolbox.phase.convert_zernike_index()`.
                The index ``-1`` (outside Zernike indexing) is used as a special case to add
                a vortex waveplate with amplitude :math:`2\pi` to the system
                (see :meth:`~slmsuite.holography.toolbox.phase.laguerre_gaussian()`).

        spot_amp : array_like OR None
            The amplitude to target for each spot. See :attr:`spot_amp`.
            If ``None``, all spots are assumed to have the same amplitude.
            Normalization is performed automatically; the user is not required to
            normalize.
            MRAF functionality still works by setting elements of ``spot_amp``
            to ``np.nan``, denoting 'noise' points where amplitude can be dumped.
        cameraslm : slmsuite.hardware.cameraslms.FourierSLM
            Must be passed. The default of ``None`` with throw an error and is only
            there such that we can retain the same order as :class:`SpotHologram`.
        **kwargs
            Passed to :meth:`.FeedbackHologram.__init__()`.
        """
        # Parse vectors.
        spot_vectors = toolbox.format_vectors(spot_vectors, handle_dimension="pass")
        (D, N) = spot_vectors.shape

        # Parse spot_amp.
        if spot_amp is not None:
            self.spot_amp = np.array(spot_amp, copy=None)
            if self.spot_amp.size != N:
                raise ValueError("spot_amp must have the same length as the provided spots.")
        else:
            self.spot_amp = np.full(N, 1.0 / np.sqrt(N))

        # Parse zernike_basis.
        if isinstance(basis, str):
            # Assume the ANSI basis (slmsuite default), but start without piston.
            if D == 2:
                self.zernike_basis = np.array([2,1])
            elif D == 3:
                self.zernike_basis = np.array([2,1,4])
            else:
                self.zernike_basis = np.arange(1, D+1)
        else:
            # Make sure we are a 1D list.
            self.zernike_basis = np.ravel(basis)
            basis = "zernike"
            if len(self.zernike_basis) != D:
                raise ValueError("zernike_basis must have the same dimension as the provided spots.")

            # Warn the user that the piston is useless.
            if 0 in self.zernike_basis:
                warnings.warn(
                    "Found ANSI index '0' (Zernike piston) in the zernike_basis, "
                    "but spot phase is controlled externally."
                )

        # Make some helper variables to use the zernike_basis.
        if not np.any(self.zernike_basis == 2) or not np.any(self.zernike_basis == 1):
            raise ValueError("Compressed basis must include x, y (Zernike ANSI indices 2, 1)")
        self.zernike_basis_cartesian = [
            np.argwhere(self.zernike_basis == 2)[0], np.argwhere(self.zernike_basis == 1)[0]
        ]
        if np.any(self.zernike_basis == 4):
            self.zernike_basis_cartesian.append(np.argwhere(self.zernike_basis == 4)[0])

        # Parse spot_vectors.
        if basis == "zernike":
            self.spot_zernike = spot_vectors
            self.spot_kxy = toolbox.convert_vector(
                spot_vectors[self.zernike_basis_cartesian, :],  # Special case to crop the basis.
                from_units="zernike",
                to_units="kxy",
                hardware=cameraslm
            )
        else:
            self.spot_zernike = toolbox.convert_vector(
                spot_vectors,
                from_units=basis,
                to_units="zernike",
                hardware=cameraslm
            )
            self.spot_kxy = toolbox.convert_vector(
                spot_vectors,
                from_units=basis,
                to_units="kxy",
                hardware=cameraslm
            )
            self.spot_ij = toolbox.convert_vector(
                spot_vectors,
                from_units=basis,
                to_units="ij",
                hardware=cameraslm
            )

        # Check to make sure spots are within bounds
        kmax = 1    # TODO: replace with correct value.
        if np.any(np.abs(self.spot_kxy[self.zernike_basis_cartesian[:2]]) > kmax):
            raise ValueError("Spots laterally outside the bounds of the farfield")

        # Generate ij point spread function (psf)
        if cameraslm is not None:
            psf_kxy = np.mean(cameraslm.slm.spot_radius_kxy())
            self.spot_ij = cameraslm.kxyslm_to_ijcam(self.spot_kxy)
            psf_ij = toolbox.convert_radius(psf_kxy, "kxy", "ij", cameraslm)
        else:
            psf_ij = np.nan
            self.spot_ij = None

        if np.isnan(psf_ij): psf_ij = 0

        # Use semi-arbitrary values to determine integration widths. The default width is:
        #  - six times the psf,
        #  - but then clipped to be:
        #    + larger than 3 and
        #    + smaller than the minimum inf-norm distance between spots divided by 1.5
        #      (divided by 1 would correspond to the largest non-overlapping integration
        #      regions; 1.5 gives comfortable padding)
        #  - and finally forced to be an odd integer.
        min_psf = 3

        # if self.spot_ij is not None:
        #     dist_ij = np.max([toolbox.smallest_distance(self.spot_ij) / 1.5, min_psf])
        #     if psf_ij > dist_ij:
        #         warnings.warn(
        #             "The expected camera spot point-spread-function is too large. "
        #             "Clipping to a smaller "
        #         )
        #     self.spot_integration_width_ij = np.clip(6 * psf_ij, 3, dist_ij)
        #     self.spot_integration_width_ij =  int(2 * np.floor(self.spot_integration_width_ij / 2) + 1)

        #     cam_shape = cameraslm.cam.shape

        #     if (
        #         np.any(self.spot_ij[0] < self.spot_integration_width_ij / 2) or
        #         np.any(self.spot_ij[1] < self.spot_integration_width_ij / 2) or
        #         np.any(self.spot_ij[0] >= cam_shape[1] - self.spot_integration_width_ij / 2) or
        #         np.any(self.spot_ij[1] >= cam_shape[0] - self.spot_integration_width_ij / 2)
        #     ):
        #         raise ValueError(
        #             "Spots outside camera bounds!\nSpots:\n{}\nBounds: {}".format(
        #                 self.spot_ij, cam_shape
        #             )
        #         )
        # else:
        self.spot_integration_width_ij = None

        # Initialize target/etc with fake shape.
        super().__init__(shape=(1,1), target_ij=None, cameraslm=cameraslm, **kwargs)

        # Replace the fake shape with the SLM shape.
        self.shape = self.slm_shape
        self.reset()

        # Fill the target with data.
        self.update_target(new_target=spot_amp, reset_weights=True)

        # Set the external amp variable to be perfect by default.
        self.external_spot_amp = np.ones(self.target.shape)

        # Default helper variables.
        self._cupy_kernel = None
        self._cupy_stack = None
        self.cuda = False

        # Storage variable to use cp.sum on the intermediate sum.
        self._nearfield2farfield_cuda_intermediate = None

        if np != cp:
            # Move the Zernike spots to the GPU.
            self.spot_zernike = cp.array(self.spot_zernike, dtype=self.dtype)

            # Custom GPU kernels for speed.
            try:
                self._near2far_cuda = cp.RawKernel(
                    CUDA_KERNELS, 
                    'compressed_nearfield2farfield_v2',
                    translate_cucomplex=True,
                    jitify=True,
                )
                self._far2near_cuda = cp.RawKernel(
                    CUDA_KERNELS, 
                    'compressed_farfield2nearfield_v2',
                    translate_cucomplex=True,
                    jitify=True,
                )

                self._near2far_cuda.compile()
                self._far2near_cuda.compile()

                c_md, i_md, pxy_m = tphase._zernike_populate_basis_map(self.zernike_basis)
                self._c_md = cp.array(c_md)
                self._i_md = cp.array(i_md)
                self._pxy_m = cp.array(pxy_m)

                self.cuda = True

                # Test the kernel.
                self._farfield2nearfield(self._nearfield2farfield(cp.full(self.slm_shape, 1j, dtype=self.dtype_complex)))
            except Exception as e:
                raise e
                warnings.warn("Raw CUDA kernels failed to load. Falling back to cupy.\n" + str(e))

    def __len__(self):
        """
        Overloads ``len()`` to return the number of spots in this :class:`FreeSpotHologram`.

        Returns
        -------
        int
            The length of :attr:`spot_amp`.
        """
        return self.spot_amp.size

    def calculate_padded_shape(self):
        """
        Vestigial from :class:`~slmsuite.holography.algorithms.Hologram`, but unneeded here.
        :class:`~slmsuite.holography.algorithms.CompressedSpotHologram`
        does not use a DFT grid and does not need padding.
        """
        raise NameError("CompressedSpotHologram does not use a DFT grid and does not need padding.")

    def refine_offset(self, *args, **kwargs):
        raise NotImplementedError("Currently not implemented for CompressedSpotHologram")

    # Projection backend helper functions.
    def _build_stack(self, D=2):
        r"""
        Builds the coordinate stack, which is a stack of the
        :math:`2\pi i x`, :math:`2\pi i y`, and
        (if desired) :math:`\pi i r^2 = \pi i (x^2 + y^2)` coordinates corresponding
        to the position of each pixel. The :math:`i\pi` factors are to make compute of
        the kernels trivial, i.e. such that these factors (and squares) do not have
        be calculated at each iteration.
        """
        X = (2 * cp.pi) * cp.array(self.cameraslm.slm.grid[0], dtype=self.dtype_complex).ravel()
        Y = (2 * cp.pi) * cp.array(self.cameraslm.slm.grid[1], dtype=self.dtype_complex).ravel()

        if D == 2:      # 2D spots
            self._cupy_stack = cp.stack((X, Y), axis=-1)       # Shape (H*W, 2)
        elif D == 3:    # 3D spots
            # Currently restricted to non-cylindrical focusing.
            # This is a good assumption if the SLM has square pixel size
            # and the optical train does not include cylindrical optics.
            # RR = pi * (x_grid ^2 + y_grid ^ 2)
            RR = (1 / (4 * cp.pi)) * (cp.square(X) + cp.square(Y))
            self._cupy_stack = cp.stack((X, Y, RR), axis=-1)   # Shape (H*W, 3)
        else:
            raise ValueError(f"Expected spots to be 2D or 3D. Found {D}D")

        # Rotate to imaginary afterward (not before so we can square X and Y when calculating RR).
        self._cupy_stack *= 1j

    def _build_kernel_batched(self, spot_kxy, out=None):
        """
        Uses the coordinate stack to produce the kernel, a stack of images corresponding
        to the blaze and lens directing power to each desired spot in the farfield.

        Parameters
        ----------
        spot_kxy : numpy.ndarray
            Vector locations of the spot, in two or three dimensions.
            See :attr:`spot_kxy`.
        out : numpy.ndarray OR None
            Array to direct data to when in-place. None if out-of-place.
        """
        (D, N) = spot_kxy.shape                         # Shape (2|3, N)

        # Parse stack.
        if self._cupy_stack is None:
            self._build_stack(D)                        # Shape (H*W, 2|3)

        # Parse out.
        out_shape = (np.prod(self.slm_shape), N)        # Shape (H*W, N)
        if out is None:
            out = cp.zeros(out_shape, dtype=self.dtype_complex)
        if out.shape != out_shape:
            raise ValueError(f"out shape {out.shape} does not matched the expected {out_shape}")

        # Evaluate the result in a (hopefully) memory and compute efficient way.
        out = cp.matmul(self._cupy_stack, spot_kxy, out=out)  # (H*W, 2|3) x (2|3, N) = (H*W, N)

        # Convert from phase to complex amplitude.
        out = cp.exp(out, out=out)

        return out

    def _nearfield2farfield(self, nearfield, farfield_out=None):
        """
        Maps the ``(H,W)`` nearfield (complex value on the SLM)
        onto the ``(N,1)`` farfield (complex value for each spot).
        """
        # print("n2f")
        if farfield_out is None:
            farfield_out = cp.zeros((len(self),), dtype=self.dtype_complex)

        if self.cuda:
            try:
                farfield_out = self._nearfield2farfield_cuda_v2(nearfield, farfield_out)
            except Exception as err:    # Fallback to cupy upon error.
                warnings.warn("Falling back to cupy:\n" + str(err))
                self.cuda = False

        if not self.cuda:
            farfield_out = self._nearfield2farfield_cupy(nearfield, farfield_out)

        return farfield_out

    def _nearfield2farfield_cuda(self, nearfield, farfield_out):
        H, W = self.slm_shape
        D, N = self.spot_kxy.shape

        threads_per_block = int(1024)
        assert self._near2far_cuda.max_threads_per_block >= threads_per_block
        if self._near2far_cuda.max_threads_per_block > threads_per_block:
            warnings.warn(
                "Threads per block can be larger than the hardcoded limit of 1024. "
                "Remove this limit for enhanced speed."
            )
        blocks_x = int(np.ceil(float(W*H) / threads_per_block))
        blocks_y = N

        print((blocks_x, blocks_y))

        if self._nearfield2farfield_cuda_intermediate is None:
            self._nearfield2farfield_cuda_intermediate = cp.zeros((blocks_y, blocks_x), dtype=self.dtype_complex)

        spot_kxy_float = cp.array(self.spot_kxy, self.dtype)    # TODO: Make this faster.

        # Call the RawKernel.
        self._near2far_cuda(
            (blocks_x, blocks_y),
            (threads_per_block, 1),
            (
                nearfield.ravel(),
                W, H, N, D,
                spot_kxy_float,
                0, 0, float(self.cameraslm.slm.pitch[0]), float(self.cameraslm.slm.pitch[1]),
                self._nearfield2farfield_cuda_intermediate.ravel()
            )
        )

        print(W*H)
        print(self._cupy_stack.shape)
        print(self._cupy_stack[0, :])
        print(self._nearfield2farfield_cuda_intermediate)
        print(cp.abs(self._nearfield2farfield_cuda_intermediate))
        print(self._nearfield2farfield_cuda_intermediate.shape)
        print(farfield_out.shape)

        # Sum over all the blocks to get the final answers using optimized cupy methods.
        farfield_out = cp.sum(self._nearfield2farfield_cuda_intermediate, axis=1, out=farfield_out)
        farfield_out *= (1 / Hologram._norm(farfield_out, xp=cp))

        return farfield_out

    def _nearfield2farfield_cuda_v2(self, nearfield, farfield_out):
        H, W = self.shape
        D, N = self.spot_zernike.shape
        M = self._i_md.shape[0]

        threads_per_block = int(1024)
        assert self._near2far_cuda.max_threads_per_block >= threads_per_block
        if self._near2far_cuda.max_threads_per_block > threads_per_block:
            warnings.warn(
                "Threads per block can be larger than the hardcoded limit of 1024. "
                "Remove this limit for enhanced speed."
            )
        blocks_x = int(np.ceil(float(W*H) / threads_per_block))
        blocks_y = N

        # print((blocks_x, blocks_y))

        if self._nearfield2farfield_cuda_intermediate is None:
            self._nearfield2farfield_cuda_intermediate = cp.zeros((blocks_y, blocks_x), dtype=self.dtype_complex)

        center_pix = np.array(self.cameraslm.slm.get_source_center())
        pitch_zernike = np.array(self.cameraslm.slm.pitch) * self.cameraslm.slm.get_source_zernike_scaling()

        # print(W, H, N, D, M)
        # print(
        #     self.spot_zernike.shape, # a_nd
        #     self._c_md.shape,
        #     self._i_md.shape,
        #     self._pxy_m.shape,
        # )
        # print(
        #     type(self.spot_zernike), # a_nd
        #     type(self._c_md),
        #     type(self._i_md),
        #     type(self._pxy_m),
        # )

        # print(self.iter)

        self._nearfield2farfield_cuda_intermediate.fill(-1)

        # print(
        #     nearfield.ravel(),
        #     W, H, N, D, M,
        #     self.spot_zernike.T,    # a_nd
        #     self._c_md,
        #     self._i_md,
        #     self._pxy_m,
        #     float(center_pix[0]), float(center_pix[1]),
        #     float(pitch_zernike[0]), float(pitch_zernike[1]),
        #     self._nearfield2farfield_cuda_intermediate.ravel()
        # )

        # Call the RawKernel.
        self._near2far_cuda(
            (blocks_x,),
            (threads_per_block, 1),
            (
                nearfield.ravel(),
                W, H, N, D, M,
                self.spot_zernike.T,    # a_nd
                self._c_md,
                self._i_md,
                self._pxy_m,
                np.float32(center_pix[0]), np.float32(center_pix[1]),
                np.float32(pitch_zernike[0]), np.float32(pitch_zernike[1]),
                self._nearfield2farfield_cuda_intermediate.ravel()
            )
        )

        # print(W*H)
        # print(self._cupy_stack.shape)
        # print(self._cupy_stack[0, :])
        # print(self._nearfield2farfield_cuda_intermediate)
        # print(cp.abs(self._nearfield2farfield_cuda_intermediate))
        # print(self._nearfield2farfield_cuda_intermediate.shape)
        # print(farfield_out.shape)

        # Sum over all the blocks to get the final answers using optimized cupy methods.
        farfield_out = cp.sum(self._nearfield2farfield_cuda_intermediate, axis=1, out=farfield_out)
        farfield_out *= (1 / Hologram._norm(farfield_out, xp=cp))
        
        # print("done")

        return farfield_out

    def _nearfield2farfield_cupy(self, nearfield, farfield_out):
        # Conjugate the nearfield to properly take the overlap integral.
        # FYI: Nearfield shape is (H,W)
        nearfield = cp.conj(nearfield, out=nearfield)

        N = self.spot_kxy.shape[1]
        N_batch_max = 500

        if farfield_out is None:
            farfield_out = cp.zeros((N, ), dtype=self.dtype_complex)

        # Do some prep work
        def collapse_kernel(kernel, out):
            # (1,H*W) x (H*W, N) = (N,1)^T
            cp.matmul(nearfield.ravel()[np.newaxis, :], kernel, out=out[np.newaxis, :])

        if self._cupy_kernel is None:
            self.spot_kxy_complex = cp.array(self.spot_kxy, copy=None, dtype=self.dtype_complex)

        # Evaluate the kernel.
        if N <= N_batch_max:
            if self._cupy_kernel is None:
                self._cupy_kernel = self._build_kernel_batched(self.spot_kxy_complex, out=self._cupy_kernel)

            collapse_kernel(self._cupy_kernel, out=farfield_out)
        else:
            if self._cupy_kernel is None:
                self._cupy_kernel = cp.zeros((np.prod(self.slm_shape), N_batch_max), dtype=self.dtype_complex)

            batches = 1 + N // N_batch_max
            for batch in range(batches):
                batch_slice = slice(batch * N_batch_max, np.clip((batch+1) * N_batch_max, 0, N))
                kernel_slice = slice(0, batch_slice.stop - batch_slice.start)
                self._cupy_kernel[:, kernel_slice] = self._build_kernel_batched(
                    self.spot_kxy_complex[:, batch_slice],
                    out=self._cupy_kernel[:, kernel_slice]
                )

                collapse_kernel(self._cupy_kernel[:, kernel_slice], out=farfield_out[batch_slice])

        # Normalize.
        farfield_out *= (1 / Hologram._norm(farfield_out, xp=cp))
        farfield_out = cp.conj(farfield_out, out=farfield_out)

        # Unconjugate nearfield (leave it unchanged).
        nearfield = cp.conj(nearfield, out=nearfield)

        return farfield_out

    def _farfield2nearfield(self, farfield, nearfield_out=None):
        """
        Maps the ``(N,1)`` farfield (complex value for each spot)
        onto the ``(H,W)`` nearfield (complex value on the SLM).
        """
        if nearfield_out is None:
            nearfield_out = cp.zeros(self.slm_shape, dtype=self.dtype_complex)

        if self.cuda:
            try:
                nearfield_out = self._farfield2nearfield_cuda_v2(farfield, nearfield_out)
            except Exception as err:    # Fallback to cupy upon error.
                warnings.warn("Falling back to cupy:\n" + str(err))
                self.cuda = False

        if not self.cuda:
            nearfield_out = self._farfield2nearfield_cupy(farfield, nearfield_out)

        return nearfield_out

    def _farfield2nearfield_cuda(self, farfield, nearfield_out):
        WH = int(self.slm_shape[0] * self.slm_shape[1])
        N = int(self.spot_kxy.shape[1])
        D = int(self.spot_kxy.shape[0])

        if self._cupy_stack is None:
            self._build_stack(D)

        threads_per_block = int(self._far2near_cuda.max_threads_per_block)
        blocks = WH // threads_per_block

        spot_kxy_float = cp.array(self.spot_kxy, self.dtype)    # TODO: Make this faster.

        # Call the RawKernel.
        self._far2near_cuda(
            (blocks,),
            (threads_per_block,),
            (
                farfield,
                WH, N, D,
                spot_kxy_float,
                self._cupy_stack[:, 0],
                self._cupy_stack[:, 1],
                self._cupy_stack[:, 2] if D == 3 else 0,
                0, 0, 0,
                nearfield_out.ravel()
            )
        )

        return nearfield_out

    def _farfield2nearfield_cuda_v2(self, farfield, nearfield_out):
        H, W = self.shape
        D, N = self.spot_zernike.shape
        M = self._i_md.shape[0]

        threads_per_block = int(1024)
        assert self._near2far_cuda.max_threads_per_block >= threads_per_block
        if self._near2far_cuda.max_threads_per_block > threads_per_block:
            warnings.warn(
                "Threads per block can be larger than the hardcoded limit of 1024. "
                "Remove this limit for enhanced speed."
            )
        blocks_x = int(np.ceil(float(W*H) / threads_per_block))

        center_pix = np.array(self.cameraslm.slm.get_source_center())
        pitch_zernike = np.array(self.cameraslm.slm.pitch) * self.cameraslm.slm.get_source_zernike_scaling()

        # Call the RawKernel.
        self._far2near_cuda(
            (blocks_x,),
            (threads_per_block, 1),
            (
                farfield,
                W, H, N, D, M,
                self.spot_zernike.T,    # a_nd
                self._c_md,
                self._i_md,
                self._pxy_m,
                np.float32(center_pix[0]), np.float32(center_pix[1]),
                np.float32(pitch_zernike[0]), np.float32(pitch_zernike[1]),
                nearfield_out
            )
        )

        return nearfield_out

    def _farfield2nearfield_cupy(self, farfield, nearfield_out):
        # FYI: Farfield shape is (N,1)
        N = self.spot_kxy.shape[1]
        N_batch_max = 500

        def expand_kernel(kernel, farfield, out):
            # (H*W, N) x (N,1) = (H*W, 1)   ===reshape===>   (H,W)
            return cp.matmul(kernel, farfield[:, np.newaxis], out=out[:, np.newaxis])

        if self._cupy_kernel is None:
            self.spot_kxy_complex = cp.array(self.spot_kxy, self.dtype_complex)

        if N <= N_batch_max:
            if self._cupy_kernel is None:
                self._cupy_kernel = self._build_kernel_batched(self.spot_kxy_complex, out=self._cupy_kernel)

            expand_kernel(self._cupy_kernel, farfield, out=nearfield_out.ravel())
        else:
            if self._cupy_kernel is None:
                self._cupy_kernel = cp.zeros((np.prod(self.slm_shape), N_batch_max), dtype=self.dtype_complex)

            nearfield_out_temp = cp.zeros(self.slm_shape, dtype=self.dtype_complex)

            batches = 1 + N // N_batch_max

            # print(N)

            for batch in range(batches):
                batch_slice = slice(batch * N_batch_max, np.clip((batch+1) * N_batch_max, 0, N))
                # kernel_slice = slice(0, np.clip(N_batch_max, 0, N - batch * N_batch_max))
                kernel_slice = slice(0, batch_slice.stop - batch_slice.start)
                # print(batch)
                # print(batch_slice)
                # print(kernel_slice)
                # print(self._cupy_kernel.shape)
                self._cupy_kernel[:, kernel_slice] = self._build_kernel_batched(
                    self.spot_kxy_complex[:, batch_slice],
                    out=self._cupy_kernel[:, kernel_slice]
                )
                # print(self._cupy_kernel[:, batch_slice].shape)

                if batch == 0:
                    expand_kernel(self._cupy_kernel[:, kernel_slice], farfield[batch_slice], out=nearfield_out.ravel())
                else:
                    expand_kernel(self._cupy_kernel[:, kernel_slice], farfield[batch_slice], out=nearfield_out_temp.ravel())
                    nearfield_out += nearfield_out_temp

        return nearfield_out

    # Target update.
    def update_target(self, new_target=None, reset_weights=False):
        """
        Change the target to something new. This method handles cleaning and normalization.

        Parameters
        ----------
        new_target : array_like OR None
            A list with ``N`` elements corresponding to the target intensities of each
            of the ``N`` spots.
            If ``None``, sets the target spot amplitudes the contents of :attr:`spot_amp`.
        reset_weights : bool
            Whether to overwrite ``weights`` with ``target``.
        """
        if new_target is None:
            self.target = cp.array(self.spot_amp, dtype=self.dtype, copy=None)
        else:
            new_target = np.squeeze(new_target.ravel())
            if new_target.shape != (len(self),):
                raise ValueError(
                    "Target must be of appropriate shape. "
                    "Initialize a new Hologram if a different shape is desired."
                )

            self.target = cp.array(new_target, dtype=self.dtype, copy=None)
            self.spot_amp = np.array(new_target, dtype=self.dtype, copy=None)

        cp.abs(self.target, out=self.target)
        self.target *= 1 / Hologram._norm(self.target)

        if reset_weights:
            self.reset_weights()

    # Weighting and stats.
    def _update_weights(self):
        """
        Change :attr:`weights` to optimize towards the :attr:`target` using feedback from
        :attr:`amp_ff`, the computed farfield amplitude.
        """
        feedback = self.flags["feedback"]

        if feedback == "computational":
            feedback = self.flags["feedback"] = "computational_spot"

        # Weighting strategy depends on the chosen feedback method.
        if feedback == "computational_spot":
            amp_feedback = self.amp_ff
        elif feedback == "experimental_spot":
            self.measure(basis="ij")

            amp_feedback = np.sqrt(analysis.take(
                np.square(np.array(self.img_ij, copy=None, dtype=self.dtype)),
                self.spot_ij,
                self.spot_integration_width_ij,
                centered=True,
                integrate=True
            ))
        elif feedback == "external_spot":
            amp_feedback = self.external_spot_amp
        else:
            raise ValueError("Feedback '{}' not recognized.".format(feedback))

        # Apply weights.
        self._update_weights_generic(
            self.weights,
            cp.array(amp_feedback, copy=None, dtype=self.dtype),
            self.target,
            nan_checks=True
        )

    def _calculate_stats_computational_spot(self, stats, stat_groups=[]):
        """
        Wrapped by :meth:`SpotHologram.update_stats()`.
        """
        if "computational_spot" in stat_groups:
            stats["computational_spot"] = self._calculate_stats(
                self.amp_ff,
                self.target,
                efficiency_compensation=False,
                raw="raw_stats" in self.flags and self.flags["raw_stats"]
            )

    def _calculate_stats_experimental_spot(self, stats, stat_groups=[]):
        """
        Wrapped by :meth:`SpotHologram.update_stats()`.
        """
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
                xp=np,
                efficiency_compensation=False,
                total=np.sum(pwr_img),
                raw="raw_stats" in self.flags and self.flags["raw_stats"]
            )

        if "external_spot" in stat_groups:
            pwr_feedback = np.square(np.array(self.external_spot_amp, copy=None, dtype=self.dtype))
            stats["external_spot"] = self._calculate_stats(
                np.sqrt(pwr_feedback),
                self.spot_amp,
                xp=np,
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

        self._calculate_stats_computational_spot(stats, stat_groups)
        self._calculate_stats_experimental_spot(stats, stat_groups)

        self._update_stats_dictionary(stats)


class SpotHologram(_AbstractSpotHologram):
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
    """

    def __init__(
        self,
        shape,
        spot_vectors,
        basis="kxy",
        spot_amp=None,
        cameraslm=None,
        null_vectors=None,
        null_radius=None,
        null_region=None,
        null_region_radius_frac=None,
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

            - ``"kxy"`` for centered normalized SLM :math:`k`-space (radians),
            - ``"knm"`` for computational SLM :math:`k`-space (pixels),
            - ``"ij"`` for camera coordinates (pixels).

            Defaults to ``"kxy"``.
        spot_amp : array_like OR None
            The amplitude to target for each spot.
            See :attr:`SpotHologram.spot_amp`.
            If ``None``, all spots are assumed to have the same amplitude.
            Normalization is performed automatically; the user is not required to
            normalize.
        cameraslm : slmsuite.hardware.cameraslms.FourierSLM OR None
            If the ``"ij"`` or ``"kxy"`` bases are chosen, and/or if the user wants to make use of camera
            feedback, a :class:`slmsuite.hardware.cameraslms.FourierSLM` must be provided.
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
        **kwargs
            Passed to :meth:`.FeedbackHologram.__init__()`.
        """
        # Parse vectors.
        vectors = toolbox.format_2vectors(spot_vectors)
        N = vectors.shape[1]

        # Parse spot_amp.
        if spot_amp is not None:
            self.spot_amp = np.ravel(spot_amp)
            if len(self.spot_amp) != N:
                raise ValueError("spot_amp must have the same length as the provided spots.")
        else:
            self.spot_amp = np.full(N, 1.0 / np.sqrt(N))

        # Set the external amp variable to be perfect by default.
        self.external_spot_amp = np.copy(self.spot_amp)

        # Parse null_vectors
        if null_vectors is not None:
            null_vectors = toolbox.format_2vectors(null_vectors)
            if not np.all(np.shape(null_vectors) == np.shape(null_vectors)):
                raise ValueError("null_vectors must have the same length as the provided spots.")
        else:
            self.null_knm = None
            self.null_radius_knm = None
        self.null_region_knm = None

        # Interpret vectors depending upon the basis.
        if basis is None or basis == "knm":  # Computational Fourier space of SLM.
            self.spot_knm = vectors

            if cameraslm is not None:
                self.spot_kxy = toolbox.convert_vector(
                    self.spot_knm,
                    from_units="knm",
                    to_units="kxy",
                    hardware=cameraslm.slm,
                    shape=shape
                )

                if "fourier" in cameraslm.calibrations:
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
        elif basis == "kxy":  # Normalized units.
            assert cameraslm is not None, "We need a cameraslm to interpret kxy."

            self.spot_kxy = vectors

            if hasattr(cameraslm, "calibrations"):
                if "fourier" in cameraslm.calibrations:
                    self.spot_ij = cameraslm.kxyslm_to_ijcam(vectors)
                    # This is okay for non-feedback GS, so we don't error.
            else:
                self.spot_ij = None

            self.spot_knm = toolbox.convert_vector(
                self.spot_kxy,
                from_units="kxy",
                to_units="knm",
                hardware=cameraslm.slm,
                shape=shape
            )
        elif basis == "ij":  # Pixel on the camera.
            assert cameraslm is not None, "We need an cameraslm to interpret ij."
            assert "fourier" in cameraslm.calibrations, (
                "We need an cameraslm with "
                "fourier-calibrated kxyslm_to_ijcam and ijcam_to_kxyslm transforms "
                "to interpret ij."
            )

            self.spot_ij = vectors
            self.spot_kxy = cameraslm.ijcam_to_kxyslm(vectors)
            self.spot_knm = toolbox.convert_vector(
                vectors,
                from_units="ij",
                to_units="knm",
                hardware=cameraslm,
                shape=shape
            )
        else:
            raise Exception("Unrecognized basis for spots '{}'.".format(basis))

        # Handle null conversions in the ij or kxy cases.
        if basis == "ij" or basis == "kxy":
            if null_vectors is not None:
                # Convert the null vectors.
                self.null_knm = toolbox.convert_vector(
                    null_vectors,
                    from_units=basis,
                    to_units="knm",
                    hardware=cameraslm,
                    shape=shape
                )

                # Convert the null radius.
                if null_radius is not None:
                    self.null_radius_knm = toolbox.convert_radius(
                        null_radius,
                        from_units=basis,
                        to_units="knm",
                        hardware=cameraslm,
                        shape=shape
                    )
                else:
                    self.null_radius_knm = None
            else:
                self.null_knm = None
                self.null_radius_knm = None

            self.null_region_knm = null_region

        # Generate point spread functions (psf) for the knm and ij bases
        if cameraslm is not None:
            psf_kxy = np.mean(cameraslm.slm.spot_radius_kxy())
            psf_knm = toolbox.convert_radius(psf_kxy, "kxy", "knm", cameraslm.slm, shape)
            psf_ij = toolbox.convert_radius(psf_kxy, "kxy", "ij", cameraslm, shape)
        else:
            psf_knm = 0
            psf_ij = np.nan

        if np.isnan(psf_knm):
            psf_knm = 0
        if np.isnan(psf_ij):
            psf_ij = 0

        # Use semi-arbitrary values to determine integration widths. The default width is:
        #  - N times the psf,
        #  - but then clipped to be:
        #    + larger than 3 and
        #    + smaller than the minimum inf-norm distance between spots divided by 1.5
        #      (divided by 1 would correspond to the largest non-overlapping integration
        #      regions; 1.5 gives comfortable padding)
        #  - and finally forced to be an odd integer.
        N = 10  # Future: non-arbitrary
        min_psf = 3

        dist_knm = np.max([toolbox.smallest_distance(self.spot_knm) / 1.5, min_psf])
        self.spot_integration_width_knm = np.clip(N * psf_knm, min_psf, dist_knm)
        self.spot_integration_width_knm = int(2 * np.floor(self.spot_integration_width_knm / 2) + 1)

        if self.spot_ij is not None:
            dist_ij = np.max([toolbox.smallest_distance(self.spot_ij) / 1.5, min_psf])
            self.spot_integration_width_ij = np.clip(N * psf_ij, min_psf, dist_ij)
            self.spot_integration_width_ij = int(
                2 * np.floor(self.spot_integration_width_ij / 2) + 1
            )
        else:
            self.spot_integration_width_ij = None

        # Check to make sure spots are within relevant camera and SLM shapes.
        if (
            np.any(self.spot_knm[0] < self.spot_integration_width_knm / 2)
            or np.any(self.spot_knm[1] < self.spot_integration_width_knm / 2)
            or np.any(self.spot_knm[0] >= shape[1] - self.spot_integration_width_knm / 2)
            or np.any(self.spot_knm[1] >= shape[0] - self.spot_integration_width_knm / 2)
        ):
            raise ValueError(
                "Spots outside SLM computational space bounds!\nSpots:\n{}\nBounds: {}".format(
                    self.spot_knm, shape
                )
            )

        if self.spot_ij is not None:
            cam_shape = cameraslm.cam.shape

            if (
                np.any(self.spot_ij[0] < self.spot_integration_width_ij / 2)
                or np.any(self.spot_ij[1] < self.spot_integration_width_ij / 2)
                or np.any(self.spot_ij[0] >= cam_shape[1] - self.spot_integration_width_ij / 2)
                or np.any(self.spot_ij[1] >= cam_shape[0] - self.spot_integration_width_ij / 2)
            ):
                raise ValueError(
                    "Spots outside camera bounds!\nSpots:\n{}\nBounds: {}".format(
                        self.spot_ij, cam_shape
                    )
                )

        # Decide the null_radius (if necessary)
        if self.null_knm is not None:
            if self.null_radius_knm is None:
                all_spots = np.hstack((self.null_knm, self.spot_knm))
                self.null_radius_knm = toolbox.smallest_distance(all_spots) / 4

            self.null_radius_knm = int(np.ceil(self.null_radius_knm))

        # Initialize target/etc. Note that this passes through FeedbackHologram.
        super().__init__(shape, target_ij=None, cameraslm=cameraslm, **kwargs)

        # Parse null_region after __init__
        if basis == "ij" and null_region is not None:
            # Transformation order of zero to prevent nan-blurring in MRAF cases.
            self.null_region_knm = (
                self.ijcam_to_knmslm(null_region, out=self.null_region_knm, order=0) != 0
            )

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
            mask = cp.square(xg) + cp.square(yg) > null_region_radius_frac**2
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

    # Target update.
    @staticmethod
    def make_rectangular_array(
        shape,
        array_shape,
        array_pitch,
        array_center=None,
        basis="knm",
        orientation_check=False,
        **kwargs,
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
                assert "fourier" in cameraslm.calibrations, (
                    "We need an cameraslm with "
                    "fourier-calibrated kxyslm_to_ijcam and ijcam_to_kxyslm transforms "
                    "to interpret ij."
                )

                array_center = toolbox.convert_vector(
                    (0, 0),
                    from_units="kxy",
                    to_units="ij",
                    hardware=cameraslm
                )

        # Make the grid edges.
        x_edge = np.arange(array_shape[0]) - (array_shape[0] - 1) / 2.0
        x_edge = x_edge * array_pitch[0] + array_center[0]
        y_edge = np.arange(array_shape[1]) - (array_shape[1] - 1) / 2.0
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

    def _update_target_spots(self, reset_weights=False):
        """
        Wrapped by :meth:`SpotHologram.update_target()`.
        """
        # Round the spot points to the nearest integer coordinates in knm space.
        self.spot_knm_rounded = np.rint(self.spot_knm).astype(int)

        # Convert these to the other coordinate systems if possible.
        if self.cameraslm is not None:
            self.spot_kxy_rounded = toolbox.convert_vector(
                self.spot_knm_rounded,
                from_units="knm",
                to_units="kxy",
                hardware=self.cameraslm.slm,
                shape=self.shape,
            )

            if "fourier" in self.cameraslm.calibrations:
                self.spot_ij_rounded = self.cameraslm.kxyslm_to_ijcam(self.spot_kxy_rounded)
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
                w = int(2 * self.null_radius_knm + 1)

                for ii in range(all_spots.shape[1]):
                    toolbox.imprint(
                        self.target,
                        (np.rint(all_spots[0, ii]), w, np.rint(all_spots[1, ii]), w),
                        0,
                        centered=True,
                        circular=True,
                    )

        # Set all the target pixels to the appropriate amplitude.
        self.target[self.spot_knm_rounded[1, :], self.spot_knm_rounded[0, :]] = self.spot_amp

        self.target /= Hologram._norm(self.target)

        if reset_weights:
            self.reset_weights()

    def update_target(self, reset_weights=False, plot=False):
        """
        From the spot locations stored in :attr:`spot_knm`, update the target pattern.

        Note
        ~~~~
        If there's a ``cameraslm``, updates the :attr:`spot_ij_rounded` attribute
        corresponding to where pixels in the :math:`k`-space where actually placed (due to rounding
        to integers, stored in :attr:`spot_knm_rounded`), rather the
        idealized floats :attr:`spot_knm`.

        Note
        ~~~~
        The :attr:`target` and :attr:`weights` matrices are modified in-place for speed,
        unlike :class:`.Hologram` or :class:`.FeedbackHologram` which make new matrices.
        This is because spot positions are expected to be corrected using :meth:`refine_offsets()`.

        Parameters
        ----------
        reset_weights : bool
            Whether to reset the :attr:`weights` to this new :attr:`target`.
        """
        self._update_target_spots(reset_weights=reset_weights)

    # Weighting and stats.
    def _update_weights(self):
        """
        Change :attr:`weights` to optimize towards the :attr:`target` using feedback from
        :attr:`amp_ff`, the computed farfield amplitude.
        """
        feedback = self.flags["feedback"]

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
                    xp=cp
                ))
            elif feedback == "experimental_spot":
                self.measure(basis="ij")

                amp_feedback = np.sqrt(
                    analysis.take(
                        np.square(np.array(self.img_ij, copy=None, dtype=self.dtype)),
                        self.spot_ij,
                        self.spot_integration_width_ij,
                        centered=True,
                        integrate=True,
                    )
                )
            elif feedback == "external_spot":
                amp_feedback = self.external_spot_amp
            else:
                raise ValueError("Feedback '{}' not recognized.".format(feedback))

            # Update the weights of single pixels.
            self.weights[self.spot_knm_rounded[1, :], self.spot_knm_rounded[0, :]] = (
                self._update_weights_generic(
                    self.weights[self.spot_knm_rounded[1, :], self.spot_knm_rounded[0, :]],
                    cp.array(amp_feedback, copy=None, dtype=self.dtype),
                    self.spot_amp,
                    nan_checks=True
                )
            )

    def _calculate_stats_computational_spot(self, stats, stat_groups=[]):
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
                    raw="raw_stats" in self.flags and self.flags["raw_stats"],
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
                        xp=cp,
                    )

                    stats["computational_spot"] = self._calculate_stats(
                        cp.sqrt(pwr_feedback),
                        self.spot_amp,
                        xp=cp,
                        efficiency_compensation=False,
                        total=cp.sum(pwr_ff),
                        raw="raw_stats" in self.flags and self.flags["raw_stats"],
                    )
                else:
                    pwr_ff = np.square(self.amp_ff)
                    pwr_feedback = analysis.take(
                        pwr_ff,
                        self.spot_knm,
                        self.spot_integration_width_knm,
                        centered=True,
                        integrate=True,
                    )

                    stats["computational_spot"] = self._calculate_stats(
                        np.sqrt(pwr_feedback),
                        self.spot_amp,
                        xp=np,
                        efficiency_compensation=False,
                        total=np.sum(pwr_ff),
                        raw="raw_stats" in self.flags and self.flags["raw_stats"],
                    )

    def _calculate_stats_experimental_spot(self, stats, stat_groups=[]):
        """
        Wrapped by :meth:`SpotHologram.update_stats()`.
        """

        if "experimental_spot" in stat_groups:
            self.measure(basis="ij")

            pwr_img = np.square(self.img_ij)

            pwr_feedback = analysis.take(
                pwr_img, self.spot_ij, self.spot_integration_width_ij, centered=True, integrate=True
            )

            stats["experimental_spot"] = self._calculate_stats(
                np.sqrt(pwr_feedback),
                self.spot_amp,
                xp=np,
                efficiency_compensation=False,
                total=np.sum(pwr_img),
                raw="raw_stats" in self.flags and self.flags["raw_stats"],
            )

        if "external_spot" in stat_groups:
            pwr_feedback = np.square(np.array(self.external_spot_amp, copy=None, dtype=self.dtype))
            stats["external_spot"] = self._calculate_stats(
                np.sqrt(pwr_feedback),
                self.spot_amp,
                xp=np,
                efficiency_compensation=False,
                total=np.sum(pwr_feedback),
                raw="raw_stats" in self.flags and self.flags["raw_stats"],
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
        self._calculate_stats_computational_spot(stats, stat_groups)
        self._calculate_stats_experimental_spot(stats, stat_groups)

        self._update_stats_dictionary(stats)

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
            self.measure(basis="ij")
            img = self.img_ij

        # Take regions around each point from the given image.
        regions = analysis.take(
            img, self.spot_ij, self.spot_integration_width_ij, centered=True, integrate=False
        )

        # Fast version; have to iterate for accuracy.
        shift_vectors = analysis.image_positions(regions)
        shift_vectors = np.clip(
            shift_vectors, -self.spot_integration_width_ij / 4, self.spot_integration_width_ij / 4
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
                img,
                self.spot_ij,
                self.spot_integration_width_ij,
                centered=True,
                integrate=False,
                return_mask=True,
            )

            plt.figure(figsize=(12, 12))
            plt.imshow(img * mask)
            plt.scatter(sv1[0, :], sv1[1, :], s=200, fc="none", ec="r")
            plt.scatter(sv2[0, :], sv2[1, :], s=300, fc="none", ec="b")
            plt.show()

        # Handle the feedback applied from this refinement.
        if basis is not None:
            if basis == "kxy" or basis == "knm":
                # Modify k-space targets. Don't modify any camera spots.
                self.spot_kxy = self.spot_kxy - (
                    self.cameraslm.ijcam_to_kxyslm(shift_vectors)
                    - self.cameraslm.ijcam_to_kxyslm((0, 0))
                )
                self.spot_knm = toolbox.convert_vector(
                    self.spot_kxy,
                    to_units="kxy",
                    from_units="knm",
                    hardware=self.cameraslm.slm,
                    shape=self.shape
                )
                self.update_target(reset_weights=True)
            elif basis == "ij":
                # Modify camera targets. Don't modify any k-vectors.
                self.spot_ij = self.spot_ij + shift_vectors
            else:
                raise Exception("Unrecognized basis '{}'.".format(basis))

        return shift_vectors