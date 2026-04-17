import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
from tqdm.auto import tqdm
import warnings

from slmsuite.holography import analysis
from slmsuite.holography import toolbox
from slmsuite.holography.toolbox.phase import binary

class _PixelCalibration(object):
    """
    Hidden superclass with pixel calibration methods
    (gamma and crosstalk correction).
    """
    ### Pixel Crosstalk and Gamma Calibration ###

    def pixel_calibrate(
        self,
        levels=2,
        periods=2,
        orders=3,
        window=None,
        field_period=10,
    ):
        r"""
        Measure the pixel crosstalk and phase response of the SLM.

        **(This feature is experimental.)**

        Physical SLMs do not produce perfectly sharp and discrete blocks of a desired
        phase at each pixel. Rather, the realized phase might deviate from the desired
        phase (error) and be blurred between pixels (crosstalk).

        We adopt a literature approach to calibrating both phenomena by `measuring the
        system response of binary gratings <https://doi.org/10.1364/OE.20.022334>`_.
        In the future, we intend to fit the measured data to `an upgraded asymmetric
        model of phase crosstalk <https://doi.org/10.1364/OE.27.025046>`_, and then
        apply the model to beam propagation during holographic optimization. A better
        understanding of the system error can lead to holograms that take this error
        into account.

        Note that this algorithm does not operate at the level of individual pixels, but
        rather on aggregate statistics over a region of pixels.
        Right now, this calibration is done for one region (which defaults to the full
        SLM). In the future, we might want to calibrate many regions across the SLM to
        measure `spatially varying phase response <https://doi.org/10.1364/OE.21.016086>`_

        Note
        ~~~~
        A Fourier calibration must be loaded.

        Caution
        ~~~~~~~
        Data must be acquired without wavefront calibration applied.
        If the uncalibrated SLM produces too defocussed of a spot,
        then this measurement may not be ideal. On the flip side, a
        too-focussed spot might increase error by integrating over fewer camera pixels.

        Parameters
        ----------
        levels : int OR array_like of int
            Which bitlevels to test, out of the :math:`2^B` levels available for a
            :math:`B`-bit SLM. Note that runtime scales with :math:`\mathcal{O}(B^2)`.
            If an integer is passed, the integer is rounded up to the next largest power of
            two, and this number of bitlevels are sampled.
        periods : int OR array_like of int
            List of periods (in pixels) of the binary gratings that we will apply.
            Must be even integers.
            If a single ``int`` is provided, then a list containing the given number of
            periods is chosen, based upon the field of view of the camera.
        orders : int OR array_like of int
            Orders (..., -1st, 0th, 1st, ...) of the binary gratings to measure data at.
            If scalar is provided, measures orders between -nth and nth order, inclusive.
        window
            If not ``None``, the pixel calibration is only done over the region of the SLM
            defined by ``window``.
            Passed to :meth:`~slmsuite.holography.toolbox.window_slice()`.
            See :meth:`~slmsuite.holography.toolbox.window_slice()` for various options.
        field_period : int
            If ``window`` is not ``None``, then the field is deflected away in an
            orthogonal direction with a grating of the given period.
        """
        # Parse levels by forcing range and datatype.
        if np.isscalar(levels):
            if levels < 1:
                levels = 1
            levels = int(2 ** (np.ceil(np.log2(levels))))

            if levels > self.slm.bitresolution:
                warnings.warn(
                    f"Requested {levels} levels are more than the "
                    f"bitresolution. Truncating to {self.slm.bitresolution}."
                )
                levels = self.slm.bitresolution

            levels = np.arange(levels) * (self.slm.bitresolution / levels)
        levels = np.mod(levels, self.slm.bitresolution).astype(self.slm.display.dtype)
        N = len(levels)

        # Parse periods by forcing integer.
        if np.isscalar(periods):
            raise NotImplementedError("TODO")

        periods = np.rint(periods).astype(int)
        periods = 2 * (periods // 2)
        P = len(periods)

        if len(np.unique(periods)) != len(periods):
            raise RuntimeError(f"Repeated periods in {periods}")

        if np.any(periods <= 0):
            raise ValueError("period should not be negative.")

        # Parse orders by forcing integer.
        if np.isscalar(orders):
            orders = int(orders)
            orders = np.arange(-orders, orders+1)
        orders = orders.astype(int)
        M = len(orders)

        if not 1 in orders:
            raise ValueError("1st order must be included.")

        # Parse window.
        if window is not None:
            (_, w, _, h) = toolbox.window_extent(window)
            if np.any(periods > w // 2) or np.any(periods > h // 2):
                raise ValueError(f"Periods {periods} must be at least half of the window size ({w}, {h}).")

        # Figure out our shape.
        shape = (2, P, N, N, M)
        data = np.zeros(shape)

        # Make all of the x-pointing vectors, then all of the y-pointing vectors.
        vectors_freq = np.zeros((2, 2*P))
        vectors_freq[0, :P] = vectors_freq[1, P:] = np.reciprocal(periods.astype(float))
        vectors_kxy = toolbox.convert_vector(
            vectors_freq,
            from_units="freq",
            to_units="norm",
            hardware=self
        )

        # Make the y-pointing field vector, then the x-pointing field vector.
        field_freq = np.zeros((2, 2))
        field_freq[0, 0] = field_freq[1, 1] = 1 / float(field_period)
        field_kxy = toolbox.convert_vector(
            field_freq,
            from_units="freq",
            to_units="norm",
            hardware=self
        )
        field_values = np.array([self.slm.bitresolution / 2, 0]).astype(self.slm.display.dtype)
        field_hi, field_lo = field_values

        field_ij = toolbox.convert_vector(
            field_freq,
            from_units="freq",
            to_units="ij",
            hardware=self
        )

        # Figure out where the orders will appear on the camera.
        vectors_ij = self.kxyslm_to_ijcam(vectors_kxy)
        center = self.kxyslm_to_ijcam((0,0))

        dorder = vectors_ij - center
        dfield = field_ij - center
        order_ij = []

        for i in range(2*P):
            order_ij.append(center + orders * dorder[:, [i]])

        integration_size = int(np.ceil(np.min([
            np.min(np.max(dorder, axis=1)),
            np.min(np.max(dfield, axis=1))
        ])))

        # FUTURE: Warn the user if any order is outside the field of view.
        if False:
            warnings.warn("FUTURE")

        # if True: iterations = tqdm(range(P*(N-1)*N))
        if True: iterations = tqdm(range(2*P*N*N))

        # Big sweep.
        for i in [0,1]:                                         # Direction (x,y)
            prange = np.arange(P) + i*P
            for j in range(P):                                  # Period
                for k in range(N):                              # Upper triangular gray level selection.
                    for l in range(N):                          # Periodic normalization when equal.
                    # for l in range(k, N):                       # Periodic normalization when equal.
                        if window is None:
                            phase = binary(
                                self.slm,
                                vector=vectors_kxy[:, prange[j]],
                                a=levels[k],
                                b=levels[l]
                            )
                        else:
                            # In windowed mode, blaze the field away from the 0th order,
                            # in the direction perpendicular to the target.
                            phase = binary(
                                grid=self.slm,
                                vector=field_kxy[:, i],
                                a=field_hi,
                                b=field_lo
                            )
                            toolbox.imprint(
                                phase,
                                window=window,
                                function=binary,
                                grid=self.slm,
                                vector=vectors_kxy[:, prange[j]],
                                a=levels[k],
                                b=levels[l]
                            )

                        # We're writing integers, so this goes directly to the SLM,
                        # bypassing phase2gray.
                        self.slm.set_phase(phase, phase_correct=False, settle=True)

                        data[i,j,k,l,:] = analysis.take(    # = data[i,j,l,k,:]
                            images=self.cam.get_image(),
                            vectors=order_ij[prange[j]],
                            size=integration_size,
                            integrate=True,
                        ).astype(float)

                        if True: iterations.update()

        if True: iterations.close()

        # Assemble the return dictionary.
        self.calibrations["pixel"] = {
            "levels" : levels,
            "periods" : periods,
            "orders" : orders,
            "data": data
        }
        self.calibrations["pixel"].update(self._get_calibration_metadata())

        # Process by default because we currently don't have any arguments.
        # self.pixel_calibration_process()

        return self.calibrations["pixel"]

    def pixel_calibration_process(self):
        """
        Currently, this method only displays debug plots of the measurements.
        In the future, the measurements will be fit in a way that can be applied to
        propagation.
        """
        cal = self.calibrations["pixel"]
        periods = cal["periods"]
        orders = cal["orders"]
        levels = cal["levels"]
        data = cal["data"]

        first_order = np.arange(len(orders))[orders == 1][0]

        rolled = data.copy()

        # rolled /= rolled[:,:,:,:,[first_order]]

        # for i in range(1, len(levels)):
        #     rolled[:,:,[i],:,:] = np.roll(rolled[:,:,[i],:,:], -i, axis=3)

        for i, direction in enumerate(["x"]): #, "y"]):
            for j, period in enumerate(periods[[0]]):
                for o, order in enumerate(orders):
                    plt.imshow(rolled[i,j,:,:,o], vmin=0)
                    plt.title(f"{period}-pixel, ${direction}$ grating; measuring order {order}")
                    # plt.clim(0,1)
                    plt.show()

    @staticmethod
    def pixel_kernel(x, a_pix=.1, n=1, a_minus_pix=None, n_minus=None):
        r"""
        Blurring kernel

        .. math:: K(x) =    \left\{
                                \begin{array}{ll}
                                    \exp\left(-\left|\frac{x}{\alpha_+}\right|^{n_+}\right), & x \ge 0, \\
                                    \exp\left(-\left|\frac{x}{\alpha_-}\right|^{n_-}\right), & x < 0.
                                \end{array}
                            \right.
        """
        # Parse minus parameters by defaulting to plus parameters.
        if a_minus_pix is None:
            a_minus_pix = a_pix
        if n_minus is None:
            n_minus = n

        # Create and normalize the kernel.
        kernel = np.where(
            x >= 0,
            np.exp(-np.power(np.abs(x / a_pix), n)),
            np.exp(-np.power(np.abs(x / a_minus_pix), n_minus)),
        )
        kernel /= np.sum(kernel)

        return kernel

    def _pixel_calibrate_simulate(self, period=16, supersample=16, **kwargs):
        N = int(period * supersample)

        x = np.linspace(-period, period, N)
        x -= np.mean(x)

        y = np.zeros_like(x)
        y[x < 0] = 1

        blaze = np.linspace(0, 2, N)
        # blaze -= np.mean(blaze)

        plt.plot(x, y)
        plt.plot(x, blaze)

        x2 = np.linspace(-2, 2, 4*supersample)
        x2 -= np.mean(x2)
        K = self.pixel_kernel(x2, **kwargs)

        y = ndimage.convolve1d(y, K, mode="wrap")

        plt.plot(x, y)
        plt.show()

        kx = np.arange(float(N)) #/ supersample
        kx -= np.mean(kx)

        Y = np.fft.fftshift(np.fft.fft(np.exp(1j * np.pi * y)))

        Y2 = np.fft.fftshift(np.fft.fft(np.exp(1j * np.pi * blaze)))

        plt.hlines(0, np.min(kx), np.max(kx))
        plt.scatter(kx, np.square(np.abs(Y)))
        plt.scatter(kx, np.square(np.abs(Y2)))
        # plt.xlim(-10, 10)
        plt.show()
