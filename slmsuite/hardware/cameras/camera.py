"""
Abstract camera functionality.
"""

import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

from slmsuite.holography import analysis
from slmsuite.misc import lorentzian_fitfun, lorentzian_jacobian


class Camera:
    """
    Abstract class for cameras. Comes with transformations and helper functions like autoexpose.

    Attributes
    ----------
    name : str
        Camera identifier.
    shape : (int, int)
        Stores (`height`, `width`) of the camera in pixels, same form as :meth:`numpy.shape`.
    bitdepth : int
        Depth of a camera pixel well in bits.
    bitresolution : int
        Stores ``2**bitdepth``.
    dx_um : float or None
        x pixel pitch in um. Defaults to ``None``. Potential future features will use this.
    dy_um : float or None
        See :attr:`dx_um`.
    window : array_like
        Window information in ``(x, width, y, height)`` form.
    default_shape : tuple
        Default ``shape`` of the camera before any WOI changes are made.
    transform : lambda
        Flip and/or rotation operator specified by the user in :meth:`__init__`.
        The user is expected to apply this transform to the matrix returned in
        :meth:`get_image()`. Note that ROI is applied on the camera hardware
        before this transformation.
    """

    def __init__(
        self,
        width,
        height,
        bitdepth,
        dx_um=None,
        dy_um=None,
        rot="0",
        fliplr=False,
        flipud=False,
    ):
        """
        Initializes a camera.

        In addition to the other class attributes, accepts the following parameters
        to set :attr:`transform`. See :meth:`~slmsuite.holograpy.analysis.get_transform()`.

        Parameters
        ----------
        width
            See :attr:`shape`.
        height
            See :attr:`shape`.
        bitdepth
            See :attr:`bitdepth`.
        dx_um
            See :attr:`dx_um`.
        dy_um
            See :attr:`dy_um`.
        rot : str or int
            Rotates returned image by the corresponding degrees in ``["90", "180", "270"]``
            or :meth:`numpy.rot90` code in ``[1, 2, 3]``. Defaults to no rotation.
            Used to determine :attr:`shape` and :attr:`transform`.
        fliplr : bool
            Flips returned image left right.
            Used to determine :attr:`transform`.
        flipud : bool
            Flips returned image up down.
            Used to determine :attr:`transform`.
        """
        if rot in ("90", 1, "270", 3):
            self.shape = (width, height)
            self.default_shape = (width, height)
        else:
            self.shape = (height, width)
            self.default_shape = (height, width)

        self.window = (0, width, 0, height)

        self.bitdepth = bitdepth
        self.bitresolution = 2 ** bitdepth

        self.dx_um = dx_um
        self.dy_um = dy_um

        # Create image transformation.
        self.transform = analysis.get_transform(rot, fliplr, flipud)

    def close(self):
        """
        Close the camera and delete related objects.
        """
        raise NotImplementedError()

    def reset(self):
        """
        Reset the camera to a default state.
        """
        raise NotImplementedError()

    def get_exposure(self):
        """
        Get integration time in seconds. Used in :meth:`.autoexposure()`.

        Returns
        -------
        float
            Integration time in seconds.
        """
        raise NotImplementedError()

    def set_exposure(self, exposure_s):
        """
        Set integration time in seconds. Used in :meth:`.autoexposure()`.

        Parameters
        ----------
        exposure_s : float
            The integration time in seconds.
        """
        raise NotImplementedError()

    def set_woi(self, window=None):
        """
        Narrows imaging region to window of interest for faster framerates.

        Parameters
        ----------
        window : list, None
            See :attr:`~slmsuite.hardware.cameras.camera.Camera.window`.
            If `None`, defaults to largest possible.

        Returns
        ----------
        window : list
            :attr:`~slmsuite.hardware.cameras.camera.Camera.window`.
        """
        raise NotImplementedError()

    def flush(self, timeout_s=1):
        """
        Cycle the image buffer (if any) such that all new :meth:`.get_image()`
        calls yield new frames.

        Parameters
        ----------
        timeout_s : float
            The time in seconds to wait for frames to catch up with triggers.
        """
        raise NotImplementedError()

    def get_image(self, timeout_s=1):
        """
        Pull an image from the camera and return.

        Parameters
        ----------
        timeout_s : float
            The time in seconds to wait for the frame to be fetched.

        Returns
        -------
        numpy.ndarray
            Array of shape :attr:`~slmsuite.hardware.cameras.camera.Camera.shape`.
        """
        raise NotImplementedError()

    def get_images(self, image_count, flush=False):
        """
        Grab `image_count` images in succession. Overwrite this
        impelementation if a camera supports faster batch aquisition.

        Parameters
        ----------
        image_count : int
            Number of images to grab.
        flush : bool
            Whether to flush before grabbing.

        Returns
        -------
        numpy.ndarray
            Array of shape `(image_count, height, width)`.
        """
        # Preallocate memory.
        imlist = np.empty((int(image_count), self.shape[0], self.shape[1]))

        # Flush if desired.
        if flush:
            self.flush()

        # Grab images.
        for i in range(image_count):
            imlist[i] = self.get_image()

        return imlist

    def autoexposure(
        self,
        set_fraction=0.5,
        tol=0.05,
        bounds_s=(1e-6, 0.1),
        window=None,
        averages=5,
        timeout_s=5,
    ):
        """
        Sets the exposure of the camera such that the maximum value is at `set_fraction`
        of the dynamic range. Useful for mitigating deleterious over- or under- exposure.

        Parameters
        --------
        set_fraction : float
            Fraction of camera dynamic range to target image maximum.
        tol : float
            Fractional tolerance for exposure adjustment.
        bounds_s : list of float
            Shortest and longest allowable integration in seconds.
        window : array_like or None
            See :attr:`~slmsuite.hardware.cameras.camera.Camera.window`.
            If `None`, the full camera frame will be used.
        averages : int
            Number of frames to average intensity over for noise reduction.
        timeout_s : float
            Stop attempting to autoexpose after timeout_s seconds.

        Returns
        --------
        float
            Resulting exposure in seconds.
        """
        # TODO: pull `bounds_s` values from the camera.

        if window is None:
            wxi = 0
            wxf = self.shape[1]
            wyi = 0
            wyf = self.shape[0]
        else:
            wxi = int(window[0] - window[1] / 2)
            wxf = int(window[0] + window[1] / 2)
            wyi = int(window[2] - window[3] / 2)
            wyf = int(window[2] + window[3] / 2)

        set_val = 0.5 * self.bitresolution
        exp = self.get_exposure()
        im_mean = np.mean(self.get_images(averages, flush=True), 0)[
            wyi:wyf, wxi:wxf
        ].ravel()
        im_max = np.mean(im_mean[im_mean.argsort()[-averages:]])
        # Calculate the error as a percent of the camera's bitresolution
        err = np.abs(im_max - set_val) / self.bitresolution
        t = time.perf_counter()

        while err > tol and time.perf_counter() - t < timeout_s:
            # Clip exposure steps to 0.5x -> 2x
            exp = exp / np.amax([0.5, np.amin([(im_max / set_val), 2])])
            exp = np.amax([bounds_s[0], np.amin([exp, bounds_s[1]])])
            self.set_exposure(exp)
            im_mean = np.mean(self.get_images(averages, flush=True), 0)[
                wyi:wyf, wxi:wxf
            ].ravel()
            im_max = np.mean(im_mean[im_mean.argsort()[-averages:]])
            err = np.abs(im_max - set_val) / self.bitresolution

            print(exp, im_max)

        exp_fin = exp * 2 * set_fraction

        if set_fraction != 0.5:  # Sets for full dynamic range
            self.set_exposure(exp_fin)

        return exp_fin

    def autofocus(self, z_get, z_set, z_list=None, plot=False):
        """
        Uses an FFT contrast metric to find optimal focus when scanning over some variable
        `z`. This `z` often takes the form of a vertical stage to position a sample precisely
        at the plane of imaging of a lens or objective. The contrast metric works particularly
        well when combined with a projected spot array hologram.

        Parameters
        ----------
        z_get : lambda
            Gets the current position of the focusing stage. Should return a `float`.
        z_set : lambda
            Sets the position of the focusing stage to a given `float`.
        z_list : array_like or None
            `z` values to sweep over during search.
            Defaults (when `None`) to ``numpy.linspace(-4,4,16)``.
        plot : bool
            Whether to provide illustrative plots.
        """
        if z_list is None:
            z_list = np.linspace(-4, 4, 16)

        self.flush()

        z_base = z_get()
        imlist = []
        z_list = z_list + z_base
        counts = np.zeros_like(z_list)

        z_set(z_list[0])

        for i, z in enumerate(z_list):
            print("Moving to " + str(z))
            z_set(z)

            # Take image.
            img = self.get_image()
            imlist.append(np.copy(img))

            # Evaluate metric.
            dft = np.fft.fftshift(np.fft.fft2(imlist[-1].astype(float)))
            dft_amp = np.abs(dft)
            dft_norm = dft_amp / np.amax(dft_amp)
            fom_ = np.sum(dft_norm)
            counts[i] = fom_
            if plot:
                _, axs = plt.subplots(1, 2)
                axs[0].imshow(imlist[-1])
                axs[0].set_title("Image")
                axs[0].set_xticks([])
                axs[0].set_yticks([])
                axs[1].imshow(dft_norm)
                axs[1].set_title(
                    "FFT\nFoM$ = \\int\\int $|FFT|$ / $max|FFT|$ = {}$".format(fom_)
                )
                axs[1].set_xticks([])
                axs[1].set_yticks([])
                plt.show()

        counts[0] = counts[1]

        popt0 = np.array(
            [z_list[np.argmax(counts)], np.max(counts), 100, np.min(counts)]
        )

        try:
            popt, _ = curve_fit(
                lorentzian_fitfun,
                z_list,
                counts,
                jac=lorentzian_jacobian,
                ftol=1e-5,
                p0=popt0,
            )
            z_opt = popt[0]
            c_opt = popt[1]
        except BaseException:
            print("Autofocus fit failed, using maximum fom as optimum...")
            z_opt = z_list[np.argmax(counts)]
            c_opt = counts[np.argmax(counts)]

        # Return to original state except focus z
        print("Moving to optimized value " + str(z_opt))
        z_set(z_opt)

        # Show result if desired
        if plot:
            plt.plot(z_list, counts)
            plt.xlabel(r"$z$ $\mu$m")
            plt.ylabel("fom: Data, Guess, & Fit")
            plt.title("Focus Sweep")
            plt.scatter(z_opt, c_opt)
            plt.plot(z_list, lorentzian_fitfun(z_list, *popt0))
            lfit = None
            try:
                lfit = lorentzian_fitfun(z_list, *popt)
            except BaseException:
                lfit = None
            if lfit is not None:
                plt.plot(z_list, lfit)
            plt.legend(["Data", "Guess", "Result"])
            plt.show()

            plt.imshow(self.get_image())
            plt.title("Focused Image")
            plt.show()

        return z_opt, imlist
