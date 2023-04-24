"""
Abstract camera functionality.
"""

import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

from slmsuite.holography import analysis
from slmsuite.misc.fitfunctions import lorentzian, lorentzian_jacobian

class Camera:
    """
    Abstract class for cameras. Comes with transformations and helper functions like autoexpose.

    Attributes
    ----------
    name : str
        Camera identifier.
    shape : (int, int)
        Stores ``(height, width)`` of the camera in pixels, the same convention as :meth:`numpy.shape`.
    bitdepth : int
        Depth of a camera pixel well in bits.
    bitresolution : int
        Stores ``2**bitdepth``.
    dx_um : float or None
        :math:`x` pixel pitch in microns. Defaults to ``None``.
        Potential future features will use this.
    dy_um : float or None
        :math:`y` pixel pitch in microns. See :attr:`dx_um`.
    exposure_bounds_s : (float, float) OR None
        Shortest and longest allowable integration in seconds.
    woi : tuple
        WOI (window of interest) in ``(x, width, y, height)`` form.
    default_shape : tuple
        Default ``shape`` of the camera before any WOI changes are made.
    transform : lambda
        Flip and/or rotation operator specified by the user in :meth:`__init__`.
        The user is expected to apply this transform to the matrix returned in
        :meth:`get_image()`. Note that WOI changes are applied on the camera hardware
        **before** this transformation.
    """

    def __init__(
        self,
        width,
        height,
        bitdepth=8,
        dx_um=None,
        dy_um=None,
        rot="0",
        fliplr=False,
        flipud=False,
        name="camera"
    ):
        """
        Initializes a camera.

        In addition to the other class attributes, accepts the following parameters
        to set :attr:`transform`. See :meth:`~slmsuite.holography.analysis.get_orientation_transformation()`.

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
        name : str
            Defaults to ``"camera"``.
        """
        # Set shape, depending upon transform.
        if rot in ("90", 1, "270", 3):
            self.shape = (width, height)
            self.default_shape = (width, height)
        else:
            self.shape = (height, width)
            self.default_shape = (height, width)

        # Create image transformation.
        self.transform = analysis.get_orientation_transformation(rot, fliplr, flipud)

        # Update WOI information.
        self.woi = (0, width, 0, height)
        self.set_woi()

        # Set other useful parameters
        self.bitdepth = bitdepth
        self.bitresolution = 2 ** bitdepth

        # Spatial dimensions
        self.dx_um = dx_um
        self.dy_um = dy_um

        self.name = name

        # Default to None, allow subclass constructors to fill.
        self.exposure_bounds_s = None

    def close(self):
        """
        Abstract method to close the camera and delete related objects.
        """
        raise NotImplementedError()

    @staticmethod
    def info(verbose=True):
        """
        Abstract method to load display information.

        Parameters
        ----------
        verbose : bool
            Whether or not to print display information.

        Returns
        -------
        list
            An empty list.
        """
        if verbose: print(".info() NotImplemented.")
        return []

    def reset(self):
        """
        Abstract method to reset the camera to a default state.
        """
        raise NotImplementedError()

    def get_exposure(self):
        """
        Abstract method to get the integration time in seconds.
        Used in :meth:`.autoexposure()`.

        Returns
        -------
        float
            Integration time in seconds.
        """
        raise NotImplementedError()

    def set_exposure(self, exposure_s):
        """
        Abstract method to set the integration time in seconds.
        Used in :meth:`.autoexposure()`.

        Parameters
        ----------
        exposure_s : float
            The integration time in seconds.
        """
        raise NotImplementedError()

    def set_woi(self, woi=None):
        """
        Abstract method to narrow the imaging region to a 'window of interest'
        for faster framerates.

        Parameters
        ----------
        woi : list, None
            See :attr:`~slmsuite.hardware.cameras.camera.Camera.woi`.
            If ``None``, defaults to largest possible.

        Returns
        ----------
        woi : list
            :attr:`~slmsuite.hardware.cameras.camera.Camera.woi`.
        """
        return
        # raise NotImplementedError()

    def flush(self, timeout_s=1):
        """
        Abstract method to cycle the image buffer (if any)
        such that all new :meth:`.get_image()`
        calls yield fresh frames.

        Parameters
        ----------
        timeout_s : float
            The time in seconds to wait for frames to catch up with triggers.
        """
        raise NotImplementedError()

    def get_image(self, timeout_s=1):
        """
        Abstract method to pull an image from the camera and return.

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
        Grab ``image_count`` images in succession. Overwrite this
        implementation if a camera supports faster batch acquisition.

        Parameters
        ----------
        image_count : int
            Number of images to grab.
        flush : bool
            Whether to flush before grabbing.

        Returns
        -------
        numpy.ndarray
            Array of shape ``(image_count, height, width)``.
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
        exposure_bounds_s=None,
        window=None,
        average_count=5,
        timeout_s=5,
        verbose=True
    ):
        """
        Sets the exposure of the camera such that the maximum value is at ``set_fraction``
        of the dynamic range. Useful for mitigating deleterious over- or under- exposure.

        Parameters
        --------
        set_fraction : float
            Fraction of camera dynamic range to target image maximum.
        tol : float
            Fractional tolerance for exposure adjustment.
        exposure_bounds_s : (float, float) OR None
            Shortest and longest allowable integration in seconds. If ``None``, defaults to
            :attr:`exposure_bounds_s`. If this attribute was not set (or not availible on
            a particular camera), then ``None`` instead defaults to unbounded.
        window : array_like or None
            See :attr:`~slmsuite.hardware.cameras.camera.Camera.window`.
            If ``None``, the full camera frame will be used.
        average_count : int
            Number of frames to average intensity over for noise reduction.
        timeout_s : float
            Stop attempting to autoexpose after ``timeout_s`` seconds.
        verbose : bool
            Whether to print exposure updates.

        Returns
        --------
        float
            Resulting exposure in seconds.
        """
        # Parse exposure_bounds_s
        if exposure_bounds_s is None:
            if self.exposure_bounds_s is None:
                exposure_bounds_s = (0, np.inf)
            else:
                exposure_bounds_s = self.exposure_bounds_s

        # Parse window
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

        # Initialize loop
        set_val = 0.5 * self.bitresolution
        exp = self.get_exposure()
        im_mean = np.mean(self.get_images(average_count, flush=True), 0)
        im_max = np.amax(im_mean[wyi:wyf, wxi:wxf])

        # Calculate the error as a percent of the camera's bitresolution
        err = np.abs(im_max - set_val) / self.bitresolution
        t = time.perf_counter()

        # Loop until timeout expires or we meet tolerance
        while err > tol and time.perf_counter() - t < timeout_s:
            # Clip exposure steps to 0.5x -> 2x
            exp = exp / np.amax([0.5, np.amin([(im_max / set_val), 2])])
            exp = np.amax([exposure_bounds_s[0], np.amin([exp, exposure_bounds_s[1]])])
            self.set_exposure(exp)
            im_mean = np.mean(self.get_images(average_count, flush=True), 0)
            im_max = np.amax(im_mean[wyi:wyf, wxi:wxf])
            err = np.abs(im_max - set_val) / self.bitresolution

            if verbose:
                print(exp, im_max)

        exp_fin = exp * 2 * set_fraction

        # The loop targets 50% of resolution
        if set_fraction != 0.5:  # Sets for full dynamic range
            self.set_exposure(exp_fin)

        return exp_fin

    def autofocus(self, z_get, z_set, z_list=None, plot=False):
        """
        Uses an FFT contrast metric to find optimal focus when scanning over some variable
        ``z``. This ``z`` often takes the form of a vertical stage to position a sample precisely
        at the plane of imaging of a lens or objective. The contrast metric works particularly
        well when combined with a projected spot array hologram.

        Parameters
        ----------
        z_get : lambda
            Gets the current position of the focusing stage. Should return a ``float``.
        z_set : lambda
            Sets the position of the focusing stage to a given ``float``.
        z_list : array_like or None
            ``z`` values to sweep over during search.
            Defaults (when ``None``) to ``numpy.linspace(-4,4,16)``.
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
            [z_list[np.argmax(counts)], np.max(counts)-np.min(counts), np.min(counts), 100]
        )

        try:
            popt, _ = curve_fit(
                lorentzian,
                z_list,
                counts,
                jac=lorentzian_jacobian,
                ftol=1e-5,
                p0=popt0,
            )
            z_opt = popt[0]
            c_opt = popt[1] + popt[2]
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
            plt.plot(z_list, lorentzian(z_list, *popt0))
            lfit = None
            try:
                lfit = lorentzian(z_list, *popt)
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

def _view_continuous(cameras, cmap=None, facecolor=None, dpi=300):
    """
    Continuously get camera frames and plot them. Intended for use in jupyter notebooks.
    Activate ``%matplotlib notebook`` before calling this function. This method
    does not halt, exit with a keyboard interrupt.

    Important
    ~~~~~~~~~
    This is probably going to get replaced with a :mod:`pyglet` interface for viewing
    realtime camera outputs while cameras loaded into python.

    Parameters
    ----------
    cameras : list of :class:`Camera`
        The cameras to view continuously.
    cmap
        See :meth:`matplotlib.pyplot.imshow`.
    facecolor
        See :meth:`matplotlib.pyplot.figure`.
    dpi
        See :meth:`matplotlib.pyplot.figure`. Default is 300.
    """
    # Get camera information.
    cam_count = len(cameras)
    cams_max_height = cams_max_width = 0
    for (cam_idx, cam) in enumerate(cameras):
        cam_height = cam.shape[0]
        cam_width = cam.shape[1]
        cams_max_height = max(cams_max_height, cam_height)
        cams_max_width = max(cams_max_width, cam_width)

    # Create figure.
    plt.ion()
    figsize = np.array((cam_count * cams_max_width, cams_max_height)) * 2 ** -9
    fig, axs = plt.subplots(
        1, cam_count, figsize=figsize, facecolor=facecolor, dpi=dpi
    )
    axs = np.reshape(axs, cam_count)
    fig.tight_layout()
    fig.show()
    fig.canvas.draw()
    for cam_idx in range(cam_count):
        axs[cam_idx].tick_params(direction="in")

    # Plot continuously.
    while True:
        for cam_idx in range(cam_count):
            cam = cameras[cam_idx]
            ax = axs[cam_idx]
            img = cam.get_image()
            ax.clear()
            ax.imshow(img, interpolation=None, cmap=cmap)
        fig.canvas.draw()
        fig.canvas.flush_events()
