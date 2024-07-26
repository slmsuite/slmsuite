"""
Abstract camera functionality.
"""
import time
import warnings
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.optimize import curve_fit

from slmsuite.holography import analysis
from slmsuite.holography.toolbox import BLAZE_LABELS
from slmsuite.misc.fitfunctions import lorentzian, lorentzian_jacobian
from slmsuite.misc.math import REAL_TYPES


class Camera():
    """
    Abstract class for cameras.
    Comes with transformations, averaging and HDR,
    and helper functions like :meth:`.autoexpose()`.

    Attributes
    ----------
    name : str
        Camera identifier.
    shape : (int, int)
        Stores ``(height, width)`` of the camera in pixels, the same convention as
        :meth:`numpy.shape`.
    bitdepth : int
        Depth of a camera pixel well in bits.
    bitresolution : int
        Stores ``2**bitdepth``.
    dtype : type
        Stores the type returned by :meth:`._get_image_hw()`.
        This value is cached upon initialization.
    averaging : int OR None
        Default setting for averaging. See :meth:`.get_image()`.
    hdr : (int, int) OR None
        Default setting for multi-exposure High Dynamic Range imaging. See :meth:`.get_image()`.
    capture_attempts : int
        If the camera returns an error or exceeds a timeout, 
        try again for a total of `capture_attempts` attempts.
        This is useful for resilience against errors that happen with low probability.
        Defaults to 5.
    pitch_um : (float, float) OR None
        Pixel pitch in microns.
    exposure_bounds_s : (float, float) OR None
        Shortest and longest allowable integration in seconds.
    woi : tuple
        WOI (window of interest) in ``(x, width, y, height)`` form.

        Warning
        ~~~~~~~
        This feature is less fleshed out than most. There may be issues
        (e.g. :meth:`.get_image()` with the ``averaging`` or ``hdr`` flags).
    default_shape : tuple
        Default ``shape`` of the camera before any WOI or transform changes are made.
    transform : function
        Flip and/or rotation operator specified by the user in :meth:`__init__`.
        The user is expected to apply this transform to the matrix returned in
        :meth:`get_image()`. Note that WOI changes are applied on the camera hardware
        **before** this transformation.
    """

    def __init__(
        self,
        resolution,
        bitdepth=8,
        pitch_um=None,
        name="camera",
        averaging=None,
        capture_attempts=5,
        hdr=None,
        rot="0",
        fliplr=False,
        flipud=False,
    ):
        """
        Initializes a camera.

        In addition to the other class attributes, accepts the following parameters
        to set :attr:`transform`. See :meth:`~slmsuite.holography.analysis.get_orientation_transformation()`.

        Parameters
        ----------
        resolution
            The width and height of the camera in ``(width, height)`` form.

            Important
            ~~~~~~~~~
            This is the opposite of the numpy ``(height, width)``
            convention stored in :attr:`shape`.
        bitdepth
            See :attr:`bitdepth`.
        pitch_um : (float, float) OR None
            Fill in extra information about the pixel pitch in ``(dx_um, dy_um)`` form
            to use additional calibrations.
        name : str
            Defaults to ``"camera"``.
        averaging : int or None
            Number of frames to average. Used to increase the effective bit depth of a camera by using
            pre-quantization noise (e.g. dark current, read-noise, etc.) to "dither" the pixel output
            signal. If ``None``, no averaging is performed.
        hdr : int OR (int, int) OR None OR False
            Exposure information for `Multi-exposure High Dynamic Range (HDR) imaging
            <https://en.wikipedia.org/wiki/Multi-exposure_HDR_capture>`_
        capture_attempts : int
            If the camera returns an error or exceeds a timeout, 
            try again for a total of `capture_attempts` attempts.
            This is useful for resilience against errors that happen with low probability.
            Defaults to 5.
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
        (width, height) = resolution

        # Set shape, depending upon transform.
        if rot in ("90", 1, "270", 3):
            self.shape = (width, height)
            self.default_shape = (width, height)
        else:
            self.shape = (height, width)
            self.default_shape = (height, width)

        # Parse capture_attempts.
        self.capture_attempts = int(capture_attempts)
        if capture_attempts <= 0:
            raise ValueError("capture_attempts must be positive.")

        # Create image transformation.
        self.transform = analysis.get_orientation_transformation(rot, fliplr, flipud)

        # Update WOI information.
        self.woi = (0, width, 0, height)
        try:
            self.set_woi()
        except NotImplementedError:
            pass

        # Set other useful parameters
        self.name = str(name)
        self.bitdepth = int(bitdepth)
        self.bitresolution = 2**bitdepth
        self.dtype = self._get_dtype()

        # Frame averaging
        self.averaging = self._parse_averaging(averaging, preserve_none=True)
        self.hdr = self._parse_hdr(hdr, preserve_none=True)

        # Spatial dimensions
        if pitch_um is not None:
            if isinstance(pitch_um, REAL_TYPES):
                pitch_um = [pitch_um, pitch_um]
            self.pitch_um = np.squeeze(pitch_um)
            if (len(self.pitch_um) != 2):
                raise ValueError("Expected (float, float) for pitch_um")
            self.pitch_um = np.array([float(self.pitch_um[0]), float(self.pitch_um[1])])
        else:
            self.pitch_um = None

        # Default to None, allow subclass constructors to fill.
        self.exposure_bounds_s = None

    # Core methods - to be implemented by subclass.

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
        if verbose:
            print(".info() NotImplemented.")
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
        raise NotImplementedError()

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

    def _get_image_hw(self, timeout_s=1):
        """
        Abstract method to capture camera images.

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

    def _get_images_hw(self, image_count, timeout_s=1, out=None):
        """
        Abstract method to capture a series of image_count images using camera-specific
        batch acquisition features.

        Parameters
        ----------
        image_count : int
            Number of frames to batch collect.
        timeout_s : float
            The time in seconds to wait for the frame to be fetched.

        Returns
        -------
        numpy.ndarray
            Array of shape (n_frames, :attr:`~slmsuite.hardware.cameras.camera.Camera.shape`).
        """
        raise NotImplementedError()

    # Capture methods one level of abstraction above _get_image_hw().

    def _get_image_hw_tolerant(self, *args, **kwargs):
        err = None

        for i in range(self.capture_attempts):
            try:
                return self._get_image_hw(*args, **kwargs)
            except Exception as e:
                if i > 0: warnings.warn(f"'{self.name}' _get_image_hw() failed on attempt {i}.")
                err = e

        raise err

    def _get_images_hw_tolerant(self, *args, **kwargs):
        e = None
        
        for i in range(self.capture_attempts):
            try:
                return self._get_images_hw(*args, **kwargs)
            except Exception as e:
                if i > 0: warnings.warn(f"'{self.name}' _get_images_hw() failed on attempt {i}.")
                err = e

        raise err

    def _get_dtype(self):
        try:
            self.dtype = np.array(self._get_image_hw_tolerant()).dtype   # Future: check if cameras change this after init.
        except NotImplementedError:
            if self.bitdepth > 16:
                self.dtype = float
            elif self.bitdepth > 8:
                self.dtype = np.uint16
            else:
                self.dtype = np.uint8

        try:
            if self.dtype(0).nbytes * 8 < self.bitdepth:
                raise warnings.warn(
                    f"Camera '{self.name}' bitdepth of {self.bitdepth} does not conform "
                    f"with the image type {self.dtype} with {self.dtype.itemsize} bytes."
                )
        except:     # The above sometimes fails for non-numpy datatypes.
            pass

    def _parse_averaging(self, averaging=None, preserve_none=False):
        """
        Helper function to get a valid averaging.
        """
        if averaging is None:
            if preserve_none:
                return None
            if not hasattr(self, "averaging") or self.averaging is None:
                averaging = 1
            else:
                averaging = self.averaging
        elif averaging is False:
            averaging = 1
        averaging = int(averaging)

        if averaging <= 0:
            raise ValueError("Cannot have negative averaging.")

        return averaging

    def _parse_hdr(self, exposures=None, preserve_none=False):
        """
        Helper function to get a valid hdr parameters.
        """
        # Parse inputs
        if exposures is None:
            if preserve_none:
                return None
            if not hasattr(self, "hdr") or self.hdr is None:
                (exposures, exposure_power) = (1, 0)
            else:
                (exposures, exposure_power) = self._parse_hdr(self.hdr)
        elif exposures is False:
            exposures = 1
            exposure_power = 0
        elif np.isscalar(exposures):
            exposure_power = 2
        else:
            (exposures, exposure_power) = exposures

        # Force int so we have a chance of exposure aligning with camera clock.
        return (int(exposures), int(exposure_power))

    def get_averaging_dtype(self, averaging=None):
        """Returns the appropriate image datatype for ``averaging`` levels of averaging."""
        if averaging is None:
            averaging = self.averaging
        averaging = int(averaging)

        if averaging <= 0:
            raise ValueError("Cannot have negative averaging.")

        # Switch based on image type
        if self.dtype.kind == "i" or self.dtype.kind == "u":
            dtype_bitdepth = self.dtype.nbytes

            # Remove depth for signed integeter.
            if self.dtype.kind == "i":
                dtype_bitdepth -= 1

            extra_bits = int(np.rint(np.log2(averaging)))

            if self.bitdepth + extra_bits <= dtype_bitdepth:
                # If we can sustain the averaging with the current type, continue.
                return self.dtype
            else:
                # Otherwise, force floating point.
                return float
        elif self.dtype.kind == "f":
            # Return floating point.
            return self.dtype
        else:
            raise ValueError(f"Datatype {self.dtype} does not make sense as a camera return.")

    def get_image(self, timeout_s=1, transform=True, hdr=None, averaging=None):
        """
        Capture, process, and return images from a camera.

        Tip
        ~~~
        This function includes two advanced capture options:

        -   `Multi-exposure High Dynamic Range (HDR) imaging
            <https://en.wikipedia.org/wiki/Multi-exposure_HDR_capture>`_
            and
        -   Software frame averaging (integrating).

        These methods can aid the user in capturing more precise data, beyond the
        default raw (and bitdepth-limited) output of the camera.

        Parameters
        ----------
        timeout_s : float
            The time in seconds to wait for the frame to be fetched.
        transform : bool
            Whether or not to transform the output image according to
            :attr:`~slmsuite.hardware.cameras.camera.Camera.transform`.
            Defaults to ``True``.
        hdr : int OR (int, int) OR None OR False
            Exposure information for `Multi-exposure High Dynamic Range (HDR) imaging
            <https://en.wikipedia.org/wiki/Multi-exposure_HDR_capture>`_
            If ``None``, the value of :attr:`hdr` is used.
            If ``False``, HDR is not used no matter the state of :attr:`hdr`.

            See Also
            ~~~~~~~~
            :meth:`.get_image_hdr()` for more information.

        averaging : int OR None OR False
            If ``int``, the number of frames to average over.
            If ``None``, the value of :attr:`averaging` is used.
            If ``False``, averaging is not used no matter the state of :attr:`averaging`.

            Tip
            ~~~
            The datatype is promoted to float if necessary but otherwise tries to stick
            with the default datatype.
            For instance, a camera that returns a 12-bit image as a 16-bit type has four
            more bits to use for averaging, i.e. :math:`2^4 = 16` possible averages without
            risk of overflow.
            Requesting more than 16 averages would cause the return type to be promoted
            to ``float``.

            Note
            ~~~~
            Averaging is a bit of a misnomer as the true functionality is to sum or
            integrate the images. This is done such that integer datatypes (useful for
            memory compactness) can still be returned; a general mean would need to be
            floating point.

        Returns
        -------
        numpy.ndarray of int OR float
            Array of shape :attr:`~slmsuite.hardware.cameras.camera.Camera.shape`.
        """
        # Parse acquisition options.
        averaging = self._parse_averaging(averaging)
        (exposures, exposure_power) = self._parse_hdr(hdr)

        # Switch based on what imaging case we're in.
        if exposures > 1:       # Average many images with increasing exposure.
            return self.get_image_hdr(
                (exposures, exposure_power),
                timeout_s=timeout_s,
                transform=transform,
                averaging=averaging,
            )
        elif averaging > 1:     # Average many images.
            averaging_dtype = self.get_averaging_dtype(averaging)

            try:
                # Using the camera-specific batch method if available
                imgs = self._get_images_hw(
                    averaging, timeout_s=timeout_s
                ).astype(averaging_dtype)

                # Cast as the proper type so we can sum.
                img = np.sum(imgs, axis=0)
            except NotImplementedError:
                # Brute-force collection as a backup
                img = np.zeros(self.default_shape, dtype=averaging_dtype)

                for _ in range(averaging):
                    img += self._get_image_hw_tolerant(timeout_s=timeout_s).astype(averaging_dtype)
        else:                   # Normal image
            img = self._get_image_hw_tolerant(timeout_s=timeout_s)

        # self.transform implements the flipping and rotating keywords passed to the
        # superclass constructor.
        if transform:
            img = self.transform(img)

        return img

    def get_images(self, image_count, timeout_s=1, out=None, transform=True, flush=False):
        """
        Grab ``image_count`` images in succession.

        Important
        ~~~~~~~~~
        This method does not support averaging or HDR features.
        Rather, it just returns a series of raw images.

        Parameters
        ----------
        image_count : int
            Number of images to grab.
        timeout_s : float
            The time in seconds to wait **for each** frame to be fetched.
        out : None OR numpy.ndarray
            If not ``None``, output data in this memory. Useful to avoid excessive allocation.
        transform : bool
            Whether or not to transform the output image according to
            :attr:`~slmsuite.hardware.cameras.camera.Camera.transform`.
            Defaults to ``True``.
        flush : bool
            Whether to flush before grabbing.

        Returns
        -------
        numpy.ndarray
            Array of shape ``(image_count, height, width)``.
        """
        # Preallocate memory if necessary
        out_shape = (int(image_count), self.default_shape[0], self.default_shape[1])
        if out is None:
            imgs = np.empty(out_shape, dtype=self.dtype)
        else:
            if out.shape != out_shape:
                raise ValueError(f"Expected out to be of shape {out_shape}. Found {out.shape}.")
            if out.dtype != self.dtype:
                raise ValueError(f"Expected out to be of type {self.dtype}. Found {out.dtype}.")
            imgs = np.array(out, copy=False, dtype=self.dtype)

        # Flush if desired.
        if flush:
            self.flush()

        # Grab images (no transformation)
        try:
            # Using the camera-specific method if available
            imgs = self._get_images_hw(image_count, timeout_s=timeout_s, out=imgs)
        except NotImplementedError:
            # Brute-force collection as a backup
            for i in range(image_count):
                imgs[i, :, :] = self._get_image_hw_tolerant(timeout_s=timeout_s)

        # Transform if desired.
        if transform:
            imgs_ = np.empty(
                (int(image_count), self.shape[0], self.shape[1]),
                dtype=self.dtype
            )
            for i in range(image_count):
                imgs_[i, :, :] = self.transform(imgs[i])

            imgs = imgs_

        return imgs

    def get_image_hdr(self, exposures=None, return_raw=False, **kwargs):
        r"""
        Often, the necessities of precision applications exceed the bitdepth of a
        camera. One way to recover High Dynamic Range (HDR) imaging is to use
        `multiple exposures <https://en.wikipedia.org/wiki/Multi-exposure_HDR_capture>`_
        each with increasing exposure time. Then, these images can be stitched together
        as floating-point data, omitting data which is under- or over- exposed.

        Tip
        ~~~
        This feature can be accessed in :meth:`.get_image()`
        using :attr:`hdr` or the ``hdr=`` flag.
        This function is exposed here also to reveal the raw data using ``return_raw=``
        and to draw attention to this useful feature.

        Caution
        ~~~~~~~
        Camera exposure is sometimes poorly defined. This might cause incorrect
        assumptions of the exposure.
        In general, a larger base exposure will produce more accurate results as a
        greater number of sample clock periods are rounded to for smaller relative variation.
        Future modifications to :meth:`get_image_hdr_analysis()` might improve image stitching.

        Parameters
        ----------
        exposures : int OR (int, int)
            The number of exposures to take.
            Each exposure increases in time multiplicatively from the base value
            (original :meth:`get_exposure()`) by a factor :math:`p`.
            The :math:`i\text{th}` image has exposure time :math:`\tau \times p^i`, zero-indexed.
            The default base of :math:`p = 2` leads to ``exposures`` being equivalent to
            `spots <https://en.wikipedia.org/wiki/Exposure_value>`_.
            This base can be changed to another number by instead passing a tuple, where
            the second ``int`` defines the desired base.
        return_raw : bool
            If ``True``, returns the raw data (stack of images with count ``exposures``)
            instead of the processed data. The data can be processed using :meth:`get_image_hdr_analysis`
        **kwargs
            Passed to :meth:`.get_image()`.

        Returns
        -------
        numpy.ndarray of float
            Array of shape :attr:`~slmsuite.hardware.cameras.camera.Camera.shape`.

            Important
            ~~~~~~~~~
            The scale of the returned image is the same as the original exposure.
        """
        (exposures, exposure_power) = self._parse_hdr(exposures)

        # Make empty data and grab the original exposure time.
        original_exposure = self.get_exposure()
        imgs = np.empty((exposures, self.shape[0], self.shape[1]))

        for i in range(exposures):
            self.set_exposure(int(exposure_power ** i) * original_exposure)
            self.flush()    # Sometimes, cameras return bad frames after exposure change.
            imgs[i, :, :] = self.get_image(hdr=False, **kwargs)

        # Reset exposure.
        self.set_exposure(original_exposure)
        self.flush()

        if return_raw:
            return imgs
        else:
            img = self.get_image_hdr_analysis(imgs, exposure_power=exposure_power, overexposure_threshold=self.bitresolution/2)
            if np.max(img) >= self.bitresolution:
                warnings.warn("HDR image is overexposed.")
            return img

    @staticmethod
    def get_image_hdr_analysis(imgs, overexposure_threshold=None, exposure_power=2):
        r"""
        Analyzes raw data for High Dynamic Range (HDR) imaging
        `multiple exposures <https://en.wikipedia.org/wiki/Multi-exposure_HDR_capture>`_
        each with increasing exposure time.

        Parameters
        ----------
        imgs : array_like
            Stack of images with increasing exposure.
        overexposure_threshold : float OR None
            For each image (except the first), data is thrown out if values are above
            this threshold. If ``None``, the threshold defaults to half the maximum.
        exposure_power : int
            Each exposure increases in time multiplicatively from the base value
            (original :meth:`get_exposure()`) by this factor :math:`p`. The :math:`i\text{th}` image has
            exposure time :math:`\tau \times p^i`, zero-indexed.
            The default value of ``2`` leads to ``exposures`` being equivalent to
            `spots <https://en.wikipedia.org/wiki/Exposure_value>`_.

        Returns
        -------
        numpy.ndarray of float
            Array of shape :attr:`~slmsuite.hardware.cameras.camera.Camera.shape`.

            Important
            ~~~~~~~~~
            The scale of the returned image is the same as the original exposure.
        """
        # Parse arguments
        exposure_power = int(exposure_power)
        if overexposure_threshold is None:
            # Default to half exposure.
            overexposure_threshold = np.max(imgs) / 2

        img = None

        for i in range(imgs.shape[0]):
            img_current = imgs[i, :, :].astype(float)

            if i == 0:
                img = img_current
            else:
                # Overwrite data when greater precision is available.
                mask = img_current < overexposure_threshold
                img[mask] = img_current[mask] / float(exposure_power ** i)

        return img

    # Other helper methods.

    def plot(self, image=None, limits=None, title="Image", ax=None, cbar=True):
        """
        Plots the provided image.

        Parameters
        ----------
        image : ndarray OR None
            Image to be plotted. If ``None``, grabs an image from the camera.
        limits : None OR float OR [[float, float], [float, float]]
            Scales the limits by a given factor or uses the passed limits directly.
        title : str
            Title the axis.
        ax : matplotlib.pyplot.axis OR None
            Axis to plot upon.
        cbar : bool
            Also plot a colorbar.

        Returns
        -------
        matplotlib.pyplot.axis
            Axis of the plotted image.
        """
        if image is None:
            self.flush()
            image = self.get_image()
        image = np.array(image, copy=(False if np.__version__[0] == '1' else None))

        if len(plt.get_fignums()) > 0:
            fig = plt.gcf()
        else:
            fig = plt.figure(figsize=(20,8))

        if ax is not None:
            plt.sca(ax)

        im = plt.imshow(image)
        ax = plt.gca()

        if cbar:
            cax = make_axes_locatable(ax).append_axes("right", size="2%", pad=0.05)
            fig.colorbar(im, cax=cax, orientation="vertical")

        # ax.invert_yaxis()
        ax.set_title(title)

        if limits is not None and limits != 1:
            if np.isscalar(limits):
                axlim = [ax.get_xlim(), ax.get_ylim()]

                centers = np.mean(axlim, axis=1)
                deltas = np.squeeze(np.diff(axlim, axis=1)) * limits / 2

                limits = np.vstack((centers - deltas, centers + deltas)).T
            elif np.shape(limits) == (2,2):
                pass
            else:
                raise ValueError(f"limits format {limits} not recognized; provide a scalar or limits.")

            ax.set_xlim(limits[0])
            ax.set_ylim(limits[1])

        if image.shape == self.shape:
            ax.set_xlabel(BLAZE_LABELS["ij"][0])
            ax.set_ylabel(BLAZE_LABELS["ij"][1])

        plt.sca(ax)

        return ax

    def autoexposure(
        self,
        set_fraction=0.5,
        tol=0.05,
        exposure_bounds_s=None,
        window=None,
        average_count=5,
        timeout_s=5,
        verbose=True,
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
                print("Reset exposure to %1.2fs; maximum image value = %d." % (exp, im_max))

        exp_fin = exp * 2 * set_fraction

        # The loop targets 50% of resolution
        if set_fraction != 0.5:  # Sets for full dynamic range
            self.set_exposure(exp_fin)

        return exp_fin

    def autofocus(self, get_z, set_z, z_list=None, plot=False):
        """
        Uses an FFT contrast metric to find optimal focus when scanning over some variable
        ``z``. This ``z`` often takes the form of a vertical stage to position a sample precisely
        at the plane of imaging of a lens or objective. The contrast metric works particularly
        well when combined with a projected spot array hologram.

        Parameters
        ----------
        get_z : function
            Gets the current position of the focusing stage. Should return a ``float``.
        set_z : function
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

        z_base = get_z()
        imlist = []
        z_list = z_list + z_base
        counts = np.zeros_like(z_list)

        set_z(z_list[0])

        for i, z in enumerate(z_list):
            print("Moving to " + str(z))
            set_z(z)

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
                axs[1].set_title(f"FFT\nFoM$ = \\int\\int $|FFT|$ / $max|FFT|$ = {fom_}$")
                axs[1].set_xticks([])
                axs[1].set_yticks([])
                plt.show()

        counts[0] = counts[1]

        popt0 = np.array(
            [z_list[np.argmax(counts)], np.max(counts) - np.min(counts), np.min(counts), 100]
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
        set_z(z_opt)

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
    for cam_idx, cam in enumerate(cameras):
        cams_max_height = max(cams_max_height, cam.shape[0])
        cams_max_width = max(cams_max_width, cam.shape[1])

    # Create figure.
    plt.ion()
    figsize = np.array((cam_count * cams_max_width, cams_max_height)) * 2**-9
    fig, axs = plt.subplots(1, cam_count, figsize=figsize, facecolor=facecolor, dpi=dpi)
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
