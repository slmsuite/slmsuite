"""
Abstract camera functionality.
"""
import time
import asyncio
import warnings
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.optimize import curve_fit
from scipy.ndimage import zoom
import PIL
import io
from abc import ABC, abstractmethod

from slmsuite.hardware import _Picklable
from slmsuite.holography import analysis
from slmsuite.holography.toolbox import BLAZE_LABELS
from slmsuite.misc.fitfunctions import lorentzian, lorentzian_jacobian
from slmsuite.misc.math import REAL_TYPES
from slmsuite.holography.analysis import image_centroids, image_remove_field
from slmsuite.holography.analysis.files import _gray2rgb


class Camera(_Picklable, ABC):
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
    dtype : np.dtype
        Stores the type returned by :meth:`._get_image_hw()`.
        This value is cached upon initialization.
    pitch_um : (float, float) OR None
        Pixel pitch in microns.
    exposure_s : float
        Caches the last result of :meth:`.get_exposure()`. Can be used if the user wants to
        avoid the overhead of calling the method.
    exposure_bounds_s : (float, float) OR None
        Shortest and longest allowable integration in seconds.
    averaging : int OR None
        Default setting for averaging. See :meth:`.get_image()`.
    hdr : (int, int) OR None
        Default setting for multi-exposure High Dynamic Range imaging. See :meth:`.get_image()`.
    capture_attempts : int
        If the camera returns an error or exceeds a timeout,
        try again for a total of `capture_attempts` attempts.
        This is useful for resilience against errors that happen with low probability.
        Defaults to 5.
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
    last_image : numpy.ndarray OR None
        Last captured image. Note that this is a pointer to the same data that the user
        receives (to avoid copying overhead). Thus, if the user modifies the returned data,
        then this data will be modified also.
        This may be of :attr:`dtype`, or may be a float, depending on whether :attr:`hdr` is
        used and the type of :attr:`averaging`.
        Is ``None`` if no image has ever been taken.
    """
    _pickle = [
        "name",
        "shape",
        "bitdepth",
        "bitresolution",
        "pitch_um",
        "exposure_s",
        "exposure_bounds_s",
        "averaging",
        "hdr",
        "woi",
        "default_shape",
    ]
    _pickle_data = [
        "last_image",
    ]

    @abstractmethod
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

        # Variable for storing the last capture.
        self.last_image = None

        # Remember the name.
        self.name = str(name)

        # Set exposure information.
        self.exposure_bounds_s = None

        self.exposure_s = 1 # Default to 1s for Simulated cameras.
        self.exposure_s = self.get_exposure()

        # Set datatype variables.
        self.bitdepth = int(bitdepth)
        self.bitresolution = 2**bitdepth
        self.dtype = self._get_dtype()

        # Frame averaging variables.
        self.averaging = self._parse_averaging(averaging, preserve_none=True)
        self.hdr = self._parse_hdr(hdr, preserve_none=True)
        self._flush_iterations = 2  # Hidden variable

        # Spatial dimensions.
        if pitch_um is not None and not (np.isscalar(pitch_um) and pitch_um <= 0):
            if isinstance(pitch_um, REAL_TYPES):
                pitch_um = [pitch_um, pitch_um]
            self.pitch_um = np.squeeze(pitch_um)
            if len(self.pitch_um) != 2 or np.any(self.pitch_um <= 0):
                raise ValueError("Expected positive (float, float) for pitch_um")
            self.pitch_um = np.array([float(self.pitch_um[0]), float(self.pitch_um[1])])
        else:
            self.pitch_um = None

        # Placeholder for live viewer handle.
        self.viewer = None

    # Core methods - to be implemented by subclass.

    @abstractmethod
    def close(self):
        """
        Abstract method to close the camera and delete related objects.
        """
        raise NotImplementedError()

    def __del__(self):
        try:
            self.close()
        except:
            pass

    @staticmethod
    def info(verbose=True):
        """
        Abstract method to load information about what cameras are available.

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

    def get_exposure(self):
        """
        Get the frame integration time in seconds.
        Used in :meth:`.autoexposure()`.

        Returns
        -------
        float
            Integration time in seconds.
        """
        self.exposure_s = self._get_exposure_hw()
        return self.exposure_s

    def set_exposure(self, exposure_s):
        """
        Set the frame integration time in seconds.
        Used in :meth:`.autoexposure()`.

        Parameters
        ----------
        exposure_s : float
            The integration time in seconds.

        Returns
        -------
        float
            Set integration time in seconds.
        """
        self._set_exposure_hw(exposure_s)
        return self.get_exposure()

    @abstractmethod
    def _get_exposure_hw(self):
        """
        Abstract method to interface with hardware and get the frame integration time in seconds.
        Subclasses must implement this.
        """
        raise NotImplementedError(f"Camera {self.name} has not implemented _get_exposure_hw")

    @abstractmethod
    def _set_exposure_hw(self, exposure_s):
        """
        Abstract method to interface with hardware and set the exposure time in seconds.
        Subclasses must implement this.

        Parameters
        ----------
        exposure_s : float
            The integration time in seconds.
        """
        raise NotImplementedError(f"Camera {self.name} has not implemented _set_exposure_hw")

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
        Cycle the image buffer such that all new :meth:`.get_image()` calls yield fresh frames.
        Without this feature, optimizations could be working on outdated information.

        Defaults to calling :meth:`.get_image()` twice, though cameras can implement
        hardware-specific alternatives.

        Parameters
        ----------
        timeout_s : float
            The time in seconds to wait for each frame.
            The frame exposure time  is **added** to this timeout
            such that there is always enough time to expose.
        """
        for _ in range(self._flush_iterations):
            self._get_image_hw_tolerant(timeout_s=timeout_s+self.exposure_s)

    @abstractmethod
    def _get_image_hw(self, timeout_s):
        """
        Abstract method to capture camera images.

        Parameters
        ----------
        timeout_s : float
            The time in seconds to wait for the frame to be fetched.
            The frame exposure time  is **NOT added** to this timeout
            such that there is always enough time to expose.

        Returns
        -------
        numpy.ndarray
            Array of shape :attr:`~slmsuite.hardware.cameras.camera.Camera.shape`.
        """
        raise NotImplementedError(f"Camera {self.name} has not implemented _get_image_hw")

    def _get_out(self, image_count, out=None):
        # Preallocate memory if necessary
        out_shape = (int(image_count), self.default_shape[0], self.default_shape[1])
        if out is None:
            out = np.empty(out_shape, dtype=self.dtype)
        else:
            if out.shape != out_shape:
                raise ValueError(f"Expected out to be of shape {out_shape}. Found {out.shape}.")
            if out.dtype != self.dtype:
                raise ValueError(f"Expected out to be of type {self.dtype}. Found {out.dtype}.")
            out = np.array(out, copy=False, dtype=self.dtype)

        return out

    def _get_images_hw(self, image_count, timeout_s, out=None):
        """
        Abstract method to capture a series of image_count images using camera-specific
        batch acquisition features.

        Parameters
        ----------
        image_count : int
            Number of frames to batch collect.
        timeout_s : float
            The time in seconds to wait for **each** frame to be fetched.
            The frame exposure time  is **NOT added** to this timeout
            such that there is always enough time to expose.
        out : None OR numpy.ndarray
            Preallocated memory for in-place operations, if applicable.

        Returns
        -------
        numpy.ndarray
            Array of shape (image_count, :attr:`~slmsuite.hardware.cameras.camera.Camera.shape`).
        """
        # Preallocate memory if necessary
        out = self._get_out(image_count, out)

        for i in range(image_count):
            out[i, :, :] = self._get_image_hw_tolerant(
                timeout_s=timeout_s+self.exposure_s
            )

        return out

    # Capture methods one level of abstraction above _get_image_hw().

    def _get_image_hw_tolerant(self, *args, **kwargs):
        err = None

        for i in range(self.capture_attempts):
            try:
                return self._get_image_hw(*args, **kwargs)
            except Exception as e:
                if i > 0: warnings.warn(f"'{self.name}' _get_image_hw() failed on attempt {i+1}.")
                err = e

        raise err

    def _get_images_hw_tolerant(self, *args, **kwargs):
        e = None

        for i in range(self.capture_attempts):
            try:
                return self._get_images_hw(*args, **kwargs)
            except Exception as e:
                if i > 0: warnings.warn(f"'{self.name}' _get_images_hw() failed on attempt {i+1}.")
                err = e

        raise err

    def _get_dtype(self):
        try:
            self.dtype = np.dtype(
                np.array(
                    self._get_image_hw_tolerant(timeout_s=1)
                ).dtype
            )   # Future: check if cameras change dtype after init.
        except:
            if self.bitdepth > 16:
                self.dtype = np.dtype(float)
            elif self.bitdepth > 8:
                self.dtype = np.dtype(np.uint16)
            else:
                self.dtype = np.dtype(np.uint8)

        try:
            if self.dtype(0).nbytes * 8 < self.bitdepth:
                raise warnings.warn(
                    f"Camera '{self.name}' bitdepth of {self.bitdepth} does not conform "
                    f"with the image type {self.dtype} with {self.dtype.itemsize} bytes."
                )
        except:     # The above sometimes fails for non-numpy datatypes.
            pass

        return self.dtype

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

    def _get_averaging_dtype(self, averaging=None):
        """Returns the appropriate image datatype for ``averaging`` levels of averaging."""
        if averaging is None:
            if self.averaging is None:
                raise ValueError("Averaging is not enabled for this camera. Set the .averaging attribute.")
            else:
                averaging = self.averaging
        averaging = int(averaging)

        if averaging <= 0:
            raise ValueError("Cannot have negative averaging.")

        # Switch based on image type
        if self.dtype.kind == "i" or self.dtype.kind == "u":
            dtype_bitdepth = 8*self.dtype.type(0).nbytes

            # Remove depth for signed integer.
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
            The frame exposure time  is **added** to this timeout
            such that there is always enough time to expose.
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
            averaging_dtype = self._get_averaging_dtype(averaging)

            try:
                # Using the camera-specific batch method if available
                imgs = self._get_images_hw(
                    averaging, timeout_s=timeout_s+self.exposure_s
                ).astype(averaging_dtype)

                # Cast as the proper type so we can sum.
                img = np.sum(imgs, axis=0)
            except NotImplementedError:
                # Brute-force collection as a backup
                img = np.zeros(self.default_shape, dtype=averaging_dtype)

                for _ in range(averaging):
                    img += self._get_image_hw_tolerant(
                        timeout_s=timeout_s+self.exposure_s
                    ).astype(averaging_dtype)
        else:                   # Normal image
            img = self._get_image_hw_tolerant(
                timeout_s=timeout_s+self.exposure_s
            )

        # self.transform implements the flipping and rotating keywords passed to the
        # superclass constructor.
        if transform:
            img = self.transform(img)

        # Store the result locally.
        self.last_image = img

        # Push to viewer if active.
        if self.viewer is not None:
            if averaging > 1:
                self.viewer.render(img / averaging)
            else:
                self.viewer.render(img)

        return img

    def get_images(self, image_count, timeout_s=1, out=None, transform=True, flush=False):
        """
        Grab ``image_count`` images in succession.

        Important
        ~~~~~~~~~
        This method **does not** support averaging or HDR features.
        Rather, it just returns a series of raw images.

        Parameters
        ----------
        image_count : int
            Number of images to grab.
        timeout_s : float
            The time in seconds to wait **for each** frame to be fetched.
            The frame exposure time  is **added** to this timeout
            such that there is always enough time to expose.
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
        # Flush if desired.
        if flush:
            self.flush()

        # Grab images (no transformation)
        imgs = self._get_images_hw(
            image_count,
            timeout_s=timeout_s+self.exposure_s,
            out=out
        )

        # Transform if desired. Future: make more efficient.
        if transform:
            imgs_ = np.empty(
                (int(image_count), self.shape[0], self.shape[1]),
                dtype=self.dtype
            )
            for i in range(image_count):
                imgs_[i, :, :] = self.transform(imgs[i])

            imgs = imgs_

        # Store the result locally.
        self.last_image = imgs[-1]

        # Push to viewer if active.
        if self.viewer is not None:
            self.viewer.render(imgs[-1])

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
        overexposure_threshold = self.bitresolution/2,
        if self.averaging is not None:
            overexposure_threshold *= self.averaging

        # Make empty data and grab the original exposure time.
        original_exposure = self.get_exposure()
        imgs = np.zeros((exposures, self.shape[0], self.shape[1]), self.dtype)
        exposure_times = np.zeros((exposures,), dtype=float)

        for i in range(exposures):
            # FUTURE: record the set exposures and use these to do better analysis.
            exposure_times[i] = self.set_exposure(int(exposure_power ** i) * original_exposure)
            self.flush()    # Sometimes, cameras return bad frames after exposure change.
            imgs[i, :, :] = self.get_image(hdr=False, **kwargs)

            # Terminate the loop if our image is entirely overexposed.
            if np.all(imgs[i, :, :] > overexposure_threshold):
                continue

        # Reset exposure.
        self.set_exposure(original_exposure)

        if return_raw:
            return imgs, exposure_times
        else:
            img = self.get_image_hdr_analysis(
                imgs,
                overexposure_threshold=overexposure_threshold,
                exposure_power=exposure_times,
            )
            if np.max(img) >= self.bitresolution:
                warnings.warn("HDR image is overexposed.")
            # Store the result locally.
            self.last_image = img
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
        exposure_power : int or list of float
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
        if np.isscalar(exposure_power):
            exposure_power = int(exposure_power)
            exposure_times = np.power(exposure_power, imgs.shape[0])
        else:
            exposure_times = exposure_power
            if np.all(exposure_times <= 0):
                raise ValueError("exposure_times cannot all be non-positive.")
            exposure_times = exposure_times / np.min(exposure_times[exposure_times > 0])

        if overexposure_threshold is None:
            # Default to half exposure.
            overexposure_threshold = np.max(imgs) / 2

        img = None

        for i in range(imgs.shape[0]):
            img_current = imgs[i, :, :].astype(float)

            if i == 0:
                img = img_current
            elif exposure_times[i] > 0:
                # Overwrite data when greater precision is available.
                mask = img_current < overexposure_threshold
                img[mask] = img_current[mask] / exposure_times[i]

        return img

    # Display methods.

    def plot(self, image=None, limits=None, title="Image", ax=None, cbar=True):
        """
        Plots the provided image.

        Parameters
        ----------
        image : ndarray OR None OR bool
            Image to be plotted.
            If ``None``, grabs an image from the camera.
            If ``False``, uses the :attr:`.last_image` attribute.
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
        if image is False:
            image = self.last_image
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

    def live(self, activate=None, widgets=True, backend="ipython", **kwargs):
        """
        Creates and displays an IPython camera viewer.
        This viewer displays the result of :meth:`get_image()`
        or the last image of :meth:`get_images()` **whenever these methods are called**.
        Averaging and HDR are displayed with the same color scaling as without.

        If ``True`` is passed to the ``widgets`` argument, this viewer is accompanied by
        a series of `IPython widgets
        <https://ipywidgets.readthedocs.io/en/latest/examples/Widget%20List.html>`_
        in the form of slides and buttons
        for controlling the color scale, colormap, viewer scale, and live viewing.
        By toggling the ``Live`` widget button,
        this viewer can be used as a realtime camera monitor within the jupyter notebook.
        Note that any user-execution will block the monitoring loop.
        Regardless, any image polling during the blocked period will still update the viewer,
        which provides useful active feedback for what is happening during the execution.

        This limitation is imposed by the
        python Global Interpreter Lock (GIL) which restricts operation to a single thread,
        especially operation connecting to a diverse set of camera and SLM hardware.
        We use :mod:`asyncio` to allow the realtime monitoring loop to be
        interrupted by user-execution (e.g. running a cell in jupyter),
        blocking until the execution is finished.

        Running multiple viewers at once might not play nicely right now.

        Parameters
        ----------
        activate : bool OR None
            If ``True``, creates a live viewer in the current cell,
            destroying any other attached viewer.
            If ``False``, destroys  any other attached viewer.
            If ``None``, toggles the live viewer, destroying any attached viewer or
            creating one in the current cell if none is attached. Defaults to ``None``.
        widgets : bool
            If ``True``, also displays sliders and controls used to hone the display properties.
        backend : str
            Placeholder option for different types of viewers.
            The default is ``"ipython"``.
        **kwargs
            Options passed to the :class:`_CameraViewer` to customize the default settings.
            These features will be made less hidden in the future.
            Most things are customizable via these keywords. For instance, the user can pass
            a custom list of colormaps to appear in the widget dropdown as ``cmap_options=``.
        """
        if backend != "ipython":
            raise ValueError(
                f"'{backend}' not recognized; "
                "'ipython' is currently the only supported .live() backend."
            )

        try:
            from ipywidgets import Image
            from IPython.display import display
        except ImportError:
            raise ImportError("jupyter must be installed to use .live().")

        if (self.viewer is None and activate is None) or activate:
            if self.viewer is not None:
                self.viewer.close()

            self.viewer = _CameraViewer(
                self,
                widgets,
                backend,
                **kwargs
            )
        elif self.viewer is not None and (activate is None or not activate):
            self.viewer.close()
            self.viewer = None

    # Other helper methods.

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
            Stop attempting adjusting exposure after ``timeout_s`` seconds.
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
        Uses a FFT contrast metric to find optimal focus when scanning over some variable
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


class _CameraViewer:
    """
    Hidden class for live camera viewing enabled by ipython widgets.
    """
    def __init__(
            self,
            cam,
            widgets,
            backend="ipython",
            live=False,
            min=None,
            max=None,
            log=False,
            cmap=True,
            scale=1,
            border=None,
            cmap_options=[
                "default", "gray", "Blues", "turbo",
                'viridis', 'plasma', 'inferno', 'magma', 'cividis'
            ],
            crosshair=False,
            centroid=False,
        ):
        self.cam = cam
        self.backend = backend

        # Parse range.
        if min is None:
            min = 0
        if max is None:
            max = cam.bitresolution-1
        range = [min, max]
        range = [np.min(range), np.max(range)]

        if cmap is True: cmap = "default"
        if cmap is False: cmap = "grayscale"

        # Parse scale
        scale = 2 ** np.round(np.log2(scale))

        self.state = {
            "backend" : backend,
            "live" : live,
            "range" : range,
            "log" : bool(log),
            "cmap" : cmap,
            "scale" : scale,
            "border" : border,
            "cmap_options" : cmap_options,
            "center_crosshair" : crosshair,
            "centroid_crosshair" : centroid,
        }

        self.task = None
        self.widgets = {}
        if widgets: self.init_widgets()
        self.init_image()

    def parse(self, img=None):
        if img is not None:
            self.prev_img = img
        if self.prev_img is None:
            return  # Nothing to render.

        # Downscaling can happen before intensive operations.
        if self.state["scale"] < 1:
            img = zoom(
                self.prev_img,
                [self.state["scale"], self.state["scale"]] + ([1] if len(self.prev_img.shape) == 3 else []),
                order=1
            )
        else:
            img = np.copy(self.prev_img)

        if self.state["centroid_crosshair"]:
            img_median_subtract = image_remove_field([img], deviations=None)
            cx, cy = np.rint(
                (
                    np.squeeze(image_centroids(img_median_subtract)) + np.flip(img.shape) / 2
                ) * (self.state["scale"] if self.state["scale"] > 1 else 1)
            ).astype(int)

        # Scale intensity of image
        r = np.array(self.state["range"]).astype(img.dtype)
        img = np.clip(img, r[0], r[1])
        img -= r[0]
        d = r[1] - r[0]

        if self.state["log"]:
            # clip to avoid log(0)
            img = (np.log10(np.clip(img, 1, np.inf)) / np.log10(d+1))

        # Make image color
        rgb = _gray2rgb(
            img,
            cmap=self.state["cmap"],
            lut=d,
            normalize=False,
            border=self.state["border"]
        )

        # Upscaling can happen after intensive operations.
        if self.state["scale"] > 1:
            rgb = zoom(rgb, (1, self.state["scale"], self.state["scale"], 1), order=0)

        # Add crosshair at the median-subtracted centroid (center of mass) position.
        if self.state["centroid_crosshair"]:
            rgb[:, :, cx, :3] = 255 - rgb[:, :, cx, :3]
            rgb[:, cy, :, :3] = 255 - rgb[:, cy, :, :3]

        # Finally, add crosshair in the center.
        if self.state["center_crosshair"]:
            rgb[:, :, int(rgb.shape[2]/2), :3] = 127 - rgb[:, :, int(rgb.shape[2]/2), :3]
            rgb[:, int(rgb.shape[1]/2), :, :3] = 127 - rgb[:, int(rgb.shape[1]/2), :, :3]

        buff = io.BytesIO()
        rgb = PIL.Image.fromarray(rgb[0])
        rgb.save(buff, format="png")

        return buff.getvalue()

    def render(self, img=None):
        try:
            self.image.value = self.parse(img)
        except Exception as e:
            with self.widgets["output"]:
                print(str(e))

    def update(self, event):
        with self.widgets["output"]:
            self.widgets["output"].clear_output(wait=True)
        for key in ["range", "log", "cmap", "scale", "live", "center_crosshair", "centroid_crosshair"]:
            self.state[key] = self.widgets[key].value

        self.render()

    def live(self, event=None):
        state = self.state["live"] = self.widgets["live"].value

        loop = asyncio.get_running_loop()

        if self.task is not None:
            try:
                self.task.cancel()
            except:
                pass

        if not state:
            self.task = None
        else:
            self.task = loop.create_task(self.live_loop())

    async def live_loop(self):
        while self.state["live"]:
            self.cam.get_image()
            await asyncio.sleep(0.01)

    def on_click(self, event):
        coord = np.array([event["x"], event["y"]])
        with self.widgets["output"]:
            self.widgets["output"].clear_output(wait=True)
            print(np.round(coord / self.state["scale"]).astype(int))

    def autorange(self, event):
        if self.prev_img is not None:
            range = [np.min(self.prev_img), np.max(self.prev_img)]
            self.state["range"] = self.widgets["range"].value = range

        self.render()

    def init_image(self):
        from ipywidgets import Image
        from IPython.display import display

        self.image = Image(value=self.parse(self.cam.get_image()), format="png")
        self.image.on_click = self.on_click
        display(self.image)

    def init_widgets(self):
        from ipywidgets import HTML, IntRangeSlider, ToggleButton, Button, Checkbox, Dropdown, FloatLogSlider, Output, Layout

        item_layout = Layout(width="auto")
        range_layout = Layout(width="70%")

        self.widgets = {
            "name" : HTML(
                value=f"<b>{self.cam.name}</b>",
                description="Viewing",
                tooltip="Name of the camera.",
                layout=item_layout,
            ),
            "live" : ToggleButton(
                value=self.state["live"],
                description="Live",
                tooltip="Toggle an asyncio loop to poll images from the camera.",
                layout=item_layout,
            ),
            "range" : IntRangeSlider(
                value=self.state["range"],
                min=0,
                max=self.cam.bitresolution-1,
                step=1,
                description="Range",
                tooltip="Color scale of the plot.",
                layout=range_layout,
            ),
            "autorange" : Button(
                description="AutoRange",
                tooltip="Scale the plot to the minimum and maximum of the current image.",
                layout=item_layout,
            ),
            "log" : Checkbox(
                value=self.state["log"],
                description="Logarithmic",
                tooltip="Toggle logarithmic scaling of the current plot.",
                layout=item_layout,
            ),
            "center_crosshair" : Checkbox(
                value=self.state["center_crosshair"],
                description="Center Crosshair",
                tooltip="Toggle a crosshair centered on the image.",
                layout=item_layout,
            ),
            "centroid_crosshair" : Checkbox(
                value=self.state["centroid_crosshair"],
                description="Centroid Crosshair",
                tooltip="Toggle a crosshair at the median-subtracted centroid (center of mass) of the image.",
                layout=item_layout,
            ),
            "cmap" : Dropdown(
                options=self.state["cmap_options"],
                value=self.state["cmap"],
                description="Colormap",
                tooltip="Choose the colormap to use for display.",
                layout=item_layout,
            ),
            "scale" : FloatLogSlider(
                value=self.state["scale"],
                base=2,
                min=-3, # 12.5%
                max=3,  # 800%
                step=1,
                description="Scale",
                tooltip="Scale the image by powers of two.",
                layout=item_layout,
            ),
            "output": Output()
        }

        for k, w in self.widgets.items():
            if k == "autorange":
                w.on_click(self.autorange)
            elif k == "live":
                w.observe(self.live, "value")
            else:
                w.observe(self.update, "value")

        from ipywidgets import HBox, VBox
        from IPython.display import display

        # self.widgets["layout"] = VBox([
        #     HBox([
        #         self.widgets["name"],
        #         self.widgets["cmap"],
        #         self.widgets["log"],
        #         self.widgets["scale"],
        #     ]),
        #     HBox([
        #         self.widgets["range"],
        #         self.widgets["autorange"],
        #     ]),
        #     self.widgets["output"],
        # ])

        box_layout1 = Layout(
            display="flex",
            flex_flow="auto",
            align_items="stretch",
            width="70%"
        )
        box_layout2 = Layout(
            display="flex",
            flex_flow="auto",
            align_items="stretch",
            width="30%"
        )

        self.widgets["layout"] = HBox([
            VBox(
                [
                    HBox([
                        self.widgets["name"],
                    ]),
                    HBox([
                        self.widgets["cmap"],
                        self.widgets["log"],
                        self.widgets["center_crosshair"],
                        self.widgets["centroid_crosshair"],
                    ]),
                    HBox([
                        self.widgets["range"],
                    ]),
                    self.widgets["output"],
                ],
                layout=box_layout1,
            ),
            VBox(
                [
                    self.widgets["live"],
                    self.widgets["scale"],
                    self.widgets["autorange"],
                ],
                layout=box_layout2,
            )
        ])

        display(self.widgets["layout"])

    def close(self):
        try:
            self.task.cancel()
            self.task = None
        except:
            pass

        for w in self.widgets.values():
            w.close()
        self.image.close()

        del self
