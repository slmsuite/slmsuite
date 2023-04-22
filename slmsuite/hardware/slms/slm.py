"""
Abstract functionality for SLMs.
"""

import time
import numpy as np
from PIL import Image

from slmsuite.holography import toolbox
from slmsuite.misc.math import INTEGER_TYPES
from slmsuite.holography import analysis


class SLM:
    """
    Abstract class for SLMs.

    Attributes
    ------
    name : str
        Name of the SLM.
    shape : (int, int)
        Stores ``(height, width)`` of the SLM in pixels, the same convention as :attr:`numpy.ndarray.shape`.
    wav_um : float
        Operating wavelength targeted by the SLM in microns. Defaults to 780 nm.
    wav_design_um : float
        Design wavelength for which the maximum settable value corresponds to a
        :math:`2\pi` phase shift.
        Defaults to :attr:`wav_um` if passed ``None``.

        Tip
        ~~~
        :attr:`wav_design_um` is useful for using, for instance, an SLM designed
        at 1064 nm for an application at 780 nm by using only a fraction (780/1064)
        of the full dynamic range. It is especially useful for SLMs which do not have builtin
        capability to change their voltage lookup tables (e.g. Thorlabs).
        Even so, the max lookup wavelength (:attr:`wav_design_um`) could be set larger
        than :attr:`wav_um` should the user want to have a phase range larger than
        :math:`2\pi`, for SLMs with lookup table capability.

    phase_scaling : float
        Wavelength normalized to the phase range of the SLM. See :attr:`wav_design_um`.
        Determined by ``phase_scaling = wav_um / wav_design_um``.
    bitdepth : int
        Depth of SLM pixel well in bits. This is useful for converting the floats which
        the user provides to the ``bitdepth``-bit ints that the SLM reads (see the
        private method :meth:`_phase2gray`).
    bitresolution : int
        Stores ``2 ** bitdepth``.
    settle_time_s : float
        Delay in seconds to allow the SLM to settle. This is mostly useful for applications
        requiring high precision. This delay is applied if the user flags ``settle``
        in :meth:`write()`. Defaults to .3 sec for precision.
    dx_um : float
        x pixel pitch in um.
    dy_um : float
        See :attr:`dx_um`.
    dx : float
        Normalized x pixel pitch ``dx_um / wav_um``.
    dy : float
        See :attr:`dx`.
    x_grid : numpy.ndarray<float> (height, width)
        Coordinates of the SLM's pixels in wavelengths
        (see :attr:`wav_um`, :attr:`dx`, :attr:`dy`)
        measured from the center of the SLM.
        Of size :attr:`shape`. Produced by :meth:`numpy.meshgrid`.
    y_grid
        See :attr:`x_grid`.
    measured_amplitude : numpy.ndarray or None
        Amplitude measured on the SLM via
        :meth:`~slmsuite.hardware.cameraslms.FourierSLM.wavefront_calibrate()`.
        Of size :attr:`shape`.
        Also see :meth:`set_measured_amplitude_analytic()` to set :attr:`measured_amplitude`
        without wavefront calibration. Defaults to ``None`` when no correction is provided.
    phase_correction : numpy.ndarray or None
        Phase correction devised for the SLM by
        :meth:`~slmsuite.hardware.cameraslms.FourierSLM.wavefront_calibrate`.
        Of size :attr:`shape`. Defaults to ``None`` when no correction is provided.
    phase : numpy.ndarray
        Displayed data in units of phase delay (normalized).
    display : numpy.ndarray
        Displayed data in SLM units (integers).
    """
    def __init__(
        self,
        width,
        height,
        bitdepth=8,
        name="SLM",
        wav_um=1,
        wav_design_um=None,
        dx_um=1,
        dy_um=1,
        settle_time_s=0.3,
    ):
        """
        Initialize SLM.

        Parameters
        ----------
        width, height
            See :attr:`shape`.
        bitdepth
            See :attr:`bitdepth`. Defaults to 8.
        name
            See :attr:`name`.
        wav_um
            See :attr:`wav_um`.
        wav_design_um
            See :attr:`wav_design_um`.
        dx_um
            See :attr:`dx_um`.
        dy_um
            See :attr:`dy_um`.
        settle_time_s
            See :attr:`settle_time_s`.
        """
        self.name = name
        self.shape = (int(height), int(width))

        # By default, target wavelength is the design wavelength
        self.wav_um = wav_um
        if wav_design_um is None:
            self.wav_design_um = wav_um
        else:
            self.wav_design_um = wav_design_um

        # Multiplier for when the target wavelengths differ from the design wavelength.
        self.phase_scaling = self.wav_um / self.wav_design_um

        # Resolution of the SLM.
        self.bitdepth = bitdepth
        self.bitresolution = 2 ** bitdepth

        # time to delay after writing (allows SLM to stabilize).
        self.settle_time_s = settle_time_s

        # Spatial dimensions
        self.dx_um = dx_um
        self.dy_um = dy_um

        self.dx = dx_um / self.wav_um
        self.dy = dy_um / self.wav_um

        # Make normalized coordinate grids.
        xpix = (width - 1) *  np.linspace(-.5, .5, width)
        ypix = (height - 1) * np.linspace(-.5, .5, height)
        self.x_grid, self.y_grid = np.meshgrid(self.dx * xpix, self.dy * ypix)

        # Phase and amplitude corrections.
        self.phase_correction = None
        self.measured_amplitude = None

        # Decide dtype
        if self.bitdepth <= 8:
            dtype = np.uint8
        else:
            dtype = np.uint16

        # Display caches for user reference.
        self.phase = np.zeros(self.shape)
        self.display = np.zeros(self.shape, dtype=dtype)

    def close(self):
        """Abstract method to close the SLM and delete related objects."""
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

    def load_vendor_phase_correction(self, file_path):
        """
        Abstract method to load vendor-provided phase correction from file,
        setting :attr:`~slmsuite.hardware.slms.slm.SLM.phase_correction`.
        By default, a bitmap is read in and 
        Subclasses should implement vendor-specific routines for loading and
        interpreting the file.

        Parameters
        ----------
        file_path : str
            File path for the vendor-provided phase correction.

        Returns
        ----------
        numpy.ndarray
            :attr:`~slmsuite.hardware.slms.slm.SLM.phase_correction`,
            the vendor-provided phase correction.
        """
        phase_correction = self.bitresolution - 1 - np.array(Image.open(file_path), dtype=float)

        if phase_correction.ndim != 2:
            raise ValueError("Expected 2D image; found shape {}.".format(phase_correction.shape))
        
        phase_correction *= 2 * np.pi / (self.phase_scaling * self.bitresolution)

        # Deal with correction shape
        file_shape_error = np.sign(np.array(phase_correction.shape) - np.array(self.shape))

        if np.abs(np.diff(file_shape_error)) > 1:
            raise ValueError(
                "Note sure how to pad or unpad correction shape {} to SLM shape {}."
                .format(phase_correction.shape, self.shape)
            )
        
        if np.any(file_shape_error > 1):
            self.phase_correction = toolbox.unpad(phase_correction, self.shape)
        elif np.any(file_shape_error < 1):
            self.phase_correction = toolbox.pad(phase_correction, self.shape)
        else:
            self.phase_correction = phase_correction

        return self.phase_correction

    def _write_hw(self, phase):
        """
        Abstract method to communicate with the SLM. Subclasses **should** overwrite this.
        :meth:`write()` contains error checks and overhead, then calls :meth:`_write_hw()`.

        Parameters
        ----------
        phase
            See :meth:`write`.
        """
        raise NotImplementedError()

    def write(
        self,
        phase,
        phase_correct=True,
        settle=False,
    ):
        r"""
        Checks, cleans, and adds to data, then sends the data to the SLM and
        potentially waits for settle. This method calls the SLM-specific private method
        :meth:`_write_hw()` which transfers the data to the SLM.

        Warning
        ~~~~~~~
        Subclasses implementing vendor-specific software *should not* overwrite this
        method. Subclasses *should* overwrite :meth:`_write_hw()` instead.

        Caution
        ~~~~~~~
        The sign on ``phase`` is flipped before converting to integer data. This is to
        convert between
        the 'increasing value ==> increasing voltage (= decreasing phase delay)' convention in most SLMs and
        :mod:`slmsuite`'s 'increasing value ==> increasing phase delay' convention.
        As a result, zero phase will appear entirely white (255 for an 8-bit SLM), and increasing phase
        will darken the displayed pattern.
        If integer data is passed, this data is displayed directly and the sign is *not* flipped.

        Important
        ~~~~~~~~~
        The user does not need to wrap (e.g. :mod:`numpy.mod(data, 2*numpy.pi)`) the passed phase data,
        unless they are pre-caching data for speed (see below).
        :meth:`.write()` uses optimized routines to wrap the phase (see the
        private method :meth:`_phase2gray()`).
        Which routine is used depends on :attr:`phase_scaling`:

         - :attr:`phase_scaling` is one.

            Fast bitwise integer modulo is used. Much faster than the other routines which
            depend on :meth:`numpy.mod()`.

         - :attr:`phase_scaling` is less than one.

            In this case, the SLM has **more phase tuning range** than necessary.
            If the data is within the SLM range ``[0, 2*pi/phase_scaling]``, then the data is passed directly.
            Otherwise, the data is wrapped by :math:`2\pi` using the very slow :meth:`numpy.mod()`.
            Try to avoid this in applications where speed is important.

         - :attr:`phase_scaling` is more than one.

            In this case, the SLM has **less phase tuning range** than necessary.
            Processed the same way as the :attr:`phase_scaling` is less than one case, with the
            important exception that phases (after wrapping) between ``2*pi/phase_scaling`` and
            ``2*pi`` are set to zero. For instance, a sawtooth blaze would be truncated at the tips.

        Caution
        ~~~~~~~
        After scale conversion, data is ``floor()`` ed to integers with ``np.copyto``, rather than
        rounded to the nearest integer (``np.around()`` equivalent). While this is
        irrelevant for the average user, it may be significant in some cases.
        If this behavior is undesired consider either: :meth:`write()` integer data
        directly or modifying the behavior of the private method :meth:`_phase2gray()` in
        a pull request. We have not been able to find an example of ``np.copyto``
        producing undesired behavior, but will change this if such behavior is found.

        Parameters
        ----------
        phase : numpy.ndarray or None
            Phase data to display in units of :math:`2\pi`,
            unless the passed data is of integer type and the data is applied directly.

             - If ``None`` is passed to :meth:`.write()`, data is zeroed.
             - If the array has a larger shape than the SLM shape, then the data is
               cropped to size in a centered manner
               (:attr:`~slmsuite.holography.toolbox.unpad`).
             - If integer data is passed with the same type as :attr:`display`
               (``np.uint8`` for <=8-bit SLMs, ``np.uint16`` otherwise),
               then this data is **directly** passed to the
               SLM, without going through the "phase delay to grayscale" conversion
               defined in the private method :meth:`_phase2gray`. In this situation,
               ``phase_correct`` and non-zero ``blaze_vector`` are **ignored**.
               This is error-checked to validate that bits with greater significance than the
               bitdepth of the SLM are zeroed (e.g. the final 6 bits of 16 bit data for a
               10-bit SLM). Integer data with type different from :attr:`display` leads
               to a TypeError.

            Usually, an **exact** stored copy of the data passed by the user under
            ``phase`` is stored in the attribute :attr:`phase`.
            However, in cases where :attr:`phase_scaling` not one, this
            copy is modified to include how the data was wrapped. If the data was
            cropped, then the cropped data is stored, etc. If integer data was passed, the
            equivalent floating point phase is computed and stored in the attribute :attr:`phase`.
        phase_correct : bool
            Whether or not to add :attr:`~slmsuite.hardware.slms.slm.SLM.phase_correction` to ``phase``.
        settle : bool
            Whether to sleep for :attr:`~slmsuite.hardware.slms.slm.SLM.settle_time_s`.

        Returns
        -------
        numpy.ndarray
           :attr:`~slmsuite.hardware.slms.slm.SLM.display`, the integer data sent to the SLM.

        Raises
        ------
        TypeError
            If integer data is incompatible with the bitdepth or if the passed phase is
            otherwise incompatible (not a 2D array or smaller than the SLM shape, etc).
        """
        # Helper variable to speed the case where phase is None.
        zero_phase = False

        # Parse phase.
        if phase is None:
            # Zero the phase pattern.
            self.phase.fill(0)
            zero_phase = True
        else:
            # Make sure the array is an ndarray.
            phase = np.array(phase)

        if phase is not None and isinstance(phase, INTEGER_TYPES):
            # Check the type.
            if phase.dtype != self.display.dtype:
                raise TypeError("Unexpected integer type {}. Expected {}.".format(phase.dtype, self.display.dtype))

            # If integer data was passed, check that we are not out of range.
            if np.any(phase >= self.bitresolution):
                raise TypeError("Integer data must be within the bitdepth ({}-bit) of the SLM.".format(self.bitdepth))

            # Copy the pattern and unpad if necessary.
            if phase.shape != self.shape:
                np.copyto(self.display, toolbox.unpad(phase, self.shape))
            else:
                np.copyto(self.display, phase)

            # Update the phase variable with the integer data that we displayed.
            self.phase = 2 * np.pi - self.display * (2 * np.pi / self.phase_scaling / self.bitresolution)
        else:
            # If float data was passed (or the None case).
            # Copy the pattern and unpad if necessary.
            if phase is not None:
                if self.phase.shape != self.shape:
                    np.copyto(self.phase, toolbox.unpad(self.phase, self.shape))
                else:
                    np.copyto(self.phase, phase)

            # Add phase correction if requested.
            if phase_correct and self.phase_correction is not None:
                self.phase += self.phase_correction
                zero_phase = False

            # Pass the data to self.display.
            if zero_phase:
                # If None was passed and neither phase_correct nor blaze_vector were
                # passed, then use a faster method.
                self.display.fill(0)
            else:
                # Turn the floats in phase space to integer data for the SLM.
                self.display = self._phase2gray(self.phase, out=self.display)

        # Write!
        self._write_hw(self.display)

        # Optional delay.
        if settle:
            time.sleep(self.settle_time_s)

        return self.display

    def _phase2gray(self, phase, out=None):
        r"""
        Helper function to convert an array of phases (units of :math:`2\pi`) to an array of
        :attr:`~slmsuite.hardware.slms.slm.SLM.bitresolution` -scaled and -cropped integers.
        This is used by :meth:`write()`. See special cases described in :meth:`write()`.

        Parameters
        ----------
        phase : numpy.ndarray
            Array of phases in radians.
        out : numpy.ndarray
            Array to store integer values scaled to SLM voltage, i.e. for in-place
            operations.
            If ``None``, an appropriate array will be allocated.

        Returns
        -------
        out
        """
        if out is None:
            out = np.zeros(self.shape, dtype=self.display.dtype)

        if self.phase_scaling == 1:
            # Prepare the 2pi -> integer conversion factor and convert.
            factor = -(self.bitresolution / 2 / np.pi)
            phase *= factor

            # There is some randomness involved in casting positive floats to integers.
            # Avoid this by going all negative.
            maximum = np.amax(phase)
            if maximum >= 0:
                toshift = self.bitresolution * 2 * np.ceil(maximum / self.bitresolution)
                phase -= toshift

            # Copy and case the data to the output (usually self.display)
            np.copyto(out, phase, casting="unsafe")

            # Restore phase (usually self.phase) as these operations are in-place.
            phase *= 1 / factor

            # Shift by one so that phase=0 --> display=max. That way, phase will be more continuous.
            out -= 1

            # This part (along with the choice of type), implements modulo much faster than np.mod().
            if self.bitresolution != 8 and self.bitresolution != 16:
                active_bits_mask = int(self.bitresolution - 1)
                np.bitwise_and(out, active_bits_mask, out=out)
        else:
            # phase_scaling is not included in the scaling.
            factor = -(self.bitresolution * self.phase_scaling / 2 / np.pi)
            phase *= factor

            # Only if necessary, modulo the phase to remain within SLM bounds.
            if np.amin(phase) <= -self.bitresolution or np.amax(phase) > 0:
                # Minus 1 is to conform with the in-bound case.
                phase -= 1
                # np.mod is the slowest step. It could maybe be faster if phase is converted to
                # an integer beforehand, but there is an amount of risk for overflow.
                # For instance, a standard double can represent numbers far larger than
                # even a 64 bit integer. If this optimization is implemented, take care to
                # generate checks for the conversion to long integer / etc before the final
                # conversion to dtype of uint8 or uint16.
                np.mod(phase, self.bitresolution * self.phase_scaling, out=phase)
                phase +=  self.bitresolution * (1-self.phase_scaling)

                # Set values still out of range to zero.
                if self.phase_scaling > 1:
                    phase[phase < 0] = self.bitresolution-1
            else:
                # Go from negative to positive.
                phase += self.bitresolution-1

            # Copy and case the data to the output (usually self.display)
            np.copyto(out, phase, casting="unsafe")

            # Restore phase (though we do not unmodulo)
            phase *= 1 / factor

        return out

    def set_measured_amplitude_analytic(self, radius, units="norm"):
        """
        Sets :attr:`~slmsuite.hardware.slms.slm.SLM.measured_amplitude` used
        for hologram generation in the absence of a proper wavefront calibration.
        :class:`~slmsuite.hardware.cameraslms.FourierSLM` includes
        capabilities for wavefront calibration via
        :meth:`~slmsuite.hardware.cameraslms.FourierSLM.wavefront_calibrate`.
        This process also measures the amplitude of the source on the SLM
        and stores this in :attr:`~slmsuite.hardware.slms.slm.SLM.measured_amplitude`.
        :attr:`~slmsuite.hardware.slms.slm.SLM.measured_amplitude`
        is used for better refinement of holograms during numerical
        optimization. If one does not have a camera to use for
        :meth:`~slmsuite.hardware.cameraslms.FourierSLM.wavefront_calibrate`,
        this method allows the user to set an approximation of the source amplitude
        based on an assumed :math:`1/e` amplitude (:math:`1/e^2` power) Gaussian beam radius.

        Parameters
        ----------
        radius : float
            Radius in normalized units to assume for the source Gaussian beam.
        units : str in {"norm", "nm", "um", "mm", "m"}
            Units for the given radius.

        Returns
        --------
        numpy.ndarray
            :attr:`~slmsuite.hardware.slms.slm.SLM.measured_amplitude`.
        """
        # Convert the x and y grid to normalized units.
        if "norm" in units:
            dx = 1
            dy = 1
        else:
            if units == "m":
                factor = 1e6
            elif units == "mm":
                factor = 1e3
            elif units == "um":
                factor = 1
            elif units == "nm":
                factor = 1e-3
            else:
                raise RuntimeError("Did not recognize units '{}'".format(units))
            dx = factor * self.dx / self.dx_um
            dy = factor * self.dy / self.dy_um

        r2_grid = np.square(self.x_grid / dx) + np.square(self.y_grid / dy)

        self.measured_amplitude = np.exp(-r2_grid * (1 / radius ** 2))

        return self.measured_amplitude

    def _get_measured_amplitude(self):
        """Deals with the None case of measured_amplitude"""
        if self.measured_amplitude is None:
            return np.ones_like(self.shape)
        else:
            return self.measured_amplitude

    def point_spread_function_knm(self, padded_shape=None):
        """
        Fourier transforms the wavefront calibration's measured amplitude to find
        the expected diffraction-limited perfomance of the system in ``"knm"`` space.

        Parameters
        ----------
        padded_shape : (int, int) OR None
            The point spread function changes in resolution depending on the padding.
            Use this variable to provide this padding.
            If ``None``, do not pad.

        Returns
        -------
        numpy.ndarray
            The point spread function of shape ``padded_shape``.
        """
        nearfield = toolbox.pad(self._get_measured_amplitude(), padded_shape)
        farfield = np.abs(np.fft.fftshift(np.fft.fft2(np.fft.fftshift(nearfield), norm="ortho")))

        return farfield

    def spot_radius_kxy(self):
        """
        Approximates the expected radius of farfield spots in the ``"kxy"`` basis based on the near-field amplitude distribution :attr:`measured_amplitude`.

        Returns
        -------
        float
            Average radius of the farfield spot.
        """
        try:
            psf_nm = np.sqrt(analysis.image_variances(self._get_measured_amplitude())[:2])

            psf_kxy = np.mean(toolbox.convert_blaze_vector(
                np.reciprocal(8 * psf_nm),
                from_units="freq",
                to_units="kxy",
                slm=self,
                shape=self.shape
            ))
        except:
            psf_kxy = np.mean([1 / self.dx / self.shape[1], 1 / self.dy / self.shape[0]])

        return psf_kxy