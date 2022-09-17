"""
Abstract functionality for SLMs.
"""

import time
import numpy as np
from slmsuite.holography.toolbox import blaze
from slmsuite.holography import toolbox


class SLM:
    """
    Abstract class for SLMs.

    Attributes
    ------
    name : str
        Name of the SLM.
    shape : (int, int)
        Stores ``(height, width)`` of the SLM in pixels, same form as :attr:`numpy.ndarray.shape`.
    wav_um : float
        Operating wavelength targeted by the SLM in microns.
    wav_design_um : float
        Design wavelength for which the maximum settable value corresponds to a
        2 pi phase shift. This is useful for using, for instance, an SLM designed
        at 1064 nm for an application at 780 nm by using only a fraction (780/1064)
        of the full dynamic range. Useful for SLMs which do not have builtin
        capability to change their voltage lookup tables (e.g. Thorlabs).
    wav_norm : float
        Wavelength normalized to the phase range of the SLM. Subclasses which pass
        ``wav_design_um`` should set ``wav_norm = wav_um / wav_design_um``
        or otherwise upon construction.
    bitdepth : int
        Depth of SLM pixel well in bits. This is useful for converting the floats which
        the user provides to the ``bitdepth``-bit ints that the SLM reads (see :meth:`_phase2gray`).
    bitresolution : int
        Stores ``2 ** bitdepth``.
    settle_time_s : float
        Delay in seconds to allow the SLM to settle. This is mostly useful for applications
        requiring high precision. This delay is applied if the user flags ``wait_for_settle``
        in :meth:`write()`.
    dx_um : float
        x pixel pitch in um.
    dy_um : float
        See :attr:`dx_um`.
    dx : float
        Normalized x pixel pitch ``dx_um / wav_um``.
    dy : float
        See :attr:`dx`.
    x_grid : numpy.ndarray of floats
        Point grid of the SLM's :attr:`shape` derived from :meth:`numpy.meshgrid` ing.
    y_grid : numpy.ndarray of floats
        See :attr:`x_grid`.
    flatmap : numpy.ndarray
        Phase correction to apply to the SLM, provided by, e.g. the slm vendor.
        Looks for correction upon intialization with :meth:`load_flatmap()`,
        but defaults to ``None`` if no correction is found.
    measured_amplitude : numpy.ndarray or None
        Amplitude measured on the SLM b
        :meth:`~slmsuite.hardware.cameraslms.FourierSLM.wavefront_calibrate()`.
        Of size :attr:`shape`.
        Also see :meth:`set_analytic_amplitude()` to set :attr:`measured_amplitude`
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
        bitdepth,
        name="SLM",
        wav_um=0.532,
        wav_design_um=None,
        dx_um=8,
        dy_um=8,
        settle_time_s=0.3,
        flatmap_path=None,
    ):
        """
        Initialize SLM.

        Parameters
        ----------
        width, height
            See :attr:`shape`.
        bitdepth
            See :attr:`bitdepth`.
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
        flatmap_path : str or None
            File path to vendor-provided wavefront calibration. If ``None``,
            no flatmap is loaded. Otherwise, passed to :meth:`load_flatmap`.
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
        self.wav_norm = self.wav_um / self.wav_design_um

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
        xpix = np.linspace(-(width - 1) / 2.0, (width - 1) / 2.0, width)
        ypix = np.linspace(-(height - 1) / 2.0, (height - 1) / 2.0, height)
        self.x_grid, self.y_grid = np.meshgrid(self.dx * xpix, self.dy * ypix)

        # Phase and amplitude corrections.
        self.phase_correction = None
        self.measured_amplitude = None
        self.flatmap = None
        if flatmap_path is not None:
            self.load_flatmap(flatmap_path)

        # Decide dtype
        if self.bitdepth <= 8:
            dtype = np.uint8
        else:
            dtype = np.uint16

        # Display caches for user reference.
        self.phase = np.zeros(self.shape)
        self.display = np.zeros(self.shape, dtype=dtype)

    def close(self):
        """Close the SLM and delete related objects."""
        raise NotImplementedError()

    def load_flatmap(self, file_path):
        """
        Load vendor-provided phase correction from file.
        Subclasses should implement vendor-specific flatmap loading.
        Otherwise, this function passes without error.

        Parameters
        ----------
        file_path : str
            File path for the vendor-provided flatmap.

        Returns
        ----------
        numpy.ndarray or None
            :attr:`~slmsuite.hardware.slms.slm.SLM.flatmap`, the vendor-provided phase correction, or None if none is loaded.
        """
        return self.flatmap

    def _write_hw(self, phase):
        """
        Abstract function to communicate with the SLM. Subclasses should overwrite this.
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
        flatmap=True,
        phase_correct=True,
        wait_for_settle=False,
        blaze_vector=None,
    ):
        r"""
        Checks, cleans, and adds to data, then sends the data to the SLM and
        potentially waits for settle. This function calls the SLM-specific private method
        :meth:`_write_hw()` which transfers the data to the SLM.

        Caution
        ~~~~~~~
        The sign on ``phase`` is flipped before converting to integer data. This is to
        convert between
        the 'increasing value ==> increasing voltage (= decreasing phase delay)' convention in most SLMs and
        :mod:`slmsuite`'s 'increasing value ==> increasing phase delay' convention.
        As a result, zero phase will appear entirely white (255 for an 8-bit SLM), and increasing phase
        will darken the displayed pattern.

        Important
        ~~~~~~~~~
        The user does not need to wrap (e.g. :meth:`numpy.mod(data, 2*np.pi)`) the passed phase data,
        unless they are pre-caching data for speed (see below).
        :meth:`.write()` uses optimized routines to wrap the phase (see :meth:`._phase2gray()`).
        Which routine is used depends on :attr:`wav_norm`:

        :attr:`wav_norm` is one.
          Fast bitwise integer modulo is used. Much faster than the other routines which
          depend on :meth:`numpy.mod()`.
        :attr:`wav_norm` is less than one.
          In this case, the SLM has **more phase tuning range** than necessary.
          If the data is within the SLM range ``[0, 2*pi/wav_norm]``, then the data is passed directly.
          Otherwise, the data is wrapped by :math:`2\pi` using the very slow :meth:`numpy.mod()`.
          Try to avoid this in applications where speed is important.
        :attr:`wav_norm` is more than one.
          In this case, the SLM has **less phase tuning range** than necessary.
          Processed the same way as the :attr:`wav_norm` is less than one case, with the
          important exception that phases (after wrapping) between ``2*pi/wav_norm`` and
          ``2*pi`` are set to zero. For instance, a sawtooth blaze would be truncated at the tips.

        Parameters
        ----------
        phase : numpy.ndarray or None
            Data to display in units of phase delay. Data must be larger than the SLM. If larger,
            the data is cropped to size in a centered manner. If ``None``, data is zeroed.
            Usually, this is an exact stored copy of the data passed by the user. However, in cases
            where :attr:`wav_norm` not one, this copy is modified to include how the data was wrapped.
        flatmap : bool
            Whether or not to add :attr:`~slmsuite.hardware.slms.slm.SLM.flatmap` to ``phase``.
        phase_correct : bool
            Whether or not to add :attr:`~slmsuite.hardware.slms.slm.SLM.phase_correction` to ``phase``.
        wait_for_settle : bool
            Whether to sleep for :attr:`~slmsuite.hardware.slms.slm.SLM.settle_time_s`.
        blaze_vector : (float, float)
            See :meth:`~slmsuite.holography.toolbox.blaze`.
            If ``None``, no blaze is applied.

        Returns
        ----------
        numpy.ndarray
           :attr:`~slmsuite.hardware.slms.slm.SLM.display`, the integer data sent to the SLM.
        """
        # Parse phase.
        if phase is None:
            # Zero the phase pattern.
            self.phase.fill(0)
        else:
            # Copy the pattern.
            # Unpad if necessary.
            if self.phase.shape != self.shape:
                np.copyto(self.phase, toolbox.unpad(self.phase, self.shape))
            else:
                np.copyto(self.phase, phase)

        # Add corrections if requested.
        if flatmap and self.flatmap is not None:
            self.phase += self.flatmap
        if phase_correct and self.phase_correction is not None:
            self.phase += self.phase_correction

        # Blaze if requested.
        if blaze_vector is not None and (blaze_vector[0] != 0 or blaze_vector[1] != 0):
            self.phase += blaze(self, blaze_vector)

        # Turn the floats in phase space to integer data for the SLM.
        self.display = self._phase2gray(self.phase, out=self.display)

        # Write!
        self._write_hw(self.display)

        # Optional delay.
        if wait_for_settle:
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
            Array to store integer values scaled to SLM voltage.
            If ``None``, an appropriate array will be allocated.

        Returns
        -------
        out
        """
        if out is None:
            out = np.zeros(self.shape, dtype=self.display.dtype)

        if self.wav_norm == 1:
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
            bw = int(self.bitresolution - 1)
            np.bitwise_and(out, bw, out=out)
        else:
            # wav_norm is not included in the scaling.
            factor = -(self.bitresolution * self.wav_norm / 2 / np.pi)
            phase *= factor

            # Only if necessary, modulo the phase to remain within SLM bounds.
            if np.amin(phase) <= -self.bitresolution or np.amax(phase) > 0:
                # Minus 1 is to conform with the in-bound case.
                phase -= 1
                np.mod(phase, self.bitresolution * self.wav_norm, out=phase)

                # Set values still out of range to zero.
                if self.wav_norm > 1:
                    phase[phase >= self.bitresolution] = 0
            else:
                # Go from negative to positive.
                phase += self.bitresolution-1

            # Copy and case the data to the output (usually self.display)
            np.copyto(out, phase, casting="unsafe")

            # Restore phase (though we do not unmodulo)
            phase *= 1 / factor

        return out

    def phase_wrapped(self):
        r"""
        Return the phase last written to the SLM (
        :attr:`~slmsuite.hardware.slms.slm.SLM.phase`), mod :math:`2\pi`.

        Returns
        -------
        numpy.ndarray
            :attr:`~slmsuite.hardware.slms.slm.SLM.phase`, mod 2pi.
        """
        return np.mod(self.phase, 2 * np.pi)

    def set_analytic_amplitude(self, radius_mm):
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
        this function allows the user to set an approximation of the source amplitude
        based on an assumed 1/e amplitude (1/e^2 power) Gaussian beam radius.

        Parameters
        ----------
        radius_mm : float
            Radius in millimeters to assume for a source Gaussian beam.

        Returns
        --------
        numpy.ndarray
            :attr:`~slmsuite.hardware.slms.slm.SLM.measured_amplitude`.
        """

        norm2mm = self.wav_um / 1e3

        r2_grid = norm2mm * (np.square(self.x_grid) + np.square(self.y_grid))

        self.measured_amplitude = np.exp(-r2_grid / radius_mm / radius_mm)

        return self.measured_amplitude
