"""
Hardware control for Texas Instruments PLMs via ti_plm.

This class wraps the :class:`ScreenMirrored` SLM for display and integrates
parameterization and phase processing from :class:`ti_plm.PLM`.

Note
~~~~
This class requires the `ti_plm` package and a valid PLM device configuration.

Attributes
----------
ti_plm : ti_plm.PLM
    Instance of the ti_plm PLM class for parameterization and phase processing.
"""

# TODO: pyglet screen size is different than slm size
# TODO: fast display w/ pyglet
# TODO: 

import numpy as np
import warnings
from slmsuite.hardware.slms.screenmirrored import ScreenMirrored

try:
    from ti_plm import PLM as TIPLM
except ImportError:
    TIPLM = None
    warnings.warn(
        "The 'ti_plm' package must be installed to enable full PLM functionality. ",
        ImportWarning,
    )


class PLM(ScreenMirrored):
    """
    Interfaces with Texas Instruments' Phase Light Modulators (PLMs).

    This class combines :class:`ScreenMirrored` for display with the
    parameterization and phase processing features of :class:`ti_plm.PLM`.
    Only PLM-specific features are documented here; see :class:`ScreenMirrored`
    for display and windowing details.

    Parameters
    ----------
    display_number : int
        Monitor number for frame to be instantiated upon.
    bitdepth : int, optional
        Bitdepth of the SLM. Defaults to 4.
    verbose : bool, optional
        Whether or not to print extra information.
    **kwargs
        Additional arguments for :class:`ScreenMirrored`.
    """

    def __init__(
        self,
        name,
        display_number,
        bitdepth=4,
        verbose=True,
        **kwargs
    ):
        if TIPLM is None:
            raise ImportError(
                "The 'ti_plm' package is required for PLM support but is not installed."
            )
        # Use TIPLM.from_db to create the device
        self.ti_plm = TIPLM.from_db(name)
        # Parse pitch_um from the device's pitch (in meters) and convert to microns
        pitch_um = tuple(np.array(self.ti_plm.pitch) * 1e6)
        super().__init__(display_number, bitdepth=bitdepth, verbose=verbose, pitch_um=pitch_um, **kwargs)

    def process_phase_map(self, phase_map, replicate_bits=True, enforce_shape=True):
        """
        Processes an array of phase data into a bitmap for display on this PLM device.
        Uses :meth:`ti_plm.PLM.process_phase_map`.

        Parameters
        ----------
        phase_map : numpy.ndarray
            Array containing phase data.
        replicate_bits : bool, optional
            Whether to multiply the final bitplane by 255. Defaults to True.
        enforce_shape : bool, optional
            Whether to check input phase map shape. Defaults to True.

        Returns
        -------
        numpy.ndarray
            Quantized and electrode-mapped data for the PLM device.
        """
        return self.ti_plm.process_phase_map(
            phase_map, replicate_bits=replicate_bits, enforce_shape=enforce_shape
        )

    @staticmethod
    def bitpack(bitmaps):
        """
        Combine multiple binary CGHs into a single 8- or 24-bit image using :meth:`ti_plm.PLM.bitpack`.

        Parameters
        ----------
        bitmaps : numpy.ndarray
            Array of bitmaps to pack.

        Returns
        -------
        numpy.ndarray
            Packed bitmap image.
        """
        return TIPLM.bitpack(bitmaps)

    @staticmethod
    def get_device_list():
        """
        Get a list of all available PLM devices in the database.

        Returns
        -------
        list
            List of available device names.
        """
        return TIPLM.get_device_list()

    @classmethod
    def from_db(cls, name, display_number, bitdepth=4, verbose=True, **kwargs):
        """
        Create a PLM instance by searching the database for a given device name.

        Parameters
        ----------
        name : str
            Device name to search for in database.
        display_number : int
            Monitor number for frame to be instantiated upon.
        bitdepth : int, optional
            Bitdepth of the SLM. Defaults to 4.
        verbose : bool, optional
            Whether or not to print extra information.
        **kwargs
            Additional arguments for :class:`ScreenMirrored`.

        Returns
        -------
        PLM
            Instance of the PLM class.
        """
        return cls(
            name,
            display_number,
            bitdepth=bitdepth,
            verbose=verbose,
            **kwargs,
        )
