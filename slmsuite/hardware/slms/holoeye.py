"""
**(NotImplemented)** Hardware control for Holoeye SLMs.
This is a partial skeleton that can be completed if desired.
"""
import os
import warnings
from .slm import SLM

try:  # Load Holoeye's SDK module.
    from holoeye import slmdisplaysdk
except ImportError:
    slmdisplaysdk = None
    warnings.warn("slmdisplaysdk not installed. Install to use Holoeye slms.")

class Holoeye(SLM):
    """
    Interfaces with Holoeye SLMs via the ``slmdisplaysdk``.

    Attributes
    ----------
    slm_lib : TODO
        Object handle for the Holoeye SLM.
    sdk_path : str
        Path of the Blink SDK folder.
    """

    def __init__(
        self,
        verbose=True,
        wav_um=1,
        pitch_um=(8,8),
        **kwargs
    ):
        r"""
        Initialize SLM and attributes.

        Parameters
        ----------
        wav_um : float
            Wavelength of operation in microns. Defaults to 1 um.
        pitch_um : (float, float)
            Pixel pitch in microns. Defaults to 8 micron square pixels.
        **kwargs
            See :meth:`.SLM.__init__` for permissible options.

        Note
        ~~~~
        These arguments, which ultimately are used to instantiate the :class:`.SLM` superclass,
        may be more accurately filled by calling the SLM's SDK functions.
        See the other implemented SLM subclasses for examples.
        """
        if slmdisplaysdk is None:
            raise ImportError("slmdisplaysdk not installed. Install to use Holoeye slms.")

        # Get the SLM.
        if verbose: print("Creating SLM instance...", end="")
        self.slm_lib = slmdisplaysdk.SLMInstance()  # ?
        # self.slm_lib = slmdisplaysdk.SLMDisplay() # ?

        # Check version somehow.
        if self.slm_lib.requiresVersion(3):
            if verbose: print("failure")
            raise RuntimeError("TODO")
        if verbose: print("success")

        # Open the SLM.
        if verbose: print("Opening SLM...", end="")
        error = self.slm_lib.open()

        if error != slmdisplaysdk.ErrorCode.NoError:
            if verbose: print("failure")
            self._handle_error(error)
        if verbose: print("success")

        # Other possibilities to consider:
        # - Setting the SLM's operating wavelength (wav_um).
        # - Updating the SLM's phase table if necessary, and/or setting the design
        #   wavelength (wav_design_um).
        # - Setting the SLM's default settle time (abstract class SLM uses
        #   settle_time_s=0.3 seconds). This is important for experimental feedback to
        #   allow the SLM to settle before viewing the result on a camera.
        # - Checking for and saving the SLM parameters (height, width, etc).

        # Instantiate the superclass
        super().__init__(
            (self.slm_lib.width(), self.slm_lib.height()),
            bitdepth=self.slm_lib.depth(),
            wav_um=wav_um,
            pitch_um=pitch_um,
            **kwargs
        )

        # Zero the display using the superclass `set_phase()` function.
        self.set_phase(None)

    def _handle_error(self, error):
        if error != slmdisplaysdk.ErrorCode.NoError:
            raise RuntimeError(self.slm_lib.errorString(error))

    @staticmethod
    def info(verbose=True):
        """
        Discovers all SLMs detected by an SDK.
        Useful for a user to identify the correct serial numbers / etc.

        Parameters
        ----------
        verbose : bool
            Whether to print the discovered information.

        Returns
        --------
        list of str
            List of serial numbers or identifiers.
        """
        if slmdisplaysdk is None:
            raise ImportError("slmdisplaysdk not installed. Install to use Holoeye slms.")

        raise NotImplementedError()
        serial_list = get_serial_list()     # TODO: Fill in proper function.
        return serial_list

    def _set_phase_hw(self, phase):
        """
        Low-level hardware interface to set_phase ``phase`` data onto the SLM.
        When the user calls the :meth:`.SLM.set_phase` method of
        :class:`.SLM`, ``phase`` is error checked before calling
        :meth:`_set_phase_hw()`. See :meth:`.SLM._set_phase_hw` for further detail.
        """
        error = self.slm_lib.showData(phase, self.flags)
        self._handle_error(error)