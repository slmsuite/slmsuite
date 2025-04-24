"""
Hardware control for Holoeye SLMs.
Tested with Holoeye SLM ERIS-NIR-153.
Created for SLM Display SDK (Python) v4.0.0.

Important
~~~~
Check that the SLM Display SDK is in the default folder C:\\Program Files\\HOLOEYE Photonics\\SLM Display SDK (Python) v4.0.0
or otherwise add the installation folder to your PYTHON PATH as done below.
"""
import warnings
from .slm import SLM

# Import the SLM Display SDK
# If the SDK is not found, run the following lines in a python shell to define the module path for the Python API bindings
# import os
# import sys
# env_path = os.getenv("HEDS_4_0_PYTHON")
# if env_path is None or not os.path.isdir(env_path):
#     env_path = os.path.abspath("../..")
# importpath_api =  os.path.join(env_path, "api", "python")
# importpath_HEDS =  os.path.join(env_path, "examples")
# sys.path.append(importpath_api)
# sys.path.append(importpath_HEDS)

# Load Holoeye's SDK module.
try:
    import HEDS
    from hedslib.heds_types import *
except ImportError:
    HEDS = None
    warnings.warn("Holoeye SDK HEDS not installed. Install to use Holoeye SLMs.")

class Holoeye(SLM):
    """
    Interfaces with Holoeye SLMs via the the ``HEDS`` library.

    Attributes
    ----------
    preselect : str
        Preselect string for the SLM. Examples:
        - "index:0"  // select first SLM available in the system.
        - "name:pluto;serial:0001"  // select a PLUTO SLM with the serial number 0001.
        - "name:luna"  // select a LUNA SLM.
        - "serial:2220-0011"  // select a LUNA/2220 SLM just by passing the serial number.
    """

    def __init__(self,verbose=True,preselect=None,wav_um=1,pitch_um=(8,8),**kwargs):
        r"""
        Initializes an instance of a Holoeye SLM.

        Caution
        ~~~~~~~
        :class:`.Holoeye` defaults to 8 micron SLM pixel size.
        This is valid for the PLUTO and ERIS models, but not true for all!

        Arguments
        ---------
        verbose : bool
            Whether to print extra information.
        preselect : str
            Preselect string for the SLM. Examples:
            - "index:0"  // select first SLM available in the system.
            - "name:pluto;serial:0001"  // select a PLUTO SLM with the serial number 0001.
            - "name:luna"  // select a LUNA SLM.
            - "serial:2220-0011"  // select a LUNA/2220 SLM just by passing the serial number.
            - "connect://ipv4:port", e.g. "connect://127.0.0.1:6230" \\ connect to a manually started process  
        wav_um : float
            Wavelength of operation in microns. Defaults to 1 um.
        pitch_um : (float, float)
            Pixel pitch in microns. Defaults to 8 micron square pixels (PLUTO and ERIS pixel size).
        **kwargs
            See :meth:`.SLM.__init__` for permissible options.

        Important
        ~~~~~~~~
        The Holoeye SLM Display SDK must be installed and the path to the SDK must be added to the PYTHON PATH.
        Check the holoeye.py module file for instructions on how to do this.
        """
        
        if HEDS is None:
            raise ImportError("SDK HEDS not installed. Install to use Holoeye SLMs.")
        
        # Initialize the SDK and check that version 4.0 of the SDK is being used.
        error = HEDS.SDK.Init(4,0)
        self._handle_error(error)
        
        # Connect and open the SLM
        if verbose: print("Opening SLM ...", end="")
        self.preselect = preselect
        self.slm = HEDS.SLM.Init(preselect=self.preselect)
        self._handle_error(self.slm.errorCode())
        if verbose: print("Success!")
        
        #Set the SLM's operating wavelength (wav_um) in nm.
        error = self.slm.setWavelength(wav_um*1000)
        self._handle_error(error)
        
        #Check the SLM's parameters.
        assert (self.slm.pixelsize_um(),self.slm.pixelsize_um()) == pitch_um, "Given pixel size does not match the SLM specifications."
        width = self.slm.width_px()
        height = self.slm.height_px()

        # Instantiate the superclass
        super().__init__(
            (width, height),
            bitdepth=8,
            wav_um=wav_um,
            pitch_um=pitch_um,
            **kwargs
        )

        # Zero the display using the superclass `set_phase()` function.
        self.set_phase(None)

    def _handle_error(self, error):
        """
        Handles errors from the Holoeye SDK.
        Raises an exception if the error code is not HEDSERR_NoError.
        Parameters
        ----------
        error : int
            Error code returned by the Holoeye SDK.
        Raises
        ------
        AssertionError
            If the error code is not HEDSERR_NoError.
        """
        assert error == HEDSERR_NoError, HEDS.SDK.ErrorString(self.slm.errorCode())
        
    @staticmethod
    def info(verbose=True):
        """
        Discovers all SLMs detected by an SDK.
        Useful for a user to identify the correct serial numbers / etc.

        Parameters
        ----------
        verbose : bool
            Whether to print the discovered information.
        Raises
        ------
        NotImplementedError
        """
        raise NotImplementedError("This functionality is not supported by Holoeye. Use the EDID device detection GUI instead.")

    def close(self):
        """
        See :meth:`.SLM.close`.
        """
        error = self.slm.window().close()
        self._handle_error(error)

    def _set_phase_hw(self, phase):
        """
        Low-level hardware interface to set_phase ``phase`` data onto the SLM.
        When the user calls the :meth:`.SLM.set_phase` method of
        :class:`.SLM`, ``phase`` is error checked before calling
        :meth:`_set_phase_hw()`. See :meth:`.SLM._set_phase_hw` for further detail.
        """
        #2*pi is the standard phase assumed by Holoeye. The package slmsuite passes 8-bit greyscale to set_phase_hw
        error = self.slm.showPhaseData(phase, phase_unit=256)
        self._handle_error(error)

    def load_vendor_phase_correction(self, file):
        """
        Load phase correction provided by Holoeye from file,
        setting ``"phase"`` in :attr:`~slmsuite.hardware.slms.slm.SLM.source`.

        Parameters
        ----------
        file_path : str
            File path for the vendor-provided phase correction.
        """
        #Enable wavefront compensation visualization in SLM preview window and stay with SLM preview scale "Fit"
        error = self.slm.preview().setSettings(flags=HEDSSLMPF_ShowWavefrontCompensation,zoom=0.0)
        self._handle_error(error)
        # Load the wavefront compensation file
        error = self.slm.window().loadWavefrontCompensationFile(str(file))
        self._handle_error(error)
