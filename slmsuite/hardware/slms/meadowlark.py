"""
Hardware control for Meadowlark SLMs.
Tested with Meadowlark (AVR Optics) P1920-400-800-HDMI-T.

Note
~~~~
Check that DLL files, etc. are in folder C:\Program Files\Meadowlark Optics\Blink 1920
HDMI\SDK.
"""
import os
import ctypes
import numpy as np

from .slm import SLM

# TODO: generalize this directory.
my_path = "C:\\Program Files\\Meadowlark Optics\\Blink 1920 HDMI\\"

class Meadowlark(SLM):
    """
    Interfaces with Meadowlark SLMs.

    Attributes
    ----------
    slm_lib : ctypes.CDLL
        Connection to the Meadowlark library.
    """

    def __init__(self, verbose=True, **kwargs):
        r"""
        Initializes an instance of a Meadowlark SLM.

        Arguments
        ------
        verbose : bool
            Whether to print extra information.
        kwargs
            See :meth:`.SLM.__init__` for permissible options.
        """
        # TODO: Determine whether image_lib should be included.
        # TODO: Determine how to connect several SLMs at the same time.
        # TODO: Handle errors correctly.
        # TODO: Check LUT issues and SetPreRampSlope / etc.
        # TODO: Implement other SDK functions.
        os.chdir(my_path + "SDK")

        awareness = ctypes.c_int()
        errorCode = ctypes.windll.shcore.GetProcessDpiAwareness(0, ctypes.byref(awareness))
        print(awareness.value)

        errorCode = ctypes.windll.shcore.SetProcessDpiAwareness(2)
        success = ctypes.windll.user32.SetProcessDPIAware()

        # Open the SLM library
        if verbose: print("Constructing blink SDK...", end="")

        ctypes.cdll.LoadLibrary(my_path + "SDK\\Blink_C_wrapper")
        self.slm_lib = ctypes.CDLL("Blink_C_wrapper")

        # Open the image generation library
        # ctypes.cdll.LoadLibrary(my_path + "SDK\\ImageGen")
        # image_lib = ctypes.CDLL("ImageGen")

        # Indicate that our images are 8 bit images instead of RGB
        # self.is_eight_bit_image = ctypes.c_uint(1)

        # Initialize the SDK. The requirements of Matlab, LabVIEW, C++ and Python are different, so pass
        # the constructor a boolean indicating if we are calling from C++/Python (true), or Matlab/LabVIEW (false)
        b_cpp_or_python = ctypes.c_uint(1)
        self.slm_lib.Create_SDK(b_cpp_or_python)

        # Adjust pre- and post-ramp slopes for accurate voltage setting (otherwise, calibration is not properly implemented).
        # You may need a special version of the SDK sent to you from Meadowlark to have access to these parameters.
        self.slm_lib.SetPreRampSlope(20) # default is 7
        self.slm_lib.SetPostRampSlope(24) # default is 24

        if verbose: print("success")

        # Load a linear lookup table. You can replace this file with your own calibration once it's generated.
        self.slm_lib.Load_lut(my_path + "LUT Files\\linear.lut")
        # self.slm_lib.Load_lut(my_path + "LUT Files\\calibrated.lut")

        super().__init__(
            1920,
            1152,
            bitdepth=8,
            name="Meadowlark",
            dx_um=8,
            dy_um=8,
            **kwargs
        )

        self.write(None)

    @staticmethod
    def info(verbose=True):
        """
        **(NotImplemented)** Discovers the names of all the displays.

        Parameters
        ----------
        verbose : bool
            Whether to print the discovered information.

        Returns
        --------
        None
        """
        return

    def close(self):
        """See :meth:`.SLM.close`."""
        self.slm_lib.Delete_SDK()

    def _write_hw(self, phase):
        """See :meth:`.SLM._write_hw`."""
        # TODO: Need to test on hardware, but this is can certainly be accomplished
        # more efficiently (e.g. by not reallocating memory each frame).
        buffer = np.empty([self.shape[0] * self.shape[1]], np.uint8, 'C')
        buffer = phase.astype(np.uint8, order='C')

        self.slm_lib.Write_image(
            buffer.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte)),
            ctypes.c_uint(1)    # Is 8-bit
        )
