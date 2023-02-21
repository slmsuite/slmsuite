"""
Defines SLM class for hardware control for Meadowlark SLMs.
Tested with Meadowlark (AVR Optics) P1920-400-800-HDMI-T.
~~~~
Note: Check that DLL files, etc. are in folder C:\Program Files\Meadowlark Optics\Blink 1920 HDMI\SDK.
~~~~
Last update: Feb. 15, 2023
Based on example code provided by Meadowlark (Sept. 12, 2019)
"""

import os
import numpy
from ctypes import *
import ctypes

my_path = "C:\\Program Files\\Meadowlark Optics\\Blink 1920 HDMI\\"

class SLM():
    def __init__(self):
        os.chdir("C:\\Program Files\\Meadowlark Optics\\Blink 1920 HDMI\\SDK")
        awareness = ctypes.c_int()
        errorCode = ctypes.windll.shcore.GetProcessDpiAwareness(0, ctypes.byref(awareness))
        print(awareness.value)
        errorCode = ctypes.windll.shcore.SetProcessDpiAwareness(2)
        success = ctypes.windll.user32.SetProcessDPIAware()
        cdll.LoadLibrary(my_path + "SDK\\Blink_C_wrapper")
        self.slm_lib = CDLL("Blink_C_wrapper")
        # Open the image generation library
        cdll.LoadLibrary(my_path + "SDK\\ImageGen")
        image_lib = CDLL("ImageGen")
        self.height = c_uint(1152).value
        self.width = c_uint(1920).value
        # Indicate that our images are 8 bit images instead of RGB
        self.is_eight_bit_image = c_uint(1)
        # Initialize the SDK. The requirements of Matlab, LabVIEW, C++ and Python are different, so pass
        # the constructor a boolean indicating if we are calling from C++/Python (true), or Matlab/LabVIEW (false)
        bCppOrPython = c_uint(1)
        self.slm_lib.Create_SDK(bCppOrPython)
        # Adjust pre- and post-ramp slopes for accurate voltage setting (otherwise, calibration is not properly implemented).
        # You may need a special version of the SDK sent to you from Meadowlark to have access to these parameters.
        self.slm_lib.SetPreRampSlope(20) # default is 7
        self.slm_lib.SetPostRampSlope(24) # default is 24
        print ("Blink SDK was successfully constructed")
        # Load a linear lookup table. You can replace this file with your own calibration once it's generated.
        self.slm_lib.Load_lut(my_path + "LUT Files\\linear.lut")
        # self.slm_lib.Load_lut(my_path + "LUT Files\\calibrated.lut")

    # A function to display a matrix on the SLM
    def display_matrix(self,matrix):
        ImageOne = numpy.empty([self.width*self.height], numpy.uint8, 'C')
        ImageOne = matrix.astype(numpy.uint8,order='C')
        self.slm_lib.Write_image(ImageOne.ctypes.data_as(POINTER(c_ubyte)), self.is_eight_bit_image)

    def __del__(self):
        self.slm_lib.Delete_SDK()


