"""
Hardware control for The Imaging Source Cameras via tisgrabber
tisgrabber is one of several different interfaces that The Imaging Source supports.
See https://github.com/TheImagingSource/IC-Imaging-Control-Samples/tree/master/Python/tisgrabber
This was tested at commit 7846b9e and Python 3.9 with DMK 27BUP031 camera
The tisgrabber dll and tisgrabber.py are needed. 
Please either install tisgrabber.py or have it in your current working directory.
"""

import ctypes
import tisgrabber as tis
import numpy as np
from slmsuite.hardware.cameras.camera import Camera

# Change this DLL path if necessary
DLL_PATH = "./tisgrabber_x64.dll"

class TISCamera(Camera):
    """
    The Imaging Source camera.

    Attributes
    ----------
    sdk : 
        ctypes.CDLL which this class uses. Shared among instances of :class:`TISCamera`.
    cam : 
        HGRABBER object to talk with the camera. See tisgrabber.h or tisgrabber documentation for more details
    width :
        The width of the current region of interest. Cached for convenience and also changes when roi changes.
    height :
        The height of the current region of interest. Cached for convenience and also changes when roi changes.
    vid_format :
        Caches the video format currently set by the user if known.
    """
    sdk = None

    @classmethod
    def init_sdk(cls):
        """
        Class method for initializing the sdk. Called when the first instance is instantiated or when the static method info is called.
        Parameters
        ----------
        cls: required parameter for a class method.

        Raises
        ------
        RuntimeError
           If the library fails to initiate. See tisgrabber.h for error codes.
        """
        sdk = ctypes.cdll.LoadLibrary(DLL_PATH)
        tis.declareFunctions(sdk)

        ERR = sdk.IC_InitLibrary(0)
        if ERR != 1:
            raise Exception("DLL library failed to initiate. Perhaps check the DLL_PATH in tis_camera.py")

        cls.sdk = sdk

        return ERR

    @staticmethod
    def safe_call(cb, to_raise, *args, **kwargs):
        """
        decorator method that automatically error checks the result from cb

        Parameters
        ----------
        cb : function
            Function that is decorated with arguments *args and **kwargs
        to_raise : bool
            Whether to raise an exception or simply print out an error.

        Returns
        -------
        ERR : int
            error code is returned regardless when Exception is raised. Error code information is in tisgrabber.h.
        """
        ERR = cb(*args, **kwargs)
        if ERR <= 0:
            err_str = "Error perforing operation: " + cb.__name__ + " err code: " + str(ERR)
            if to_raise:
                raise Exception(err_str)
            else:
                print(err_str)
        return ERR

    def __init__(
        self,
        serial="",
        vid_format=None,
        verbose=True,
        **kwargs
    ):
        """
        Initialize camera and attributes.

        Parameters
        ----------
        serial : str
            This serial is used to open a camera by unique name (see tisgrabber.h). It is usually the model name followed by a space and the serial number. If "", then opens first camera found.
        vid_format : str
            If None, no format is set and will default to whatever the camera is currently. See tisgrabber.h for more information. Example "Y800 (2592x1944)"
        verbose : bool
            Whether or not to print extra information.
        kwargs
            See :meth:`.Camera.__init__` for permissible options.
        """
        # Initialize the SDK if ndeeded
        if verbose: print("TIS Camera SDK initializing... ", end="")
        if TISCamera.sdk is None:
            ERR = TISCamera.init_sdk()
            if ERR != 1:
                raise Exception("Error when loading SDK: " + str(ERR))
        if verbose: print("success")

        # Then we load the camera from the SDK
        if verbose: print('"{}" initializing... '.format(serial), end="")
        self.cam = TISCamera.sdk.IC_CreateGrabber()              # cam will be the handle that represents the camera.
        if serial == "":
            connected_devs = TISCamera.info()
            if len(connected_devs) == 0:
                raise Exception("No cameras found")
            serial = connected_devs[0] # By default use the first camera that is found
        ERR = TISCamera.sdk.IC_OpenDevByUniqueName(self.cam, tis.T(serial))
        if ERR != 1:
            raise Exception("Error when opening Camera: " + str(ERR))
        if verbose: print("success")

        self.vid_format = vid_format

        # Get in prepared mode and then set the video format
        TISCamera.safe_call(TISCamera.sdk.IC_PrepareLive, 1, self.cam)
        if vid_format is not None:
            TISCamera.safe_call(TISCamera.sdk.IC_SetVideoFormat, 1, self.cam, tis.T(vid_format))

        # Acquire the description of the image.
        width=ctypes.c_long()
        height= ctypes.c_long()
        bitsPerPixel=ctypes.c_int()
        COLORFORMAT=ctypes.c_int()

        TISCamera.safe_call(TISCamera.sdk.IC_GetImageDescription, 1, self.cam, width, height, bitsPerPixel, COLORFORMAT)

        width = width.value
        height = height.value
        bitdepth = int(bitsPerPixel.value / 8 / 3) # Dividing by 3 since it seems like even with format Y800 which is monochrome, it still uses 24 bits per pixel
        
        self.width = width
        self.height = height
        # Finally, use the superclass constructor to initialize other required variables.
        super().__init__(
            width,
            height,
            bitdepth=bitdepth,
            name=serial,
            **kwargs
        )

        # ... Other setup.

    def close(self):
        """See :meth:`.Camera.close`."""
        TISCamera.safe_call(TISCamera.sdk.IC_ReleaseGrabber, self.cam)
        del self.cam

    @staticmethod
    def info(verbose=True):
        """
        Discovers all cameras detected by the SDK.
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
        if TISCamera.sdk is None:
            ERR = TISCamera.init_sdk()
            if ERR != 1:
                raise Exception("Error when loading SDK: " + str(ERR))
        # Get device count and then iterate through each device
        devicecount = TISCamera.sdk.IC_GetDeviceCount()
        serial_list = []
        for i in range(0, devicecount):
            serial_list.append(tis.D(TISCamera.sdk.IC_GetUniqueNamefromList(i)))
        if verbose: print(serial_list)
        return serial_list

    ### Property Configuration ###

    def get_exposure(self):
        """See :meth:`.Camera.get_exposure`."""
        exposure = ctypes.c_float()
        TISCamera.safe_call(TISCamera.sdk.IC_GetPropertyAbsoluteValue, 1, self.cam, tis.T("Exposure"), tis.T("Value"), exposure)
        return float(exposure.value)

    def set_exposure(self, exposure_s):
        """See :meth:`.Camera.set_exposure`."""
        # Turn off auto exposure and use the value given.
        TISCamera.safe_call(TISCamera.sdk.IC_SetPropertySwitch, 1, self.cam, tis.T("Exposure"), tis.T("Auto"), 0)
        TISCamera.safe_call(TISCamera.sdk.IC_SetPropertyAbsoluteValue, 1, self.cam, tis.T("Exposure"), tis.T("Value"), ctypes.c_float(exposure_s))

    def set_woi(self, woi=None):
        """See :meth:`.Camera.set_woi`."""
        if woi is None:
            width=ctypes.c_long()
            height= ctypes.c_long()
            bitsPerPixel=ctypes.c_int()
            COLORFORMAT=ctypes.c_int()

            TISCamera.safe_call(TISCamera.sdk.IC_GetImageDescription, self.cam, width, height, bitsPerPixel, COLORFORMAT)

            width = width.value
            height = height.value
            self.woi = (0, width, 0, height)
        else:
            # Format for woi is specified in :meth:`.Camera.set_woi`.
            width = int(woi[1])
            height = int(woi[3])
            xpos = int(woi[0])
            ypos = int(woi[2])
            # This keeps the original format
            idx = self.vid_format.find("(")
            this_vid_format = self.vid_format[:idx]
            # We bring in the new width and height specified in the video format
            tot_format = this_vid_format + "(" + str(width) +"x" + str(height) + ")"
            TISCamera.safe_call(TISCamera.sdk.IC_SetVideoFormat, 1, self.cam, tis.T(tot_format))
            # Now offset
            TISCamera.safe_call(TISCamera.sdk.IC_SetPropertySwitch, 1, self.cam, tis.T("Partial scan"), tis.T("Auto-center"), 0)
            TISCamera.safe_call(TISCamera.sdk.IC_SetPropertyValue, 1, self.cam, tis.T("Partial scan"), tis.T("X Offset"), xpos)
            TISCamera.safe_call(TISCamera.sdk.IC_SetPropertyValue, 1self.cam, tis.T("Partial scan"), tis.T("Y Offset"), ypos)
        self.width = width
        self.height = height

    def get_image(self, timeout_ms=2000):
        """See :meth:`.Camera.get_image`."""
        buffer_size = 3 * self.bitdepth * self.width * self.height # times 3 is because even Y800 is RGB
        # Starts the image acquisition
        TISCamera.safe_call(TISCamera.sdk.IC_StartLive, 0, self.cam, 0)
        # Snap image 
        ERR = TISCamera.safe_call(TISCamera.sdk.IC_SnapImage, 0, self.cam, timeout_ms)
        # If there is an error, then snap image again
        while ERR <= 0:
            ERR = TISCamera.safe_call(TISCamera.sdk.IC_SnapImage, 0, self.cam, timeout_ms)
        # Get image
        ptr = TISCamera.safe_call(TISCamera.sdk.IC_GetImagePtr, 0, self.cam)
        img_ptr = ctypes.cast(ptr, ctypes.POINTER(ctypes.c_ubyte * buffer_size))
        # Reshape the image according to the width and height
        img = np.ndarray(buffer = img_ptr.contents, dtype = np.uint8, shape = (self.height, self.width, 3)) # 3 for RGB
        TISCamera.safe_call(TISCamera.sdk.IC_StopLive, 0, self.cam)
        # We take only the 1st component, assuming that the image is monochromatic.
        return self.transform(img[:,:,0]) 

    def flush(self):
        """See :meth:`.Camera.flush`."""
        # No flushing is required for TISCamera
