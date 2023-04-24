"""
Template for new cameras added to :mod:`slmsuite`.
"""

from slmsuite.hardware.cameras.camera import Camera

class Template(Camera):
    """
    Template for adding a new camera to :mod:`slmsuite`.

    Attributes
    ----------
    sdk : object
        Many cameras have a singleton SDK class which handles all the connected cameras
        of a certain brand. This is generally implemented as a class variable.
    cam : object
        Most cameras will wrap some handle which connects to the the hardware.
    """

    sdk = None

    def __init__(
        self, serial="", verbose=True, **kwargs
    ):
        """
        Initialize camera and attributes.

        Parameters
        ----------
        serial : str
            Most SDKs identify different cameras by some serial number or string.
        verbose : bool
            Whether or not to print extra information.
        kwargs
            See :meth:`.Camera.__init__` for permissible options.
        """
        if verbose:
            print("Template SDK initializing... ", end="")

        # Most cameras have an SDK that needs to be loaded before the camera
        raise NotImplementedError()
        # Template.sdk = something()...

        if verbose:
            print("success")

        if verbose:
            print('"{}" initializing... '.format(serial), end="")

        # Then we load the camera from the SDK
        raise NotImplementedError()
        # self.cam = sdk.something(serial)...

        if verbose:
            print("success")

        super().__init__(
            self.cam.get_width(),           # These function names will depend on the camera.
            self.cam.get_height(),
            bitdepth=self.cam.get_depth(),
            name=serial,
            **kwargs
        )

        # ... Other setup.

    def close(self):
        """See :meth:`.Camera.close`."""
        raise NotImplementedError()
        self.cam.close()                    # This function name will depend on the camera.
        del self.cam

    @staticmethod
    def info(verbose=True):
        """
        Discovers all cameras detected by the SDK.

        Parameters
        ----------
        verbose : bool
            Whether to print the discovered information.

        Returns
        --------
        list of str
            List of serial numbers or identifiers.
        """
        raise NotImplementedError()
        serial_list = sdk.get_serial_list()             # This function name will depend on the camera.
        return serial_list


    ### Property Configuration ###

    def get_exposure(self):
        """See :meth:`.Camera.get_exposure`."""
        raise NotImplementedError()
        return float(self.cam.get_exposure()) / 1e3     # This function name will depend on the camera.

    def set_exposure(self, exposure_s):
        """See :meth:`.Camera.set_exposure`."""
        raise NotImplementedError()
        self.cam.get_exposure(1e3 * exposure_s)         # This function name will depend on the camera.

    def set_woi(self, woi=None):
        """See :meth:`.Camera.set_woi`."""
        raise NotImplementedError()
        # Use self.cam to crop the window of interest.

    def get_image(self, timeout_s=1):
        """See :meth:`.Camera.get_image`."""
        raise NotImplementedError()
        # The core method: grabs an image from the camera.
        return self.transform(self.cam.get_image())     # This function name will depend on the camera.
