"""
Template for writing a subclass for camera hardware control in :mod:`slmsuite`.
Outlines which camera superclass functions must be implemented.
"""

from slmsuite.hardware.cameras.camera import Camera

class Template(Camera):
    """
    Template for adding a new camera to :mod:`slmsuite`. Replace :class:`Template`
    with the desired subclass name. :class:`~slmsuite.hardware.cameras.camera.Camera` is the
    superclass that sets the requirements for :class:`Template`.

    Attributes
    ----------
    sdk : object
        Many cameras have a singleton SDK class which handles all the connected cameras
        of a certain brand. This is generally implemented as a class variable.
    cam : object
        Most cameras will wrap some handle which connects to the the hardware.
    """

    # Class variable (same for all instances of Template) pointing to a singleton SDK.
    sdk = None

    def __init__(
        self,
        serial="",
        verbose=True,
        **kwargs
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
        # TODO: Insert code here to initialize the camera hardware, load properties, etc.

        # Mandatory functions:
        # - Opening a connection to the device

        # Other possibilities to consider:
        # - Loading a connection to the SDK, if applicable.
        # - Gathering parameters such a width, height, and bitdepth.

        # Most cameras have an SDK that needs to be loaded before the camera
        if verbose: print("Template SDK initializing... ", end="")
        raise NotImplementedError()
        Template.sdk = something()                      # TODO: Fill in proper function.
        if verbose: print("success")

        # Then we load the camera from the SDK
        if verbose: print('"{}" initializing... '.format(serial), end="")
        raise NotImplementedError()
        self.cam = sdk.something(serial)                # TODO: Fill in proper function.
        if verbose: print("success")

        # Finally, use the superclass constructor to initialize other required variables.
        super().__init__(
            self.cam.get_width(),                       # TODO: Fill in proper functions.
            self.cam.get_height(),
            bitdepth=self.cam.get_depth(),
            name=serial,
            **kwargs
        )

        # ... Other setup.

    def close(self):
        """See :meth:`.Camera.close`."""
        raise NotImplementedError()
        self.cam.close()                                # TODO: Fill in proper function.
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
        raise NotImplementedError()
        serial_list = Template.sdk.get_serial_list()    # TODO: Fill in proper function.
        return serial_list

    ### Property Configuration ###

    def get_exposure(self):
        """See :meth:`.Camera.get_exposure`."""
        raise NotImplementedError()
        return float(self.cam.get_exposure()) / 1e3     # TODO: Fill in proper function.

    def set_exposure(self, exposure_s):
        """See :meth:`.Camera.set_exposure`."""
        raise NotImplementedError()
        self.cam.get_exposure(1e3 * exposure_s)         # TODO: Fill in proper function.

    def set_woi(self, woi=None):
        """See :meth:`.Camera.set_woi`."""
        raise NotImplementedError()
        # Use self.cam to crop the window of interest.

    def get_image(self, timeout_s=1):
        """See :meth:`.Camera.get_image`."""
        raise NotImplementedError()
        # The core method: grabs an image from the camera.
        # self.transform implements the flipping and rotating keywords passed to the
        # superclass constructor. This is handled automatically.
        return self.transform(self.cam.get_image())     # TODO: Fill in proper function.
