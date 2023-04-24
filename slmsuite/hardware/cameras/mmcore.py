"""
**(NotImplemented)** Hardware control for Micro-Manager cameras via the :mod:`pymmcore` interface.
See :mod:`pymmcore` `documentation <https://github.com/micro-manager/pymmcore>`_.

Warning
~~~~~~~~
Implementation unfinished and untested.
"""

import os

from slmsuite.hardware.cameras.camera import Camera

try:
    import pymmcore
except ImportError:
    print("mmcore.py: pymmcore not installed. Install to use Micro-Manager cameras.")


class MMCore(Camera):
    """
    Micro-Manager camera.

    Attributes
    ----------
    cam : pymmcore.CMMCore
        Object to talk with the desired camera.
    """

    def __init__(
        self, config, path="C:/Program Files/Micro-Manager-2.0", verbose=True, **kwargs
    ):
        """
        Initialize camera and attributes.

        Parameters
        ----------
        config : str
            Name of the config file corresponding to the desired camera. This is assumed to be
            a ``.cfg`` file stored in the Micro-Manager ``path`` (see below). ``.cfg`` may be included
            or omitted, but the :attr:`name` of the camera will be without it.
        path : str
            Directory of the Micro-Manager installation. Defaults to the directory of a default
            Micro-Manager 2.0 installation.
        verbose : bool
            Whether or not to print extra information.
        kwargs
            See :meth:`.Camera.__init__` for permissible options.
        """

        if config[-4:] == ".cfg":
            config = config[:-4]

        if verbose:
            print("CMMCore initializing... ", end="")
        self.cam = pymmcore.CMMCore()
        self.cam.setDeviceAdapterSearchPaths(path)
        if verbose:
            print("success")

        if verbose:
            print('"{}" initializing... '.format(config), end="")
        self.cam.loadSystemConfiguration(os.path.join(path, config, ".cfg"))
        if verbose:
            print("success")

        super().__init__(
            self.cam.getImageWidth(),
            self.cam.getImageHeight(),
            bitdepth=self.cam.getImageBitDepth(),
            name=config,
            **kwargs
        )

        # ... Other setup.

    def close(self):
        """See :meth:`.Camera.close`."""
        del self.cam

    ### Property Configuration ###

    def get_exposure(self):
        """See :meth:`.Camera.get_exposure`."""
        return float(self.cam.getExposure()) / 1e3

    def set_exposure(self, exposure_s):
        """See :meth:`.Camera.set_exposure`."""
        self.cam.setExposure(1e3 * exposure_s)

    def set_woi(self, woi=None):
        """See :meth:`.Camera.set_woi`."""
        return

    def get_image(self, timeout_s=1):
        """See :meth:`.Camera.get_image`."""
        return self.transform(self.cam.getImage())
