"""
**(NotImplemented)** Hardware control for FLIR cameras via the :mod:`PySpin` interface to the Spinnaker SDK.
Install Spinnaker by following the
`provided instructions <https://www.flir.com/products/spinnaker-sdk/>`_.
Inspired by `this implementation <https://github.com/klecknerlab/simple_pyspin/>`_.

Warning
~~~~~~~~
Implementation unfinished and untested. Consider using ``simple_pyspin`` as a dependency instead.
"""
import warnings

from .camera import Camera

try:
    import PySpin
except ImportError:
    PySpin = None
    warnings.warn("PySpin not installed. Install to use FLIR cameras.")


class FLIR(Camera):
    """
    FLIR camera.

    Attributes
    ----------
    sdk : PySpin.System
        Spinnaker SDK.
    cam : PySpin.Camera
        Object to talk with the desired camera.
    """

    sdk = None

    ### Initialization and termination ###

    def __init__(self, serial="", pitch_um=None, verbose=True, **kwargs):
        """
        Initialize camera and attributes.

        Parameters
        ----------
        serial : str
            Serial number of the camera to open.
        pitch_um : (float, float) OR None
            Fill in extra information about the pixel pitch in ``(dx_um, dy_um)`` form
            to use additional calibrations.
        verbose : bool
            Whether or not to log.
        **kwargs
            See :meth:`.Camera.__init__` for permissible options.
        """
        if PySpin is None:
            raise ImportError("PySpin not installed. Install to use FLIR cameras.")

        if FLIR.sdk is None:
            if verbose:
                print("PySpin initializing... ", end="")
            FLIR.sdk = PySpin.System.get_instance()
            if verbose:
                print("success")

        if verbose:
            print("Looking for cameras... ", end="")
        camera_list = PySpin.sdk.GetCameras()
        if verbose:
            print("success")

        if verbose:
            print("Camera initializing... ", end="")
        if serial == "":
            self.cam = camera_list.GetByIndex(0)
        else:
            self.cam = camera_list.GetBySerial(serial)
        self.cam.Init()

        super().__init__(
            (self.cam.SensorWidth.get(), self.cam.SensorHeight.get()),
            bitdepth=int(self.cam.PixelSize.get()),
            pitch_um=pitch_um,
            name=serial,
            **kwargs
        )
        if verbose:
            print("success")

        raise NotImplementedError()

    def close(self, close_sdk=True):
        """See :meth:`.Camera.close`."""
        try:
            self.cam.EndAcquisition()
        except BaseException:
            pass
        del self.cam

    ### Property Configuration ###

    def _get_exposure_hw(self):
        """See :meth:`.Camera._get_exposure_hw`."""
        return

    def _set_exposure_hw(self, exposure_s):
        """See :meth:`.Camera._set_exposure_hw`."""
        return

    def set_woi(self, window=None):
        """See :meth:`.Camera.set_woi`."""
        return

    def _get_image_hw(self, blocking=True):
        """
        See :meth:`.Camera._get_image_hw`.

        Parameters
        ----------
        blocking : bool
            Whether to wait for the camera to return a frame, blocking other acquisition.
        """
        frame = self.cam.GetNextImage(
            PySpin.EVENT_TIMEOUT_INFINITE if blocking else PySpin.EVENT_TIMEOUT_NONE
        )

        return frame.GetNDArray()
