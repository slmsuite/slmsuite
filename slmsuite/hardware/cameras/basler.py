"""
**(Untested)** Hardware control for Basler cameras via the :mod:`pypylon` interface.
Consider also installing Basler software for testing cameras outside of python
(see `downloads <https://www.baslerweb.com/en/downloads/software-downloads/#type=pylonsoftware;language=all;version=7.3.0>`_).
Install :mod:`pypylon` by following the `provided instructions <https://github.com/basler/pypylon>`_.
"""
import warnings
from slmsuite.hardware.cameras.camera import Camera

try:
    from pypylon import pylon
except ImportError:
    pylon = None
    warnings.warn("pypylon not installed. Install to use Basler cameras.")


class Basler(Camera):
    """
    Interface to Basler cameras.

    Attributes
    ----------
    sdk : pylon.TlFactory
        "Transport layer" factory used by Basler to find camera devices.
    cam : pylon.InstantCamera
        Object used to communicate with the camera.
    """

    # Class variable (same for all instances of Template) pointing to a singleton SDK.
    sdk = None

    def __init__(self, serial="", pitch_um=None, verbose=True, **kwargs):
        """
        Initialize Basler camera and attributes.

        Parameters
        ----------
        serial : str
            Serial number of the camera to open.
            If empty, defaults to the first camera in the list
            returned by :meth:`.info()`.
        pitch_um : (float, float) OR None
            Fill in extra information about the pixel pitch in ``(dx_um, dy_um)`` form
            to use additional calibrations.
        verbose : bool
            Whether or not to print extra information.
        kwargs
            See :meth:`.Camera.__init__` for permissible options.
        """
        if pylon is None:
            raise ImportError("pypylon not installed. Install to use Basler cameras.")

        if Basler.sdk is None:
            if verbose:
                print("pylon initializing... ", end="")
            Basler.sdk = pylon.TlFactory.GetInstance()
            if verbose:
                print("success")

        if verbose:
            print("Looking for cameras... ", end="")
        device_list = Basler.sdk.EnumerateDevices()
        if verbose:
            print("success")

        serial_list = [dev.GetSerialNumber() for dev in device_list]
        if serial == "":
            if len(device_list)==0:
                raise RuntimeError("No cameras found by pylon.")
            if len(device_list) > 1 and verbose:
                print("No serial given... Choosing first of ", serial_list)
            serial = serial_list[0]
            device = device_list[0]
        else:
            if serial in serial_list:
                device = device_list[serial_list.index(serial)]
            else:
                raise RuntimeError(
                    "Serial " + serial + " not found by pylon. Available: ", serial_list
                )

        if verbose:
            print("pylon sn " "{}" " initializing... ".format(serial), end="")
        self.cam = pylon.InstantCamera()
        # TODO: ichr modified this to be compatible with arbitrary serials (not just
        # first), also camera is opened at the start.
        self.cam.Attach(device)
        self.cam.Open()

        # Apply default settings.
        self.cam.BinningHorizontal.SetValue(1)
        self.cam.BinningVertical.SetValue(1)

        self.cam.GainAuto.SetValue('Off')
        self.cam.ExposureAuto.SetValue('Off')
        self.cam.ExposureMode.SetValue('Timed')

        self.cam.AcquisitionMode.SetValue('SingleFrame')

        self.cam.TriggerSelector.SetValue('FrameStart')
        self.cam.TriggerMode.SetValue('Off')

        self.cam.TriggerActivation.SetValue('RisingEdge')
        self.cam.TriggerSource.SetValue('Software')

        # TODO: ichr moved this from .get_image() to here. Does this work?
        self.cam.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

        # Initialize the superclass attributes.
        super().__init__(
            (self.cam.SensorWidth(), self.cam.SensorHeight()), #pixels
            bitdepth=self.cam.PixelSize.GetIntValue(), #bits
            pitch_um=pitch_um,
            name=serial,
            **kwargs
        )

        if verbose:
            print("success")

    def close(self, close_sdk=True):
        """
        See :meth:`.Camera.close`.

        Parameters
        ----------
        close_sdk : bool
            Does nothing, as the ``pylon.TlFactory`` instance stored in :attr:`sdk`
            does not appear to need to be closed.
        """
        #self.cam.__exit__(None, None, None) weird
        self.cam.StopGrabbing()
        self.cam.Close()

        if close_sdk:
            pass

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
        if pylon is None:
            raise ImportError("pypylon not installed. Install to use Basler cameras.")

        if Basler.sdk is None:
            Basler.sdk = pylon.TlFactory.GetInstance()
            #Basler.sdk.__enter__()
            close_sdk = True
        else:
            close_sdk = False

        camera_list = Basler.sdk.EnumerateDevices()

        serial_list = [cam.GetSerialNumber() for cam in camera_list]

        if verbose:
            print('Basler cameras:')
            for serial in serial_list:
                print("\"{}\"".format(serial))

        if close_sdk:
            Basler.close_sdk()

        return serial_list

    @classmethod
    def close_sdk(cls):
        """"
        Close the :mod:'pylon' instance.
        """
        if cls.sdk is not None:
            #cls.sdk.__exit__(None, None, None)
            cls.sdk = None

    ### Property Configuration ###

    def get_properties(self, properties=None):
        """"
        Print the list of camera properties.

        Parameters
        ----------
        properties: dict or None
            The target camera's property dictionary. If ''None'', the property
            dictionary is fetched from the camera associated with the cancelling instance.
        """
        if properties is None:
            properties = self.cam.__dict__.keys()

        for key in properties:
            prop=self.cam.__dict__[key]
            try:
                print(prop.get_name(), end="\t")
            except BaseException as e:
                print("Error accessing property dictionary, '{}':{}".format(key, e))
                continue

            try:
                print(prop.get(), end="\t")
            except:
                pass

            try:
                print(prop.get_unit(), end="\t")
            except:
                pass

            try:
                print(prop.get_description(), end="\n")
            except:
                print("")

    def set_adc_bitdepth(self, bitdepth):
        """
        Set the digitization bitdepth.

        Parameters
        ----------
        bitdepth : int
            Desired digitization bitdepth.
        """
        bitdepth = int(bitdepth)

        for entry in self.cam.PixelSize.GetEntries():
            value = entry.as_tuple()
            if str(bitdepth) in value[0]:
                self.cam.PixelSize.SetValue(value[1])
                break
            raise RuntimeError("ADC bitdepth {} not found.".format(bitdepth))

    def get_adc_bitdepth(self):
        """
        Get the digitization bitdepth.

        Returns
        -------
        int
            The digitization bitdepth.
        """
        value = str(self.cam.PixelSize.GetValue())
        bitdepth = int("".join(char for char in value if char.isdigit()))
        return bitdepth

    def _get_exposure_hw(self):
        """See :meth:`.Camera._get_exposure_hw`."""
        return float(self.cam.ExposureTime.GetValue()) / 1e6   # in seconds

    def _set_exposure_hw(self, exposure_s):
        """See :meth:`.Camera._set_exposure_hw`."""
        self.cam.ExposureTime.SetValue(float(1e6 * exposure_s))   # in seconds

    def set_woi(self, woi=None):
        """See :meth:`.Camera.set_woi`."""
        return
        # Use self.cam to crop the window of interest.

    def _get_image_hw(self, timeout_s):
        """See :meth:`.Camera.get_image`."""
        # TODO: ichr added WaitForFrameTriggerReady and ExecuteSoftwareTrigger
        # timeout_s is now used. _get_image_hw is the new subclass hardware method.
        self.cam.WaitForFrameTriggerReady(200, pylon.TimeoutHandling_ThrowException)
        self.cam.ExecuteSoftwareTrigger()
        grab = self.cam.RetrieveResult(int(timeout_s*1000), pylon.TimeoutHandling_Return)

        # Image grabbed successfully?
        if not grab.GrabSucceeded():
            raise RuntimeError("Basler error: ", grab.ErrorCode, grab.ErrorDescription)

        im = grab.GetArray()    # This returns an np.array
        # TODO: ichr added Release(); does this work?
        grab.Release()
        return im
