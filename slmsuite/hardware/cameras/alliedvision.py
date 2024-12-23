"""
Hardware control for AlliedVision cameras via the Vimba-X :mod:`vmbpy` interface.
This class also supports backwards compatibility with the
`archived <https://github.com/alliedvision/VimbaPython>`_ :mod:`vimba` interface.
Install :mod:`vmbpy` by following the
`provided instructions <https://github.com/alliedvision/VmbPy>`_.
Be sure to include the ``numpy`` flag in the ``pip install`` command,
as the :class:`AlliedVision` class makes use of these features. See especially the
`vimba python manual <https://github.com/alliedvision/VimbaPython/blob/master/Documentation/Vimba%20Python%20Manual.pdf>`_
for reference.

Note
~~~~
Color camera functionality is not currently implemented, and will lead to undefined behavior.
"""

import time
import numpy as np
import warnings

from slmsuite.hardware.cameras.camera import Camera

try:
    import vmbpy
    vimba_system = vmbpy.VmbSystem
    vimba_name = "vmbpy"

except ImportError:
    try:
        import vimba
        vimba_system = vimba.Vimba
        vimba_name = "vimba"

        warnings.warn("vmbpy not installed; falling back to vimba")
    except ImportError:
        vimba_system = None
        vimba_name = ""
        warnings.warn("vimba or vmbpy are not installed. Install to use AlliedVision cameras.")


class AlliedVision(Camera):
    r"""
    AlliedVision camera.

    Attributes
    ----------
    sdk : vmbpy.Vimba
        AlliedVision SDK. Shared among instances of :class:`AlliedVision`.
    cam : vmbpy.Camera
        Object to talk with the desired camera.

    Caution
    ~~~~~~~~
    The AlliedVision SDK :mod:`vmbpy` includes protections to maintain camera connectivity:
    specifically, the SDK :class:`vmbpy.VmbSystem` and cameras :class:`vmbpy.Camera` are designed
    to be used in concert with ``with`` statements. Unfortunately, this does not mesh with the
    architecture of :mod:`slmsuite`, where notebook-style operation is desired.
    Using ``with`` statements inside :class:`.AlliedVision` methods is likewise not an option,
    as the methods to :meth:`__enter__()` and :meth:`__exit__()` the ``with`` are time-consuming
    due to calls to :meth:`_open()` and :meth:`_close()` the objects, to the point of
    :math:`\mathcal{O}(\text{s})` overhead. :class:`.AlliedVision` disables these protections by
    calling :meth:`__enter__()` and :meth:`__exit__()` directly during :meth:`.__init__()` and
    :meth:`.close()`, instead of inside ``with`` statements.
    """

    sdk = None

    def __init__(self, serial="", pitch_um=None, verbose=True, **kwargs):
        """
        Initialize camera and attributes.

        Parameters
        ----------
        serial : str
            Serial number of the camera to open.
            Use :meth:`.info()` to see detected options.
            If empty, defaults to the first camera in the list
            returned by :meth:`vmbpy.get_all_cameras()`.
        pitch_um : (float, float) OR None
            Fill in extra information about the pixel pitch in ``(dx_um, dy_um)`` form
            to use additional calibrations.
        verbose : bool
            Whether or not to print extra information.
        **kwargs
            See :meth:`.Camera.__init__` for permissible options.
        """
        if vimba_system is None:
            raise ImportError("vimba or vmbpy are not installed. Install to use AlliedVision cameras.")

        if AlliedVision.sdk is None:
            if verbose:
                print(f"{vimba_name} initializing... ", end="")
            AlliedVision.sdk = vimba_system.get_instance()
            AlliedVision.sdk.__enter__()
            if verbose:
                print("success")

        if verbose:
            print("Looking for cameras... ", end="")
        camera_list = AlliedVision.sdk.get_all_cameras()
        if verbose:
            print("success")

        serial_list = [cam.get_serial() for cam in camera_list]
        if serial == "":
            if len(camera_list) == 0:
                raise RuntimeError(f"No cameras found by {vimba_name}.")
            if len(camera_list) > 1 and verbose:
                print(f"No serial given... Choosing first of {serial_list}")

            self.cam = camera_list[0]
            serial = self.cam.get_serial()
        else:
            if serial in serial_list:
                self.cam = camera_list[serial_list.index(serial)]
            else:
                raise RuntimeError(
                    f"Serial {serial} not found by {vimba_name}. Available: {serial_list}"
                )

        if verbose:
            print(f"{vimba_name} sn '{serial}' initializing... ", end="")
        self.cam.__enter__()
        if verbose:
            print("success")

        try:
            self.cam.BinningHorizontal.set(1)
            self.cam.BinningVertical.set(1)
        except:
            pass  # Some cameras do not have the option to set binning.

        self.cam.GainAuto.set("Off")

        self.cam.ExposureAuto.set("Off")
        self.cam.ExposureMode.set("Timed")

        self.cam.AcquisitionMode.set("SingleFrame")

        # Future: triggered instead of SingleFrame.
        self.cam.TriggerSelector.set("AcquisitionStart")
        self.cam.TriggerMode.set("Off")
        self.cam.TriggerActivation.set("RisingEdge")
        self.cam.TriggerSource.set("Software")

        super().__init__(
            (self.cam.SensorWidth.get(), self.cam.SensorHeight.get()),
            bitdepth=int(self.cam.PixelSize.get()),
            pitch_um=pitch_um,
            name=serial,
            **kwargs,
        )

    def close(self, close_sdk=True):
        """
        See :meth:`.Camera.close`

        Parameters
        ----------
        close_sdk : bool
            Whether or not to close the :mod:`vmbpy` instance.
        """
        self.cam.__exit__(None, None, None)

        if close_sdk:
            self.close_sdk()

        del self.cam

    @staticmethod
    def info(verbose=True):
        """
        Discovers all Thorlabs scientific cameras.

        Parameters
        ----------
        verbose : bool
            Whether to print the discovered information.

        Returns
        --------
        list of str
            List of AlliedVision serial numbers.
        """
        if vimba_system is None:
            raise ImportError("vimba or vmbpy are not installed. Install to use AlliedVision cameras.")

        if AlliedVision.sdk is None:
            AlliedVision.sdk = vimba_system.get_instance()
            AlliedVision.sdk.__enter__()
            close_sdk = True
        else:
            close_sdk = False

        camera_list = AlliedVision.sdk.get_all_cameras()
        serial_list = [cam.get_serial() for cam in camera_list]

        if verbose:
            print(f"{vimba_name} serials:")
            for serial in serial_list:
                print(f"'{serial}'")

        if close_sdk:
            AlliedVision.close_sdk()

        return serial_list

    @classmethod
    def close_sdk(cls):
        """
        Close the :mod:`vmbpy` instance.
        """
        if cls.sdk is not None:
            cls.sdk.__exit__(None, None, None)
            cls.sdk = None

    ### Property Configuration ###

    def get_properties(self, properties=None):
        """
        Print the list of camera properties.

        Parameters
        ----------
        properties : dict or None
            The target camera's property dictionary. If ``None``, the property
            dictionary is fetched from the camera associated with the calling instance.
        """
        if properties is None:
            properties = self.cam.__dict__.keys()

        for key in properties:
            prop = self.cam.__dict__[key]
            try:
                print(prop.get_name(), end="\t")
            except BaseException as e:
                print(f"Error accessing property dictionary, '{key}':{e}")
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

        for entry in self.cam.SensorBitDepth.get_all_entries():
            value = entry.as_tuple()  # (name : str, value : int)
            if str(bitdepth) in value[0]:
                self.cam.SensorBitDepth.set(value[1])
                break
            raise RuntimeError(f"ADC bitdepth {bitdepth} not found.")

    def get_adc_bitdepth(self):
        """
        Get the digitization bitdepth.

        Returns
        -------
        int
            The digitization bitdepth.
        """
        value = str(self.cam.SensorBitDepth.get())
        bitdepth = int("".join(char for char in value if char.isdigit()))
        return bitdepth

    def _get_exposure_hw(self):
        """See :meth:`.Camera._get_exposure_hw`."""
        return float(self.cam.ExposureTime.get()) / 1e6

    def _set_exposure_hw(self, exposure_s):
        """See :meth:`.Camera._set_exposure_hw`."""
        self.cam.ExposureTime.set(float(exposure_s * 1e6))

    def set_woi(self, woi=None):
        """See :meth:`.Camera.set_woi`."""
        return

    def _get_image_hw(self, timeout_s):
        """See :meth:`.Camera._get_image_hw`."""
        t = time.time()

        # Convert timeout_s to ms
        frame = self.cam.get_frame(timeout_ms=int(1e3 * timeout_s))
        frame = frame.as_numpy_ndarray()

        # We have noticed that sometimes the camera gets into a state where
        # it returns a frame of all zeros apart from one pixel with value of 31.
        # This method is admittedly a hack to try getting a frame a few more times.
        # We welcome contributions to fix this.
        while np.sum(frame) == np.amax(frame) == 31 and time.time() - t < timeout_s:
            frame = self.cam.get_frame(timeout_ms=int(1e3 * timeout_s))
            frame = frame.as_numpy_ndarray()

        return np.squeeze(frame)
