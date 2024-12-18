"""
Light wrapper for `PyLabLib cameras
<https://pylablib.readthedocs.io/en/stable/devices/cameras_root.html>`_.
The :mod:`pylablib` module must be installed ``pip install pylablib``.

Note
~~~~
Color camera functionality is not currently implemented, and will lead to undefined behavior.
"""

from slmsuite.hardware.cameras.camera import Camera
from pylablib.devices.interface.camera import ICamera

class PyLabLib(Camera):
    """
    A wrapped :mod:`instrumental` camera.

    Attributes
    ----------
    cam : instrumental.drivers.cameras.Camera
        Object to talk with the desired camera.
    exposure_s : float
        Instrumental doesn't save exposure. It sets the exposure at each
        :meth:`get_image`. This variable stores the desired exposure.
        Defaults to .001 (1 ms).
    """

    ### Initialization and termination ###

    def __init__(self, cam=None, pitch_um=None, verbose=True, **kwargs):
        """
        Initialize camera and attributes. Initial profile is ``"single"``.

        Parameters
        ----------
        cam : pylablib.devices.interface.camera.Camera
            This class is just a wrapper for :mod:`pylablib`, so the user must pass a
            constructed :mod:`pylablib` camera.
        pitch_um : (float, float) OR None
            Fill in extra information about the pixel pitch in ``(dx_um, dy_um)`` form
            to use additional calibrations.
        verbose : bool
            Whether or not to print extra information.
        kwargs
            See :meth:`.Camera.__init__` for permissible options.

        Raises
        ------
        RuntimeError
           If the camera can not be reached.
        """
        if not isinstance(cam, ICamera):
            raise ValueError(
                "A subclass of pylablib.devices.interface.camera.Camera must be passed as cam."
            )
        # info = cam.get_full_info()
        # if verbose: print(info)

        # Create a name for the camera, defaulting to kwargs.
        name = ""
        di = cam.get_device_info()
        info_counter = 1
        for info in di:
            if isinstance(info, str):
                name += info + "_"
                info_counter += 1

            if info_counter > 3:
                break
        name = name.strip("_")
        if len(name) == 0:
            name = "pylablibcamera"
        name = kwargs.pop("name", name) # info["device_info"].model + "_" + info["device_info"].serial_number)

        if verbose: print(f"Cam {name} parsing... ", end="")
        height, width = cam.get_data_dimensions()
        self.cam = cam

        super().__init__(
            (width, height),
            bitdepth=8,         # Currently defaults to 8 because pylablib doesn't cache this. Update in the future, maybe.
            pitch_um=pitch_um,  # Currently unset because pylablib doesn't cache this. Update in the future, maybe.
            name=name,
            **kwargs
        )
        if verbose: print("success")

    def close(self):
        """
        See :meth:`.Camera.close`.
        """
        try:
            self.cam.close()
        except:
            raise RuntimeError("This instrumental camera does not support .close().")

    @staticmethod
    def info(verbose=True):
        """
        Method to load display information.

        Returns
        -------
        list
            An empty list.
        """
        raise RuntimeError(
            ".info() is not applicable to pylablib cameras, which are "
            "constructed outside this wrapper."
        )

    def reset(self):
        """
        See :meth:`.Camera.reset`.
        """
        raise RuntimeError("Instrumental cameras do not support reset.")

    def get_exposure(self):
        """
        Method to get the integration time in seconds.
        Used in :meth:`.autoexposure()`.

        Returns
        -------
        float
            Integration time in seconds.
        """
        return self.cam.get_exposure()

    def set_exposure(self, exposure_s):
        """
        Method to set the integration time in seconds.
        Used in :meth:`.autoexposure()`.

        Parameters
        ----------
        exposure_s : float
            The integration time in seconds.
        """
        self.cam.set_exposure(float(exposure_s))

    def set_woi(self, woi=None):
        """
        Method to narrow the imaging region to a 'window of interest'
        for faster framerates.

        Parameters
        ----------
        woi : list, None
            See :attr:`~slmsuite.hardware.cameras.camera.Camera.woi`.
            If ``None``, defaults to largest possible.

        Returns
        ----------
        woi : list
            :attr:`~slmsuite.hardware.cameras.camera.Camera.woi`.
        """
        raise NotImplementedError()

    def flush(self, timeout_s=1):
        """
        Method to cycle the image buffer (if any)
        such that all new :meth:`.get_image()`
        calls yield fresh frames.

        This is currently implemented by capturing five camera frames, and should be
        improved in the future.

        Parameters
        ----------
        timeout_s : float
            The time in seconds to wait for frames to catch up with triggers.
        """
        self.get_images(5, timeout_s=timeout_s)

    def _get_image_hw(self, timeout_s=1):
        """
        Method to pull an image from the camera and return.

        Parameters
        ----------
        timeout_s : float
            The time in seconds to wait for the frame to be fetched (currently unused).

        Returns
        -------
        numpy.ndarray
            Array of shape :attr:`~slmsuite.hardware.cameras.camera.Camera.shape`.
        """
        return self.cam.snap(timeout=timeout_s)

    def _get_images_hw(self, image_count, timeout_s=1, out=None):
        """See :meth:`.Camera._get_images_hw`."""
        return self.cam.grab(nframes=image_count, frame_timeout=timeout_s)
