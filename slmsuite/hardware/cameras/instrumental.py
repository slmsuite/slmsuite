"""
Light wrapper for `Instrumental cameras
<https://instrumental-lib.readthedocs.io/en/stable/install.html>`_.
The :mod:`instrumental` module must be installed ``pip install instrumental-lib``.

Note
~~~~
Color camera functionality is not currently implemented, and will lead to undefined behavior.
"""
from slmsuite.hardware.cameras.camera import Camera

try:
    import instrumental.drivers.cameras.Camera as InstrumentalCamera
except ImportError:
    print("instrumental-lib not installed. Install to use Instrumental cameras.")


class Instrumental(Camera):
    """
    Wrapped :mod:`instrumental` camera.

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

    def __init__(self, cam, **kwargs):
        """
        Initialize camera and attributes. Initial profile is ``"single"``.

        Parameters
        ----------
        cam : instrumental.drivers.cameras.Camera
            This class is just a wrapper for :mod:`instrumental`, so the user must pass a
            constructed :mod:`instrumental` camera.
        kwargs
            See :meth:`.Camera.__init__` for permissible options.

        Raises
        ------
        RuntimeError
           If the camera can not be reached.
        """
        assert isinstance(cam, InstrumentalCamera), "A subclass of instrumental.drivers.cameras.Camera must be passed as cam."
        self.cam = cam
        self.exposure_s = .001

        super().__init__(
            (self.cam.width, self.cam.height),
            bitdepth=8,         # Currently defaults to 8 because instrumental doesn't cache this. Update in the future, maybe.
            pitch_um=None,      # Currently unset because instrumental doesn't cache this. Update in the future, maybe.
            name=cam.__class__.__name__,
            **kwargs
        )

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
            ".info() is not applicable to instrumental cameras, which are "
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
        return self.exposure_s

    def set_exposure(self, exposure_s):
        """
        Method to set the integration time in seconds.
        Used in :meth:`.autoexposure()`.

        Parameters
        ----------
        exposure_s : float
            The integration time in seconds.
        """
        self.exposure_s = exposure_s

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
        self.get_images(5)

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
        return self.cam.grab_image(timeouts=str(self.exposure_s) + "s", copy=True)