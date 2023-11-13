"""
Simulated camera to image the simulated SLM.
"""

import numpy as np
import warnings

try:
    import cupy as cp
    from cupyx.scipy.ndimage import map_coordinates
except:
    cp = np
    from scipy.ndimage import map_coordinates

import matplotlib.pyplot as plt

from slmsuite.hardware.cameras.camera import Camera
from slmsuite.holography.algorithms import Hologram
from slmsuite.holography import toolbox
from slmsuite.misc.math import REAL_TYPES


class SimulatedCam(Camera):
    """
    Simulated camera.

    Outputs simulated images (i.e., the far-field of an SLM interpolated to
    camera pixels based on the camera's location and orientation.
    Serves as a future testbed for simulation of other imaging artifacts, including non-affine
    aberrations (e.g. pincushion distortion) and imaging readout noise.

    Note
    ~~~~
    For fastest simulation, initialize :class:`SimulatedCam` with a
    :class:`~slmsuite.hardware.slms.simulated.SimulatedSLM` *only*. Simulated camera images
    will directly sample the (quickly) computed SLM far-field (`"knm"`) via a one-to-one
    mapping instead of interpolating the SLM's far-field intensity at
    each camera pixel location (i.e. `"knm"`->"ij" basis change),
    which may also require additional padding (computed automatically upon initialization) for
    sufficient resolution.

    Attributes
    ----------
    resolution : (int, int)
        (width, height) of the SLM in pixels.
    exposure : float
        Digital gain value to simulate exposure time. Directly proportional to imaged power.
    x_grid : ndarray
        Pixel column number (``"ij"`` basis) used for far-field interpolation.
    y_grid : ndarray
        Pixel row number (``"ij"`` basis) used for far-field interpolation.
    shape_padded : (int, int)
        Size of the FFT computational space required to faithfully reproduce the far-field at
        full camera resolution.
    noise : dict
        Dictionary of single-argument functions (returning the normalized noise amplitude 
        for any normalized input pixel amplitude) to simulate various noise sources. Currently,
        ``'dark'`` and ``'read'``, representing exposure-dependent dark current/background noise
        and exposure-independent readout noise, respectively, are the only accepted keys.

        Example
        ~~~~~~~
        The following code adds a Gaussian background with 50% mean and 5% standard
        deviation (relative to the dynamic range at the default ``self.exposure = 1``) and
        a Poisson readout noise (independent of ``self.exposure``) with an average value
        of 20% of the camera's dynamic range.

        .. code-block:: python
            
            self.noise = {'dark': lambda img: np.random.normal(0.5*img, 0.05*img),
                          'read': lambda img: np.random.poisson(0.2*img)}

    """

    def __init__(self, slm, resolution=None, M=None, b=None, noise=None, **kwargs):
        """
        Initialize simulated camera.

        Parameters
        ----------
        slm : :class:`~slmsuite.hardware.slms.simulated.SimulatedSLM`
            Simulated SLM creating the image.
        resolution : tuple
            See :attr:`resolution`. If ``None``, defaults to the resolution of `slm`.
        M : ndarray
            2 x 2 affine transform matrix to convert between `slm`'s k-space and the
            simulated camera's pixel (``"ij"``) basis. If ``None``, defaults to the
            identity matrix.
        b : tuple
            Lateral displacement (in pixels) of the camera center from `slm`'s
            optical axis. If ``None``, defaults to 0 offset.
        noise : dict
            See :attr:`noise`.
        kwargs
            See :meth:`.Camera.__init__` for permissible options.
        """

        # Store a reference to the SLM: we need this to compute the far-field camera images.
        self._slm = slm

        # Don't interpolate (slower) by default unless required.
        self._interpolate = False

        if resolution is None:
            resolution = slm.shape[::-1]
        elif any([r != s for r, s in zip(resolution, slm.shape[::-1])]):
            self._interpolate = True

        super().__init__(int(resolution[0]), int(resolution[1]), **kwargs)

        # Digital gain emulates exposure
        self.exposure = 1

        # Add user-defined noise dictionary
        self.noise = noise

        # Compute the camera pixel grid in `basis` units (currently "ij")
        self.x_grid, self.y_grid = np.meshgrid(
            np.arange(resolution[0]),
            np.arange(resolution[1]),
        )

        # Affine transform the camera grid ("ij"->"kxy")
        if M is not None or b is not None:
            self._interpolate = True
            (self.x_grid, self.y_grid) = toolbox.transform_grid(self, M, b, direction="rev")

        # Fourier space must be sufficiently padded to resolve the camera pixels.
        dkxy = np.sqrt(
            (self.x_grid[:2, :2] - self.x_grid[0, 0]) ** 2
            + (self.y_grid[:2, :2] - self.y_grid[0, 0]) ** 2
        )
        dkxy_min = dkxy.ravel()[1:].min()

        self.shape_padded = Hologram.calculate_padded_shape(slm, precision=dkxy_min)
        print(
            "Padded SLM k-space shape set to (%d,%d) to achieve required "
            "imaging resolution." % (self.shape_padded[1], self.shape_padded[0])
        )

        phase = -self._slm.display.astype(float)/self._slm.bitresolution*(2*np.pi)
        self._hologram = Hologram(
            self.shape_padded,
            amp = self._slm.source["amplitude_sim"],
            # phase = self._slm.phase + self._slm.source["phase_sim"],
            phase = phase - phase.min() + self._slm.source["phase_sim"],
            slm_shape = self._slm,
        )

        # Convert kxy -> knm (0,0 at corner): 1/dx -> Npx
        self.knm_cam = cp.array(
            [
                self.shape_padded[0] * self._slm.dy * self.y_grid + self.shape_padded[0] / 2,
                self.shape_padded[1] * self._slm.dx * self.x_grid + self.shape_padded[1] / 2,
            ]
        )

        if (
            cp.amax(cp.abs(self.knm_cam[0] - self.shape_padded[0]/2)) > self.shape_padded[1]/2
            or cp.amax(cp.abs(self.knm_cam[1] - self.shape_padded[1]/2)) > self.shape_padded[0]/2
        ):
            warnings.warn(
                "Camera extends beyond the accessible SLM k-space;"
                " some pixels may not be targetable."
            )

    @staticmethod
    def build_affine(f_eff=1, theta=0, shear_angle=0, offset=(0, 0), basis="ij", pitch_um=None):
        """
        Build an affine transform defining the SLM -> camera transformation as
        detailed in :meth:`~slmsuite.hardware.cameraslms.FourierSLM.kxyslm_to_ijcam`.

        Parameters
        ----------
        f_eff : float or (float, float)
            Effective focal length (in `basis` units) of the
            optical train separating the Fourier-domain SLM from the camera. If a float is provided,
            `f_eff` is isotropic; otherwise, `f_eff` is defined along the SLM's x and y axes.

            Important
            ~~~~~~~~~
            The default unit for `f_eff` is pixels/radian, i.e. the units of :math:`M` matrix
            elements required to convert normalized angles/:math:`k`-space coordinates to camera
            pixels in the ``"ij"`` basis.
            See :meth:`~slmsuite.hardware.cameraslms.FourierSLM.kxyslm_to_ijcam` for additional
            details. To convert to true distance units (e.g., `"um"`), multiply `f_eff` by the the
            pixel size in the same dimensions.
            As noted in :meth:`~slmsuite.hardware.cameraslms.get_effective_focal_length`, non-square
            pixels therefore imply different effective focal lengths along each axis when using
            true distance units.
        theta : float
            Rotation angle (in radians, ccw) of the camera relative to the SLM orientation.
            Defaults to 0 (i.e., aligned with the SLM).
        shear_angle : float or (float, float)
            Shearing angles (in radians) along the SLM's x and y axes. If a float is provided,
            shear is applied isotropically.
        offset : tuple
            Lateral displacement (in pixels units) of the camera center from `slm`'s
            optical axis. If ``None``, defaults to 0 offset.
        basis : str
            Sets the units for `f_eff` and `offset`. Defaults to ``"ij"``, the camera pixel basis.
        pitch_um : float or (float, float)
            Camera pixel pitch in microns. Must be provided if ``basis != "ij"``. A square pixel
            is assumed if a single float is provided.

        Returns
        -------
        ndarray
            2 x 2 affine matrix :math:`M`
        tuple
            Affine vector :math:`b`
        """

        if isinstance(f_eff, REAL_TYPES):
            f_eff = [f_eff, f_eff]
        if isinstance(pitch_um, REAL_TYPES):
            pitch_um = [pitch_um, pitch_um]
        if isinstance(shear_angle, REAL_TYPES):
            shear_angle = [shear_angle, shear_angle]

        if basis != "ij":
            assert pitch_um is not None, "Must provide pixel pitch when using real units!"

            if basis in toolbox.LENGTH_FACTORS.keys():
                factor = toolbox.LENGTH_FACTORS[basis]
            else:
                raise RuntimeError("Did not recognize units '{}'".format(basis))

            f_eff = [(factor * f) / p for f, p in zip(f_eff, pitch_um)]
            b = [(factor * o) / p for o, p in zip(offset, pitch_um)]

        else:
            b = offset

        mag = np.array([[f_eff[0], 0], [0, f_eff[1]]])
        shear = np.array([[1, np.tan(shear_angle[0])], [np.tan(shear_angle[1]), 1]])
        rot = np.array([[np.cos(-theta), np.sin(-theta)], [-np.sin(-theta), np.cos(-theta)]])

        M = mag @ shear @ rot

        return M, b

    def flush(self):
        """
        See :meth:`.Camera.flush`.
        """
        return

    def set_exposure(self, exposure):
        """
        Set the simulated exposure (i.e. digital gain).

        Parameters
        ----------
        exposure : float
            Digital gain.
        """
        self.exposure = exposure

    def get_exposure(self):
        """
        Get the simulated exposure (i.e. digital gain).
        """
        return self.exposure

    def _get_image_hw(self, timeout_s=None):
        """
        See :meth:`.Camera._get_image_hw`. Computes and samples the affine-transformed SLM far-field.

        Returns
        -------
        numpy.ndarray
            Array of shape :attr:`shape`
        """

        # Update phase; calculate the far-field (keep on GPU if using cupy for follow-on interp)
        # FUTURE: in the case where sim is being used inside a GS loop, there could be
        # something clever here to use the existing Hologram's data.
        # Analog phase
        # self._hologram.reset_phase(self._slm.phase + self._slm.source["phase_sim"])
        # Quantized phase
        phase = -self._slm.display.astype(float)/self._slm.bitresolution*(2*np.pi)
        self._hologram.reset_phase(phase - phase.min() + self._slm.source["phase_sim"])

        ff = self._hologram.extract_farfield(get=True if (cp == np) else False)

        # Use map_coordinates for fastest interpolation
        # Note: by default, map_coordinates sets pixels outside the SLM k-space to 0 as desired
        if self._interpolate:
            img = map_coordinates(cp.abs(ff) ** 2, self.knm_cam, order=0)
        else:
            img = cp.abs(ff) ** 2
            img = toolbox.unpad(img, self.shape)
        if cp != np:
            img = img.get()

        # Quantize: all power in one pixel (img=1) -> maximum readout value at base exposure=1
        img = np.rint(img * self.exposure * self.bitresolution)

        # Basic noise sources. 
        if self.noise is not None:
            for key in self.noise.keys():
                if key == 'dark':
                    # Background/dark current - exposure dependent
                    dark = self.noise['dark'](np.ones_like(img) * self.bitresolution)/self.exposure
                    img = img + dark
                elif key == 'read':
                    # Readout noise - exposure independent
                    read = self.noise['read'](np.ones_like(img) * self.bitresolution)
                    img = img + read
                else: 
                    raise RuntimeError('Unknown noise source %s specified!'%(key))

        # Truncate to maximum readout value
        img[img > self.bitresolution] = self.bitresolution

        return img
