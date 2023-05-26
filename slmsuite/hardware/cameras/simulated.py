"""
Simulated camera to image the simulated SLM.
"""

try:
    import cupy as mp
except:
    import numpy as mp
import numpy as np
import matplotlib.pyplot as plt

from slmsuite.hardware.cameras.camera import Camera
from slmsuite.holography.algorithms import Hologram
from slmsuite.holography import toolbox


class SimulatedCam(Camera):
    """
    Simulated camera.

    Outputs simulated images an affine transform of computed far-field.
    Serves as a future testbed for simulation of other imaging artifacts, including non-affine
    abberations (e.g. pincushion distortion) and imaging readout noise.

    Attributes
    ----------
    resolution : tuple
        (width, height) of the SLM in pixels.
    exposure : float
        Digital gain value to simulate exposure time. Directly proprtional to imaged power.
    affine : tuple
        (M, b) 2x2 affine matrix and 2x1 offset vector to convert SLM k-space to camera-space.
    shape_padded : tuple
        Size of the FFT computational space required to faitfully reproduce the far-field at
        full camera resolution.
    """

    def __init__(self, resolution, slm, mag=None, theta=None, **kwargs):
        """
        Initialize simulated camera.

        Parameters
        ----------
        resolution : tuple
            See :attr:`resolution`.
        slm : :class:`~slmsuite.hardware.slms.simulated.SimulatedSLM`
            Simulated SLM creating the image.
        mag : float
            Magnification between camera and SLM fields of view.
        theta : float
            Rotation angle (in radians) between camera and SLM.
        kwargs
            See :meth:`.Camera.__init__` for permissible options.

        """

        super().__init__(int(resolution[0]), int(resolution[1]), **kwargs)

        # Digital gain emulates exposure
        self.exposure = 1

        # Store a reference to the SLM: we need this to compute the far-field camera images.
        self._slm = slm

        # Hologram for calculating the far-field
        # Padded shape: must 1) be >= slm.shape and 2) larger than resolution by a factor of 1/mag
        pad_order = max(
            [
                max([rs / rc for rs, rc in zip(slm.shape, self.shape)]),
                1 / mag if mag is not None else 1,
            ]
        )
        pad_order = np.round(pad_order).astype(int)
        self.shape_padded = Hologram.calculate_padded_shape(self.shape, pad_order)

        self._pad_window = toolbox.unpad(self.shape_padded, self.shape)
        self._hologram = Hologram(
            self.shape_padded,
            amp=self._slm.amp_profile,
            phase=self._slm.phase + self._slm.phase_offset,
            slm_shape=self._slm,
        )

        # Affine transform: slm -> cam
        if mag is None:
            mag = 1
        if theta is None:
            M = mp.array([[mag, 0], [0, mag]])
        else:
            rot = mp.array(
                [[mp.cos(theta), mp.sin(theta)], [-mp.sin(theta), mp.cos(theta)]]
            )
            M = mp.array([[mag, 0], [0, mag]]) @ rot
        c = mp.array(self.shape_padded)[mp.newaxis].T / 2
        self.affine = {"M": M, "b": c - M @ c}

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

    def get_image(self, plot=False):
        """
        See :meth:`.Camera.get_image`. Computes and samples the affine-transformed SLM far-field.

        Parameters
        ----------
        plot : bool
            Whether to plot the output.

        Returns
        -------
        numpy.ndarray
            Array of shape :attr:`shape`
        """

        # Update phase; calculate the far-field
        self._hologram.reset_phase(self._slm.phase + self._slm.phase_offset)
        ff = self._hologram.extract_farfield(affine=self.affine)

        # INCORRECT! Unpadding FF to same resolution --> magnification.

        img = (
            self.exposure * np.abs(ff[self._pad_window[0] : self._pad_window[1],
                                      self._pad_window[2] : self._pad_window[3],]) ** 2
        )

        if plot:
            # Look at the associated near- and far-fields
            # self._hologram.plot_nearfield(cbar=True)
            # self._hologram.plot_farfield(cbar=True)

            _, ax = plt.figure()
            # Note simualted cam currently has infinite dynamic range.
            ax.imshow(img, clim=[0, img.max()], interpolation="none")
            ax.set_title("Simulated Image")
            ax.set_xticks([])
            ax.set_yticks([])

        return img