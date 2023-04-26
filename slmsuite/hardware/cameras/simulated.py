"""
Simulated camera to image the simulated SLM.


Effects considered:
- TODO: Phase offset due to physical curvature.
"""

import numpy as np
from slmsuite.hardware.cameras.camera import Camera
from slmsuite.holography.algorithms import Hologram

class SimulatedCam(Camera):
    """
    Simulated camera. Image output is 

    Attributes
    ----------
    resolution : tuple
        (width, height) of the SLM in pixels.
    affine : tuple
        (M, b) 2x2 affine matrix and 2x1 offset vector to convert SLM k-space to camera-space. 
    """

    def __init__(self, resolution, slm, affine=None, **kwargs):
        """
        Initialize simulated camera.

        Parameters
        ----------
        resolution : tuple
            See :attr:`resolution`.
        slm : :class:`~slmsuite.hardware.slms.simulated.SimulatedSLM`
            Simulated SLM creating the image.
        affine : ndarray
            See :attr:`M`.
        kwargs
            See :meth:`.Camera.__init__` for permissible options.

        """
        if affine is None:
            self.affine = (np.array([[1,0],[0,1]]), np.array([0,0]))
        else:
            self.affine = affine

        super().__init__(
            int(resolution[0]),
            int(resolution[1]),
            **kwargs
        )

        # Digital gain emulates exposure
        self.exposure = 1
        
        # Store a reference to the SLM: we need this to compute the far-field camera images.
        self._slm = slm

        # Hologram for calculating the far-field
        self.hologram = Hologram(self.shape,
                                 amp=self._slm.amp_profile,
                                 phase=self._slm.phase+self._slm.phase_offset,
                                 slm_shape=self._slm)

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
        self.hologram.reset_phase(self._slm.phase + self._slm.phase_offset)
        ff = self.hologram.extract_farfield()

        if plot:
            # Look at the associated near- and far-fields
            self.hologram.plot_nearfield(cbar=True)
            self.hologram.plot_farfield(cbar=True)

        return self.exposure*np.abs(ff)**2

