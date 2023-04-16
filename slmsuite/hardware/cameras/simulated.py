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

    def __init__(self, resolution, affine=None, **kwargs):
        """
        Initialize simulated camera.

        Parameters
        ----------
        resolution : tuple
            See :attr:`resolution`.
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

    def get_image(self, slm, plot=False):
        """
        See :meth:`.Camera.get_image`. Computes and samples the affine-transformed SLM far-field.

        Parameters
        ----------
        slm : :class:`~slmsuite.hardware.slms.simulated.SimulatedSLM`
            Simulated SLM creating the image.
        plot : bool
            Whether to plot the output.

        Returns
        -------
        numpy.ndarray
            Array of shape :attr:`shape`
        """

        # Efficiently calculate the far-field with a 0-iteration GS loop
        hologram = Hologram(slm.shape,phase=slm.phase+slm.phase_offset,slm_shape=slm)
        hologram.optimize(method='GS', maxiter=0)

        if plot:
            # Look at the associated near- and far- fields
            hologram.plot_nearfield(cbar=True)
            hologram.plot_farfield(cbar=True) #TODO: limits docstring update (tuple v. vec)

        im = hologram.amp_ff

        return im
