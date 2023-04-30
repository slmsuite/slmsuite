"""
Simulated camera to image the simulated SLM.


Effects considered:
- TODO: Phase offset due to physical curvature.
"""

try:
    import cupy as mp
except:
    import numpy as mp
import numpy as np

from slmsuite.hardware.cameras.camera import Camera
from slmsuite.holography.algorithms import Hologram
from slmsuite.holography import toolbox


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

    def __init__(self, resolution, slm, mag=None, theta=None, **kwargs):
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
        # Padded shape: must be >= slm.shape and larger than resolution by a factor of 1/mag
        pad_order = max([max([rs/rc for rs,rc in zip(slm.shape,self.shape)]), 
                         1/mag if mag is not None else 1])
        pad_order = np.ceil(pad_order).astype(int)
        self.shape_padded = Hologram.calculate_padded_shape(self.shape,pad_order)
        self.pad_window = toolbox.unpad(self.shape_padded, self.shape)
        self.hologram = Hologram(self.shape_padded,
                                 amp=self._slm.amp_profile,
                                 phase=self._slm.phase+self._slm.phase_offset,
                                 slm_shape=self._slm)
        
        # Affine transform: slm -> cam
        if mag is None: mag = 1
        if theta is None: 
            M = mp.array([[mag,0],[0,mag]])
        else:
            rot = mp.array([[mp.cos(theta), mp.sin(theta)],[-mp.sin(theta),mp.cos(theta)]])
            M = mp.array([[mag,0],[0,mag]]) @ rot
        c = mp.array(self.shape_padded)[mp.newaxis].T/2
        self.affine = {"M":M,"b":c - M @ c}

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
        ff = self.hologram.extract_farfield(affine=self.affine)
        # self.hologram.optimize(maxiter=0)

        if plot:
            # Look at the associated near- and far-fields
            self.hologram.plot_nearfield(cbar=True)
            self.hologram.plot_farfield(cbar=True)

        return self.exposure*np.abs(ff[self.pad_window[0]:self.pad_window[1],
                                       self.pad_window[2]:self.pad_window[3]])**2