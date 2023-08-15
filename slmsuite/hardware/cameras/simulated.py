"""
Simulated camera to image the simulated SLM.
"""

try:
    import cupy as mp
    from cupyx.scipy.ndimage import map_coordinates
except:
    import numpy as mp
    from scipy.ndimage import map_coordinates
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

    def __init__(self, resolution, slm, f_eff=None, theta=0, offset=(0,0), basis="ij", **kwargs):
        """
        Initialize simulated camera.

        Parameters
        ----------
        resolution : tuple
            See :attr:`resolution`.
        slm : :class:`~slmsuite.hardware.slms.simulated.SimulatedSLM`
            Simulated SLM creating the image.
        f_eff : float
            Effective focal length (in `basis` units) of the
            optical train separating the Fourier-domain SLM from the camera.
            If `None`, defaults to the minimum focal length for which the camera is
            fully contained within the SLM's accessible Fourier space.
        theta : float
            Rotation angle (in radians, ccw) of the camera from the SLM axis.
            Defaults to 0 (i.e., aligned with the SLM).
        offset : tuple
            Lateral displacement (in `basis` units) of the camera from the center of the SLM 
        basis : str
            Sets the units for `f_eff` and `offset`. Currently, only `"ij"` is supported.
            TODO: Add support for `"um"`.
        kwargs
            See :meth:`.Camera.__init__` for permissible options.

        TODO: add some notes here about the computation process/unit conversions, etc. 

        """

        super().__init__(int(resolution[0]), int(resolution[1]), **kwargs)

        # Digital gain emulates exposure
        self.exposure = 1

        # Store a reference to the SLM: we need this to compute the far-field camera images.
        self._slm = slm

        # Compute the camera pixel grid in `basis` units (currently "ij")
        self.xgrid, self.ygrid = mp.meshgrid(mp.linspace(-1/2,1/2,resolution[0])*resolution[0],
                                             mp.linspace(-1/2,1/2,resolution[1])*resolution[1])
        if theta != 0:
            rot = mp.array(
                [[mp.cos(-theta), mp.sin(-theta)], [-mp.sin(-theta), mp.cos(-theta)]]
            )
            # Rotate
            self.xgrid, self.ygrid = mp.einsum('ji, mni -> jmn', rot,
                                               mp.dstack([self.xgrid, self.ygrid]))
        # Translate
        self.xgrid = self.xgrid + offset[0]
        self.ygrid = self.ygrid + offset[1]

        # Compute SLM Fourier-space grid in `basis` units (currently "ij")
        f_min = 2*max([mp.amax(mp.abs(self.xgrid))*slm.dx,
                       mp.amax(mp.abs(self.ygrid))*slm.dy])
        if f_eff is None:
            self.f_eff = f_min
            print("Setting f_eff = f_min = %1.2f pix/rad to place camera \
within accessible SLM k-space."%(self.f_eff))
        elif f_eff < f_min:
            raise RuntimeError("Camera extends beyond SLM's accessible Fourier space!")
        else:
            self.f_eff = f_eff
        
        # Fourier space must be sufficiently padded to resolve the camera pixels
        # TODO: account for small rotation factors
        self.shape_padded = Hologram.calculate_padded_shape(slm, precision=1/self.f_eff)
        self._hologram = Hologram(
            self.shape_padded,
            amp=self._slm.amp_profile,
            phase=self._slm.phase + self._slm.phase_offset,
            slm_shape=self._slm,
        )
        print("Padded SLM k-space shape set to (%d,%d) to achieve \
required imaging resolution."%(self.shape_padded[1], self.shape_padded[0]))

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

        # Update phase; calculate the far-field (keep on GPU if using cupy for follow-on interp)
        self._hologram.reset_phase(self._slm.phase + self._slm.phase_offset)
        ff = self._hologram.extract_farfield(get = True if (mp==np) else False)

        # Use map_coordinates for fastest interpolation; but need to reshape pixel dimensions
        # to account for additional padding.
        img = map_coordinates(mp.abs(ff)**2,mp.array([
                                            self.shape_padded[0]/(self.f_eff/self._slm.dy)*
                                            self.ygrid+self.shape_padded[0]/2,
                                            self.shape_padded[1]/(self.f_eff/self._slm.dx)*
                                            self.xgrid+self.shape_padded[1]/2]),
                                            order=0)
        if mp != np:
            img = img.get()
        img = self.exposure * img

        if plot:
            # Look at the associated near- and far-fields
            # self._hologram.plot_nearfield(cbar=True)
            # self._hologram.plot_farfield(cbar=True)

            # Note simualted cam currently has infinite dynamic range.
            plt.imshow(img, clim=[0, img.max()], interpolation="none")
            plt.colorbar()
            ax = plt.gca()
            ax.set_title("Simulated Image")
            # ax.set_xticks([])
            # ax.set_yticks([])

        return img