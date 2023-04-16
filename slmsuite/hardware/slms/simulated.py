"""
Simulated SLM. 

Effects considered:
- TODO: Phase offset due to physical curvature.
- TODO: Phase crosstalk
- TODO: Incomplete fill
- TODO: Imperfect phase control across aperture
"""

import numpy as np
from .slm import SLM


class SimulatedSLM(SLM):
    """
    A simulated SLM to emmulate physical artifacts of actual SLMs.

    Attributes
    ----------
    resolution : tuple
        (width, height) of the SLM in pixels.
    phase_offset : ndarray
        Phase delay array (radians) emmulating the result of physical SLM curvature. 
    amp_profile : ndarray
        Simulated amplitude profile illuminating the SLM. 
    """

    def __init__(self, resolution, phase_offset=None, amp_profile=None, **kwargs):
        r"""
        Initializes an instance of a Santec SLM.

        Arguments
        ------
        resolution : tuple
            See :attr:`resolution`.
        phase_offset : ndarray
            See :attr:`phase_offset`.
        amp_profile : ndarray
            See :attr:`amp_profile`.
        kwargs
            See :meth:`.SLM.__init__` for permissible options.
        """

        if phase_offset is None:
            self.phase_offset = 2*np.pi*np.random.rand(resolution[1],resolution[0])
        else:
            self.phase_offset = phase_offset
        # TODO: else, build phase mask based on provided zernike coeffs.

        if amp_profile is None:
            self.amp_profile = np.ones(resolution)
        # TODO: else, build some desired amplitudes (gaussians and non-gaussian)

        super().__init__(
            int(resolution[0]),
            int(resolution[1]),
            **kwargs
        )

        self.write(None)

    def _write_hw(self, phase):
        """Updates SLM.display to implement various physical artifacts of SLMs."""

        # Apply the phase_offset due to physical curvature of the SLM panel

        # Apply phase coupling

        self.display = self._phase2gray(self.phase+self.phase_offset, out=self.display)
        return


    

