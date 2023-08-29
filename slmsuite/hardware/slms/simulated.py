"""
Simulated SLM.
"""

import numpy as np
from .slm import SLM
from slmsuite.misc.fitfunctions import gaussian2d

class SimulatedSLM(SLM):
    """
    A simulated SLM to emmulate physical artifacts of actual SLMs.

    Attributes
    ----------
    resolution : tuple
        (width, height) of the SLM in pixels.
    phase_offset : ndarray
        Phase delay array (radians) emulating the result of physical SLM curvature.
    amp_profile : ndarray
        Simulated amplitude profile illuminating the SLM.
    """

    def __init__(self, resolution, phase_offset=None, amp_profile=None, **kwargs):
        r"""
        Initialize simulated slm.

        Arguments
        ------
        resolution : tuple
            See :attr:`resolution`.
        phase_offset : ndarray
            See :attr:`phase_offset`. Defaults to flat phase if ``None``.
        amp_profile : tuple or ndarray
            See :attr:`amp_profile`. Defaults to uniform illumination if ``None``.
            If a tuple is provided, amp_profile contains the fractional (i.e. relative to
            panel size) standard deviation of a Gaussian *amplitude* profile. Otherwise,
            the provided ndarray is directly applied as an amplitude profile.
        kwargs
            See :meth:`.SLM.__init__` for permissible options.
        """

        super().__init__(
            int(resolution[0]),
            int(resolution[1]),
            settle_time_s = 0,
            **kwargs
        )

        if phase_offset is None:
            self.phase_offset = np.zeros_like(self.x_grid)
        else:
            assert np.all(phase_offset.shape == np.array(resolution)[::-1]), \
            "The shape of the provided phase profile must match the SLM resolution!"

            self.phase_offset = phase_offset

        if amp_profile is None:
            self.amp_profile = np.ones(resolution[::-1])
        elif isinstance(amp_profile, tuple):
            # Gaussian profile w/ amp_profile/width std dev.
            self.amp_profile = gaussian2d(np.array(list((zip(
                                          (self.y_grid,self.x_grid))))).squeeze(), 0, 0, 1, 0,
                                          amp_profile[1]*resolution[1],
                                          amp_profile[0]*resolution[0])
        else:
            assert np.all(amp_profile.shape == np.array(resolution)[::-1]), \
            "The shape of the provided amplitude profile must match the SLM resolution!"

            self.amp_profile = amp_profile

        self.write(None)

    def _write_hw(self, phase):
        """Updates SLM.display to implement various physical artifacts of SLMs."""

        # Apply the phase_offset due to physical curvature of the SLM panel
        self.display = self._phase2gray(self.phase+self.phase_offset, out=self.display)

        return