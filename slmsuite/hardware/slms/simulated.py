"""
Simulated SLM.
"""

import numpy as np
from .slm import SLM


class SimulatedSLM(SLM):
    """
    A simulated SLM to emulate physical artifacts of actual SLMs.

    Attributes
    ----------
    resolution : tuple
        (width, height) of the SLM in pixels.
    source : dict
        For a :class:`SimulatedSLM()`, :attr:`source` stores 'amplitude_sim' and 'phase_sim',
        which are used to compute the SLM's simulated far-field.
            'amplitude_sim' : numpy.ndarray
                User-defined source amplitude (with the dimensions of :attr:`shape`) on the SLM.
            'phase_sim' : numpy.ndarray
                User-defined source phase (with the dimensions of :attr:`shape`) on the SLM.
    """

    def __init__(self, resolution, source=None, **kwargs):
        r"""
        Initialize simulated slm.

        Arguments
        ------
        resolution : tuple
            See :attr:`resolution`.
        source : dict
            See :attr:`source`. Defaults to uniform illumination with a flat phase if ``None``.
        kwargs
            See :meth:`.SLM.__init__` for permissible options.
        """

        super().__init__(int(resolution[0]), int(resolution[1]), settle_time_s=0, **kwargs)

        if source is None:
            self.source["amplitude_sim"] = np.ones(resolution[::-1])
            self.source["phase_sim"] = np.zeros_like(self.x_grid)
        else:
            assert np.all(source["phase_sim"].shape == np.array(resolution)[::-1]) and np.all(
                source["amplitude_sim"].shape == np.array(resolution)[::-1]
            ), "The shape of the provided phase profile must match the SLM resolution!"
            self.source.update(source)

        self.write(None)

    def _write_hw(self, phase):
        """Updates SLM.display to implement various physical artifacts of SLMs."""

        # Apply the SLM phase (added to the simulated source phase)
        self.display = self._phase2gray(self.phase + self.source["phase_sim"], out=self.display)

        return
