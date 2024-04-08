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
    source : dict
        For a :class:`SimulatedSLM()`, :attr:`source` stores 'amplitude_sim' and 'phase_sim',
        which are used to compute the SLM's simulated far-field.
            'amplitude_sim' : numpy.ndarray
                User-defined source amplitude (with the dimensions of :attr:`shape`) on the SLM.
            'phase_sim' : numpy.ndarray
                User-defined source phase (with the dimensions of :attr:`shape`) on the SLM.
    """

    def __init__(self, width, height, source=None, **kwargs):
        r"""
        Initialize simulated slm.

        Parameters
        ----------
        width : int
            Width of the SLM in pixels.
        height : int
            Height of the SLM in pixels
        source : dict
            See :attr:`source`. Defaults to uniform illumination with a flat phase if ``None``.
        kwargs
            See :meth:`.SLM.__init__` for permissible options.
        """
        super().__init__(width, height, settle_time_s=0, **kwargs)

        if source is None:
            self.source["amplitude_sim"] = np.ones_like(self.x_grid)
            self.source["phase_sim"] = np.zeros_like(self.x_grid)
        else:
            # assert np.all([source[kw].shape == self.shape for kw in source.keys()]
            # ), "The shape of the provided phase profile must match the SLM resolution!"
            self.source.update(source)

            # Handle case where `source` only has real values from experiment
            if "amplitude_sim" not in source.keys():
                self.source["amplitude_sim"] = self.source["amplitude"]
                self.source["phase_sim"] = -self.source["phase"]

        self.write(None)

    def _write_hw(self, phase):
        """Updates SLM.display to implement various physical artifacts of SLMs."""

        # FUTURE: apply physical effects directly to SLM.display

        return
