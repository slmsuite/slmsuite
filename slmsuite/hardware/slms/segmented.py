"""
TODO
"""
import numpy as np

from slmsuite.hardware.slms.slm import SLM
from slmsuite.holography.toolbox import window_extent, window_slice, imprint

class SegmentedSLM(SLM):
    """
    A segment of a larger SLM.

    This class allows you to work with a specific region of a parent SLM
    as if it were a separate SLM.
    """

    def __init__(
        self,
        parent,
        window,
        name,
        update=False,
    ):
        r"""
        Initialize SLM and attributes.

        Parameters
        ----------
        parent : SLM
            The SLM to be segmented.
        window : (int, int, int, int) OR (array_like, array_like) OR array_like
            Format used by :func:`~slmsuite.holography.toolbox.window_slice`
            to define the window of interest on the parent SLM.
        name : str
            Name of this segment of the SLM.
        update : bool, optional
            If ``True``, this segment will update the parent SLM's display
            data after being called.
        """
        # Parse parent.
        if not isinstance(parent, SLM):
            raise ValueError("Parent must be an instance of SLM.")
        self.parent = parent
        self.update = bool(update)

        # Parse window.
        self.window = window_slice(window)
        self.extent = window_extent(window)
        self.extent_slice = window_slice(self.extent)
        resolution = (self.extent[1], self.extent[3])

        # Error check the window against the parent SLM's shape.
        if (
            self.extent[0] < 0 or self.extent[0] + self.extent[1] > parent.shape[1] or
            self.extent[2] < 0 or self.extent[2] + self.extent[3] > parent.shape[0]
        ):
            raise ValueError("Window is out of bounds of the parent SLM.")

        # Instantiate the superclass
        super().__init__(
            resolution,
            bitdepth=parent.bitdepth,
            name=name,
            wav_um=parent.wav_um,
            wav_design_um=parent.wav_design_um,
            pitch_um=parent.pitch_um,
            settle_time_s=parent.settle_time_s,
        )

        # Load source data from the parent SLM.
        self.source["amplitude"] = self.parent.source["amplitude"][*self.extent_slice]
        self.source["phase"] = self.parent.source["phase"][*self.extent_slice]

    def close(self):
        """Raise an error when attempting to close a segmented SLM."""
        raise RuntimeError("Close the parent SLM instead of the segmented SLM.")

    @staticmethod
    def info(verbose=True):
        """
        Prints instructions on how to use segmented SLMs.
        """
        if verbose:
            print("Call slm.segment() to produce child SegmentedSLMs.")

    def _set_phase_hw(
        self,
        display,
        update=None,
    ):
        """
        Overwrites the phase data in the parent SLM's display
        and writes the full parent display to hardware if desired.

        Parameters
        ----------
        display
            Integer data to display on the SLM. See :meth:`.SLM._set_phase_hw`.
        update : bool, optional
            Whether to update the full parent SLM.
        """
        # Update the parent SLM's display data.
        self.parent.display[*self.extent_slice] = display

        # Update the parent SLM's hardware if desired.
        if update is None:
            update = self.update
        if update:
            self.parent._set_phase_hw(self.parent.display["phase"])

    def set_input_trigger(self, on : bool = False):
        r"""
        Program the input trigger on the parent SLM.

        Raises
        ------
        RuntimeError
        """
        raise RuntimeError("Program the input trigger on the parent SLM.")

    def set_output_trigger(self, on : bool = False):
        r"""
        Program the output trigger on the parent SLM.

        Raises
        ------
        RuntimeError
        """
        raise RuntimeError("Program the output trigger on the parent SLM.")
