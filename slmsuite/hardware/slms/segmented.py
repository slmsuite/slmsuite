"""
A segment of a larger SLM.
This class allows the user to work with a specific region
of a parent SLM as if it were a separate SLM.
"""
import numpy as np

from slmsuite.hardware.slms.slm import SLM
from slmsuite.holography.toolbox import window_extent, window_slice

class SegmentedSLM(SLM):
    """
    A segment of a larger SLM.

    This class allows the user to work with a specific region of a parent SLM
    as if it were a separate SLM.

    Attributes
    ----------
    parent : SLM
        The parent SLM of which this is a segment.
    refresh : bool
        If ``True``, this segment will by default project the entire parent SLM's
        display after being updated.
    extent_slice : tuple of slice
        The rectangular extent of this segment on the parent SLM in slice format.
    subwindow : None OR tuple of arrays of indices
        If the window of this segment is non-rectangular, this stores the indices of the
        pixels in the segment's window within the rectangular extent. If the window is
        rectangular, this is ``None``.
    """

    def __init__(
        self,
        parent,
        window,
        name,
        refresh=False,
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
        refresh : bool, optional
            If ``True``, this segment will by default project the entire parent SLM's
            display after being updated.
        """
        # Parse parent.
        if not isinstance(parent, SLM):
            raise ValueError("Parent must be an instance of SLM.")
        self.parent = parent
        self.refresh = bool(refresh)

        # Parse window.
        window = window_slice(window, shape=parent.shape)  # 2 slice, 2 indices, or boolean array format

        # Get the rectangular extent of the window.
        extent = window_extent(window)                     # (x, w, y, h) format
        self.extent_slice = window_slice(extent)           # 2 slice format

        # Handle the case where the window is not rectangular.
        self.subwindow = None
        if np.ndim(window) == 2:    # Boolean array
            self.subwindow = window[*self.extent_slice]
        elif len(window) == 2 and not isinstance(window[0], slice):     # Lists of indices
            self.subwindow = (
                window[0] - extent[0],
                window[1] - extent[2]
            )

        # Error check the window against the parent SLM's shape.
        if (
            extent[0] < 0 or extent[0] + extent[1] > parent.shape[1] or
            extent[2] < 0 or extent[2] + extent[3] > parent.shape[0]
        ):
            raise ValueError("Window is out of bounds of the parent SLM.")

        # Instantiate the superclass
        super().__init__(
            (extent[1], extent[3]),
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
        return []

    def _set_phase_hw(
        self,
        display,
        refresh=None,
    ):
        """
        Overwrites the phase data in the parent SLM's display
        and writes the full parent display to hardware if desired.

        Parameters
        ----------
        display
            Integer data to display on the SLM. See :meth:`.SLM._set_phase_hw`.
        refresh : bool, optional
            Whether to update the full parent SLM.
            If ``None``, uses the value of ``self.refresh``, which is ``True``
            for the final segment of a segmented SLM by default.
        """
        # Update the parent SLM's display data.
        if self.subwindow is None:                  # Rectangular window case
            self.parent.display[*self.extent_slice] = display
        else:                                       # Non-rectangular window case
            self.parent.display[*self.extent_slice][*self.subwindow] = display[*self.subwindow]

        # Update the parent SLM's hardware if desired.
        if refresh is None:
            refresh = self.refresh
        if refresh:
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
