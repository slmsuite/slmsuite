from typing import Any
from slmsuite.holography.algorithms._header import *
from slmsuite.holography.algorithms._hologram import Hologram
from slmsuite.holography.algorithms._spots import SpotHologram, CompressedSpotHologram
from slmsuite.holography.algorithms._feedback import FeedbackHologram

class MultiplaneHologram(Hologram):
    """
    Holography combining multiple objectives, potentially across planes of focus or color.
    Other :class:`Hologram` subclasses are restricted to either optimizing a hologram
    within a fixed basis of spots or
    within the grid of a discrete Fourier transform at a fixed plane of focus.
    This :class:`MultiplaneHologram` acts as a metaclass to optimize many individual
    holograms simultaneously---over many planes or pointsets---producing a composite
    phase pattern.

    Note
    ~~~~
    Though the infrastructure to make this trivial is not yet in place,
    the idea of a 'plane' extends to planes of color. That is, this class
    :class:`MultiplaneHologram` could also be used to optimize a multicolor hologram and
    account for how the farfield of each color scales with wavelength.

    Attributes
    ----------
    holograms : list of :class:`Hologram`
        List of sub-holograms to optimize simultaneously.
    """
    def __init__(self, holograms, weights=None):
        """
        Initializes a 'meta' hologram consisting of several sub-holograms optimizing at
        the same time.

        Parameters
        ----------
        holograms : list of :class:`Hologram`
            List of ``N`` sub-holograms to optimize simultaneously.
        weights : array_like of float OR None
            List of ``N`` floats
            If ``None``, defaults to even power.
        """
        self.holograms = holograms

        # Check that all holograms are actually holograms and not MultiplaneHolograms.
        for h in self.holograms:
            if h.shape != self.holograms[0].shape:
                raise ValueError("Multiplane composite Holograms must have the same SLM shape.")
            if isinstance(h, MultiplaneHologram):
                raise ValueError("Multiplane hologram recursion is not supported.")
            if not isinstance(h, (Hologram, SpotHologram, CompressedSpotHologram, FeedbackHologram)):
                raise ValueError(f"Multiplane hologram must be provided child holograms, not {type(h)}")

        # Construct the parent hologram with empty goals but complete context.
        super().__init__(
            target=holograms[0].slm_shape,      # This hologram has a fake target.
            amp=holograms[0].amp,
            phase=holograms[0].phase,
            slm_shape=holograms[0].slm_shape,
            dtype=holograms[0].dtype,
        )
        self.target = None

        # Force all the child holograms to point to the same data.
        for h in self.holograms:
            h.amp = self.amp
            h.phase = self.phase

        # Parse weights
        if weights is None:
            weights = np.ones(len(self), dtype=self.dtype)

        self.weights = np.array(weights, copy=None, dtype=self.dtype)
        self.weights /= Hologram._norm(weights)

    def __len__(self):
        return len(self.holograms)

    # Overload user functions with meta functionality.

    def _update_flags(self, method, verbose, feedback, stat_groups, **kwargs):
        # First update the parent flags.
        super()._update_flags(method, verbose, feedback, stat_groups, **kwargs)

        # Then update each of the child flags. TODO: document this behavior.
        for h in self.holograms:
            h.flags.update(self.flags)

    def _update_weights(self, *args, **kwargs):
        for h in self.holograms: h._update_weights(*args, **kwargs)

    def _gs_farfield_routines(self, *args, **kwargs):
        for h in self.holograms: h._gs_farfield_routines(*args, **kwargs)

    def _get_target_moments_knm_norm(self):
        # Get the data from the child holograms.
        centers = []
        stds = []
        for h in self.holograms:
            center, std = h._get_target_moments_knm_norm()
            centers.append(center)
            stds.append(std)

        # Weight the centers.
        centers = np.vstack(centers)
        center = np.sum(np.square(self.weights).reshape(-1, 1) * centers, axis=0)

        # With the center, now weight the stds. We're doing an analytic integration of
        # x^2 over rectangles corresponding to the center \pm sqrt(3) * std of each hologram.
        stds = np.vstack(stds)

        c = centers - center.reshape(1, 2)
        l = c - stds * np.sqrt(3)
        r = c + stds * np.sqrt(3)

        integral_normalized = (r * r * r - l * l * l) / (2 * stds * np.sqrt(3)) / 3
        std = np.sqrt(np.sum(np.square(self.weights).reshape(-1, 1) * integral_normalized, axis=0))

        return center, std

    def reset(self, reset_phase=True, reset_flags=False):
        # Resetting the phase of the parent resets the phase of the children because
        # phase is shared.
        super().reset(reset_phase, reset_flags)

        # Reset the other child variables.
        for h in self.holograms: h.reset(reset_phase=False, reset_flags=reset_flags)

    def reset_weights(self):
        for h in self.holograms: h.reset_weights()

    def plot_farfield(self, *args, **kwargs):
        for h in self.holograms: h.plot_farfield(*args, **kwargs)

    # def plot_nearfield(self, *args, **kwargs):
    #     for h in self.holograms: h.plot_nearfield(*args, **kwargs)

    def plot_stats(self, *args, **kwargs):
        for h in self.holograms: h.plot_stats(*args, **kwargs)

    def _update_stats(self, stat_groups=[]):
        # TODO: make meta stat group.
        for h in self.holograms: h._update_stats(stat_groups)

    def set_target(self, *args, **kwargs):
        raise RuntimeError(
            "Do not use MultiplaneHologram.set_target(). "
            "Instead, update the targets of the children holograms directly."
        )

    # Multiplane hacks to get meta optimization to work.

    def _cg_loss(self, phase_torch):
        """Sum the losses of all the child holograms."""
        loss = self.holograms[0]._cg_loss(phase_torch)

        for h in self.holograms[1:]:
            loss += h._cg_loss(phase_torch)

        return loss

    def _nearfield2farfield(self):
        """Have all the holograms populate their own farfield variables."""
        for h in self.holograms:
            h._nearfield2farfield()

    def _farfield2nearfield(self):
        """Sum all the complex nearfields together for the meta nearfield."""
        self.nearfield.fill(0)

        for h, w in zip(self.holograms, self.weights):
            h._farfield2nearfield(extract=False)    # Avoid individually extracting phase.

            (i0, i1, i2, i3) = toolbox.unpad(h.shape, h.slm_shape)

            # Add the complex individual nearfields to our meta nearfield.
            if h.propagation_kernel is None:
                self.nearfield += w * h.nearfield[i0:i1, i2:i3]
            else:
                # Remove the propagation kernel if necessary.
                self.nearfield += w * h.nearfield[i0:i1, i2:i3] * cp.exp(-1j * h.propagation_kernel)

        # Get meta self phase.
        self._nearfield_extract()

    def _mraf_helper_routines(self):
        return {
            "mraf_enabled":False,
            "where_working":None,
            "signal_region":None,
            "noise_region":None,
            "zero_region":None,
        }