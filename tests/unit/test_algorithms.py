"""
Unit tests for slmsuite.holography.algorithms module.
"""
import pytest
import numpy as np

try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    cp = np
    HAS_CUPY = False

from slmsuite.holography.algorithms import Hologram


def convert_stats_to_list(hologram):
    """
    Convert hologram.stats from dict-of-lists to list-of-dicts format.

    Parameters
    ----------
    hologram : Hologram
        The hologram object containing stats.

    Returns
    -------
    list of dict
        Stats in list-of-dicts format, where each dict represents one iteration.
        Returns empty list if no stats are available.
    """
    if hologram.stats is None or "stats" not in hologram.stats:
        return []

    stats_dict = hologram.stats["stats"]
    if not stats_dict:
        return []

    # Collect all stat groups (e.g., "computational", "experimental")
    result = []

    # Get the length from the first stat group
    first_group = next(iter(stats_dict.values()))
    if not first_group:
        return []

    first_key = next(iter(first_group.keys()))
    num_iterations = len(first_group[first_key])

    # Convert dict-of-lists to list-of-dicts
    for i in range(num_iterations):
        iter_stats = {}
        for group_name, group_stats in stats_dict.items():
            for stat_name, stat_values in group_stats.items():
                iter_stats[stat_name] = stat_values[i]
        result.append(iter_stats)

    return result


class TestHologramConstruction:
    """Tests for Hologram class construction."""

    def test_hologram_basic_construction(self):
        """Test basic Hologram construction."""
        target = np.zeros((256, 256))
        target[120:136, 120:136] = 1  # Square target

        hologram = Hologram(target=target)

        assert hologram.shape == (256, 256)
        assert hologram.slm_shape == (256, 256)
        assert hologram.phase is not None
        assert hologram.phase.shape == (256, 256)

    def test_hologram_with_padding(self):
        """Test Hologram construction with padding."""
        slm_shape = (256, 256)
        shape = (512, 512)
        target = np.zeros(shape)
        target[240:272, 240:272] = 1

        hologram = Hologram(target=target, slm_shape=slm_shape)

        assert hologram.slm_shape == slm_shape
        assert hologram.shape == shape

    def test_hologram_dtype(self):
        """Test Hologram dtype handling."""
        target = np.zeros((128, 128), dtype=np.float32)
        target[60:68, 60:68] = 1

        hologram = Hologram(target=target, dtype=np.float32)

        assert hologram.dtype == np.float32
        assert hologram.dtype_complex == np.complex64

    def test_hologram_with_initial_phase(self):
        """Test Hologram with user-provided initial phase."""
        target = np.zeros((128, 128))
        target[60:68, 60:68] = 1
        initial_phase = np.random.rand(128, 128) * 2 * np.pi

        hologram = Hologram(target=target, phase=initial_phase)

        assert np.array_equal(hologram.phase, initial_phase)

    def test_hologram_with_amplitude_constraint(self):
        """Test Hologram with nearfield amplitude constraint."""
        target = np.zeros((128, 128))
        target[60:68, 60:68] = 1
        amp = np.ones((128, 128)) * 0.5  # Reduced amplitude

        hologram = Hologram(target=target, amp=amp)

        assert hologram.amp is not None
        assert np.array_equal(hologram.amp, amp)


class TestHologramShapeConvention:
    """Test that shape conventions are followed."""

    def test_shape_convention_height_width(self):
        """Verify (height, width) convention."""
        height, width = 100, 200
        target = np.zeros((height, width))
        target[45:55, 95:105] = 1

        hologram = Hologram(target=target)

        assert hologram.shape[0] == height
        assert hologram.shape[1] == width
        assert hologram.phase.shape[0] == height
        assert hologram.phase.shape[1] == width


class TestHologramGS:
    """Tests for Gerchberg-Saxton algorithm."""

    def test_gs_runs_without_error(self):
        """Test that GS algorithm runs without errors."""
        target = np.zeros((128, 128))
        target[60:68, 60:68] = 1

        hologram = Hologram(target=target)
        hologram.optimize(method="GS", maxiter=10, verbose=False, stat_groups=["computational"])
        stats = convert_stats_to_list(hologram)

        assert stats is not None
        assert len(stats) > 0

    def test_gs_improves_efficiency(self):
        """Test that GS improves hologram efficiency."""
        target = np.zeros((128, 128))
        target[60:68, 60:68] = 1

        hologram = Hologram(target=target)

        # Run a few iterations
        hologram.optimize(method="GS", maxiter=20, verbose=False, stat_groups=["computational"])
        stats = convert_stats_to_list(hologram)

        # Efficiency should improve (later iterations better than first)
        assert stats[-1]["efficiency"] >= stats[0]["efficiency"]

    def test_gs_converges(self):
        """Test that GS converges over iterations."""
        target = np.zeros((128, 128))
        target[60:68, 60:68] = 1

        hologram = Hologram(target=target)
        hologram.optimize(method="GS", maxiter=50, verbose=False, stat_groups=["computational"])
        stats = convert_stats_to_list(hologram)

        # Efficiency should stabilize (last 10 iters similar)
        recent_efficiencies = [s["efficiency"] for s in stats[-10:]]
        efficiency_std = np.std(recent_efficiencies)
        assert efficiency_std < 0.05  # Small variation indicates convergence

    def test_gs_with_callback(self):
        """Test GS with callback function."""
        target = np.zeros((128, 128))
        target[60:68, 60:68] = 1

        hologram = Hologram(target=target)

        callback_count = [0]
        def callback(hologram):
            callback_count[0] += 1

        hologram.optimize(method="GS", maxiter=10, callback=callback, verbose=False, stat_groups=["computational"])

        assert callback_count[0] == 10


class TestHologramWGS:
    """Tests for Weighted Gerchberg-Saxton variants."""

    @pytest.mark.parametrize("method", ["WGS-Leonardo", "WGS-Kim", "WGS-Nogrette"])
    def test_wgs_methods_run(self, method):
        """Test that various WGS methods run without errors."""
        target = np.zeros((128, 128))
        target[60:68, 60:68] = 1

        hologram = Hologram(target=target)
        hologram.optimize(method=method, maxiter=10, verbose=False, stat_groups=["computational"])
        stats = convert_stats_to_list(hologram)

        assert stats is not None
        assert len(stats) > 0

    def test_wgs_modifies_weights(self):
        """Test that WGS modifies the weights array."""
        target = np.zeros((128, 128))
        target[60:68, 60:68] = 1

        hologram = Hologram(target=target)
        initial_weights = hologram.weights.copy()

        hologram.optimize(method="WGS-Leonardo", maxiter=10, verbose=False, stat_groups=["computational"])

        # Weights should have changed
        assert not np.array_equal(hologram.weights, initial_weights)


class TestHologramMRAF:
    """Tests for Mixed Region Amplitude Freedom."""

    def test_mraf_with_nan_target(self):
        """Test MRAF by setting target regions to NaN."""
        target = np.zeros((128, 128))
        target[60:68, 60:68] = 1  # Signal region
        target[0:20, 0:20] = np.nan  # Noise region (ignored)

        hologram = Hologram(target=target)
        hologram.optimize(method="GS", maxiter=10, verbose=False, stat_groups=["computational"])
        stats = convert_stats_to_list(hologram)

        assert stats is not None
        # Should run without errors despite NaN values


class TestHologramStatistics:
    """Tests for hologram statistics and metrics."""

    def test_stats_dict_structure(self):
        """Test that stats dictionaries have expected keys."""
        target = np.zeros((128, 128))
        target[60:68, 60:68] = 1

        hologram = Hologram(target=target)
        hologram.optimize(method="GS", maxiter=5, verbose=False, stat_groups=["computational"])
        stats = convert_stats_to_list(hologram)

        # Check that stats have expected keys
        assert "efficiency" in stats[0]
        assert "uniformity" in stats[0]

    def test_efficiency_range(self):
        """Test that efficiency is in valid range."""
        target = np.zeros((128, 128))
        target[60:68, 60:68] = 1

        hologram = Hologram(target=target)
        hologram.optimize(method="GS", maxiter=10, verbose=False, stat_groups=["computational"])
        stats = convert_stats_to_list(hologram)

        for stat in stats:
            assert 0 <= stat["efficiency"] <= 1


class TestHologramPadding:
    """Tests for zero-padding functionality."""

    def test_get_padded_shape(self):
        """Test get_padded_shape static method."""
        slm_shape = (1080, 1920)
        padding = 2

        padded_shape = Hologram.get_padded_shape(slm_shape, padding=padding)

        assert padded_shape[0] == slm_shape[0] * padding
        assert padded_shape[1] == slm_shape[1] * padding

    def test_padding_improves_resolution(self):
        """Test that padding improves farfield resolution."""
        # Small target spot
        target_small = np.zeros((64, 64))
        target_small[30:34, 30:34] = 1

        target_padded = np.zeros((256, 256))
        target_padded[126:130, 126:130] = 1

        hologram_small = Hologram(target=target_small)
        hologram_padded = Hologram(target=target_padded, slm_shape=(64, 64))

        # Padded version should have better resolution
        assert hologram_padded.shape > hologram_small.shape


@pytest.mark.skipif(not HAS_CUPY, reason="CuPy not available")
class TestHologramGPU:
    """Tests for GPU acceleration with CuPy."""

    def test_hologram_on_gpu(self):
        """Test that Hologram can run on GPU."""
        target = cp.zeros((128, 128))
        target[60:68, 60:68] = 1

        hologram = Hologram(target=target)

        # Should create phase on GPU
        assert isinstance(hologram.phase, cp.ndarray)

    def test_gs_on_gpu(self):
        """Test GS algorithm on GPU."""
        target = cp.zeros((128, 128))
        target[60:68, 60:68] = 1

        hologram = Hologram(target=target)
        hologram.optimize(method="GS", maxiter=10, verbose=False, stat_groups=["computational"])
        stats = convert_stats_to_list(hologram)

        assert stats is not None
        # Results should still be on GPU
        assert isinstance(hologram.phase, cp.ndarray)


class TestHologramEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_target_error_or_warning(self):
        """Test handling of all-zero target."""
        target = np.zeros((128, 128))

        # Should either raise error or run with warning
        # Different behavior might be acceptable
        hologram = Hologram(target=target)
        # Test that it doesn't crash
        assert hologram.target is not None

    def test_small_hologram(self):
        """Test with very small hologram."""
        target = np.zeros((16, 16))
        target[7:9, 7:9] = 1

        hologram = Hologram(target=target)
        hologram.optimize(method="GS", maxiter=5, verbose=False, stat_groups=["computational"])
        stats = convert_stats_to_list(hologram)

        assert stats is not None

    def test_rectangular_hologram(self):
        """Test with non-square hologram."""
        target = np.zeros((128, 256))  # 2:1 aspect ratio
        target[60:68, 120:136] = 1

        hologram = Hologram(target=target)
        hologram.optimize(method="GS", maxiter=5, verbose=False, stat_groups=["computational"])
        stats = convert_stats_to_list(hologram)

        assert stats is not None
        assert hologram.shape == (128, 256)
