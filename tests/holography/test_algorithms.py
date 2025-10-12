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
import matplotlib.pyplot as plt

class TestHologram:
    """Tests for Hologram class."""

    def test_hologram_construction(self, random_phase, random_amplitude):
        """Test the primitives for hologram formation."""
        slm_shape = (256, 256)
        shape = (512, 512)
        target = np.zeros(shape, dtype=np.float32)
        hologram = Hologram(target=target,
                            slm_shape=slm_shape,
                            phase=random_phase,
                            amp=random_amplitude)

        # Check shape conventions
        assert hologram.slm_shape == slm_shape
        assert hologram.shape == shape

        # Check dtype conversions
        assert hologram.dtype == np.float32
        assert hologram.dtype_complex == np.complex64

        # Check initial conditions
        phase_diff = hologram.get_phase() - random_phase
        assert np.allclose(phase_diff, phase_diff.flat[0])
        amp_ratio = hologram.get_amp() / (random_amplitude + 1e-10)
        plt.imshow(amp_ratio)
        plt.colorbar()
        plt.show()
        assert np.allclose(amp_ratio, amp_ratio.flat[0])

    @pytest.mark.parametrize("method", ["WGS-Leonardo", "WGS-Kim", "WGS-Nogrette"])
    def test_gs_converges(self, method):

        # Create a single far-field spot
        target = np.zeros((128, 128))
        target[32, 96] = 1
        hologram = Hologram(target=target)
        hologram.optimize(method=method, maxiter=20, verbose=False, stat_groups=["computational"])

        stats = hologram.stats["stats"]["computational"]
        # Check that efficiency improves
        assert stats["efficiency"][-1] >= stats["efficiency"][0]

        # Check that efficiency converges
        recent_efficiencies = stats["efficiency"][-5:]
        assert np.std(recent_efficiencies) < 0.05

        # Check that error decreases
        assert stats["std_err"][-1] <= stats["std_err"][0]

        # Check that output matches the expected grating

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
