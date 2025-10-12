"""
Unit tests for SLM base class.
"""
import pytest
import numpy as np
from slmsuite.hardware.slms.simulated import SimulatedSLM

def test_slm_init(slm):
    """Test basic SimulatedSLM construction."""
    # Test basic SLM attributes.
    assert slm.shape == (1080, 1920)
    assert slm.bitdepth == 8
    assert slm.bitresolution == 256
    assert slm.wav_um == 0.78

    # Verify (height, width) shape convention.
    height, width = 1080, 1920
    slm = SimulatedSLM(resolution=(width, height))  # Note: resolution is (width, height)

    assert slm.shape[0] == height
    assert slm.shape[1] == width

    # Test that grid is properly created.
    assert slm.grid is not None
    assert len(slm.grid) == 2
    assert slm.grid[0].shape == slm.shape
    assert slm.grid[1].shape == slm.shape

    # Test pitch_um attribute.
    assert np.allclose(slm.pitch_um, [8.0, 8.0])

    slm.test()  # Basic self-test should pass

    slm.close()

def test_slm_test(slm):
    """Test that the SLM's self-test method works."""
    # The test method should return True on success
    result = slm.test()
    assert result is True


class TestSLMEdgeCases:
    """Test edge cases and error handling."""

    def test_phase_wrong_shape(self, slm):
        """Test handling of incorrect phase shape."""
        wrong_phase = np.zeros((100, 100))

        # Depending on implementation, this might resize or error
        # For now, just test it doesn't crash catastrophically
        try:
            slm.set_phase(wrong_phase)
        except (ValueError, AssertionError):
            pass  # Expected to fail

    def test_negative_phase(self, slm):
        """Test handling of negative phase values."""
        phase = -np.ones(slm.shape) * np.pi

        gray = slm._phase2gray(phase)

        # Should wrap to positive values
        assert np.all(gray >= 0)
        assert np.all(gray < slm.bitresolution)

    def test_large_phase_values(self, slm):
        """Test handling of phase values > 2π."""
        phase = np.ones(slm.shape) * 10 * np.pi

        gray = slm._phase2gray(phase)

        # Should wrap via modulo
        assert np.all(gray >= 0)
        assert np.all(gray < slm.bitresolution)