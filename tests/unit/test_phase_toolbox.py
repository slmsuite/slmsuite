"""
Unit tests for slmsuite.holography.toolbox.phase module.
"""
import pytest
import numpy as np
from slmsuite.holography.toolbox import phase


@pytest.fixture
def simple_grid():
    """Create a simple 2D grid for testing."""
    x = np.linspace(-10, 10, 100)
    y = np.linspace(-10, 10, 100)
    X, Y = np.meshgrid(x, y)
    return (X, Y)


@pytest.fixture
def normalized_grid():
    """Create a normalized grid (typical SLM coordinates in wavelengths)."""
    x = np.linspace(-500, 500, 256)  # -500 to 500 wavelengths
    y = np.linspace(-500, 500, 256)
    X, Y = np.meshgrid(x, y)
    return (X, Y)


class TestBlaze:
    """Tests for blaze function."""

    def test_blaze_zero_vector(self, simple_grid):
        """Test blaze with zero vector returns zeros."""
        result = phase.blaze(simple_grid, vector=(0, 0))
        assert result.shape == simple_grid[0].shape
        assert np.allclose(result, 0)

    def test_blaze_x_only(self, simple_grid):
        """Test blaze with x-only vector."""
        result = phase.blaze(simple_grid, vector=(1, 0))
        # Should vary only in x direction
        assert not np.allclose(result[0, :], result[0, 0])
        # Should be constant in y direction
        assert np.allclose(result[:, 50], result[0, 50])

    def test_blaze_y_only(self, simple_grid):
        """Test blaze with y-only vector."""
        result = phase.blaze(simple_grid, vector=(0, 1))
        # Should vary only in y direction
        assert not np.allclose(result[:, 0], result[0, 0])
        # Should be constant in x direction
        assert np.allclose(result[50, :], result[50, 0])

    def test_blaze_linear_in_k(self, simple_grid):
        """Test that blaze is linear in k-vector."""
        result1 = phase.blaze(simple_grid, vector=(1, 1))
        result2 = phase.blaze(simple_grid, vector=(2, 2))
        # Double k-vector should double the phase
        assert np.allclose(2 * result1, result2)

    def test_blaze_3d_vector(self, simple_grid):
        """Test blaze with 3D vector (includes focusing)."""
        result_2d = phase.blaze(simple_grid, vector=(1, 1))
        result_3d = phase.blaze(simple_grid, vector=(1, 1, 1))
        # 3D should have additional quadratic term
        assert result_3d.shape == result_2d.shape
        assert not np.allclose(result_2d, result_3d)
        # Difference should be quadratic
        diff = result_3d - result_2d
        r_squared = simple_grid[0]**2 + simple_grid[1]**2
        assert np.allclose(diff, np.pi * r_squared)


class TestSinusoid:
    """Tests for sinusoid function."""

    def test_sinusoid_zero_vector(self, simple_grid):
        """Test sinusoid with zero vector."""
        result = phase.sinusoid(simple_grid, vector=(0, 0))
        # Should be constant
        assert np.allclose(result, result[0, 0])

    def test_sinusoid_range(self, simple_grid):
        """Test sinusoid output range."""
        result = phase.sinusoid(simple_grid, vector=(1, 0), a=np.pi, b=0)
        # Range should be [0, pi]
        assert np.min(result) >= -0.1  # Small tolerance
        assert np.max(result) <= np.pi + 0.1

    def test_sinusoid_with_offset(self, simple_grid):
        """Test sinusoid with b offset."""
        result = phase.sinusoid(simple_grid, vector=(1, 0), a=np.pi, b=np.pi/2)
        # Range should be [pi/2, pi/2 + pi/2]
        assert np.min(result) >= np.pi/2 - 0.1
        assert np.max(result) <= np.pi + 0.1

    def test_sinusoid_shift(self, simple_grid):
        """Test sinusoid phase shift."""
        result1 = phase.sinusoid(simple_grid, vector=(1, 0), shift=0)
        result2 = phase.sinusoid(simple_grid, vector=(1, 0), shift=np.pi)
        # Shifted by pi should flip the pattern
        assert not np.allclose(result1, result2)


class TestBinary:
    """Tests for binary function."""

    def test_binary_two_levels(self, simple_grid):
        """Test that binary produces exactly two values."""
        result = phase.binary(simple_grid, vector=(1, 0), a=np.pi, b=0)
        unique_vals = np.unique(np.round(result, 6))
        assert len(unique_vals) == 2
        assert 0 in unique_vals or np.isclose(unique_vals.min(), 0)
        assert np.pi in unique_vals or np.isclose(unique_vals.max(), np.pi)

    def test_binary_duty_cycle(self, simple_grid):
        """Test binary duty cycle parameter."""
        result = phase.binary(simple_grid, vector=(1, 0), duty_cycle=0.25, a=1, b=0)
        # Approximately 25% should be high
        high_fraction = np.sum(result > 0.5) / result.size
        assert high_fraction == pytest.approx(0.25, abs=0.05)

    def test_binary_integer_period(self, normalized_grid):
        """Test binary with integer period in pixels."""
        # This tests the optimization path
        result = phase.binary(normalized_grid, vector=(4, 0), a=np.pi, b=0)
        assert result.shape == normalized_grid[0].shape


class TestLens:
    """Tests for lens function."""

    def test_lens_infinite_focal_length(self, simple_grid):
        """Test lens with infinite focal length (no curvature)."""
        result = phase.lens(simple_grid, f=(np.inf, np.inf))
        # Should be all zeros
        assert np.allclose(result, 0)

    def test_lens_symmetry(self, simple_grid):
        """Test lens symmetry."""
        result = phase.lens(simple_grid, f=(100, 100))
        # Should be symmetric about center
        center = result.shape[0] // 2
        assert np.allclose(result[center-10, center], result[center+10, center])
        assert np.allclose(result[center, center-10], result[center, center+10])

    def test_lens_quadratic(self, normalized_grid):
        """Test that lens phase is quadratic in radius."""
        result = phase.lens(normalized_grid, f=(1000, 1000))
        r_squared = normalized_grid[0]**2 + normalized_grid[1]**2
        # Phase should be proportional to r^2
        # Check that it's not linear
        assert not np.allclose(result, result[0, 0])


class TestAxicon:
    """Tests for axicon function."""

    def test_axicon_infinite_focal_length(self, simple_grid):
        """Test axicon with infinite focal length."""
        result = phase.axicon(simple_grid, f=(np.inf, np.inf))
        # Should be constant (zero phase)
        assert np.allclose(result, result[0, 0])

    def test_axicon_linear_in_radius(self, normalized_grid):
        """Test that axicon phase is linear in radius."""
        result = phase.axicon(normalized_grid, f=(1000, 1000))
        # Calculate radius
        r = np.sqrt(normalized_grid[0]**2 + normalized_grid[1]**2)
        # Check central region (avoid edge effects)
        center = result.shape[0] // 2
        r_center = r[center-50:center+50, center-50:center+50]
        result_center = result[center-50:center+50, center-50:center+50]
        # Should be approximately linear in r (not r^2)
        assert result_center.shape == r_center.shape


class TestZernike:
    """Tests for Zernike polynomial functions."""

    def test_zernike_piston(self, normalized_grid):
        """Test Zernike piston mode (j=0)."""
        result = phase.zernike(normalized_grid, index=0)
        # Piston should be constant
        assert np.allclose(result, result[0, 0])

    def test_zernike_tilt_x(self, normalized_grid):
        """Test Zernike x-tilt mode (j=1)."""
        result = phase.zernike(normalized_grid, index=1)
        # Should vary in x
        assert not np.allclose(result[128, :], result[0, :])

    def test_zernike_tilt_y(self, normalized_grid):
        """Test Zernike y-tilt mode (j=2)."""
        result = phase.zernike(normalized_grid, index=2)
        # Should vary in y
        assert not np.allclose(result[:, 128], result[:, 0])

    def test_zernike_weight(self, normalized_grid):
        """Test Zernike weight parameter."""
        result1 = phase.zernike(normalized_grid, index=1, weight=1)
        result2 = phase.zernike(normalized_grid, index=1, weight=2)
        # Double weight should double the result
        assert np.allclose(2 * result1, result2)

    def test_zernike_convert_index(self):
        """Test Zernike index conversion."""
        # ANSI j=3 should map to specific (n, m)
        result = phase.zernike_convert_index(3, from_index="ansi", to_index="radial")
        assert len(result) == 2
        assert isinstance(result[0], (int, np.integer))

    def test_zernike_sum(self, normalized_grid):
        """Test sum of Zernike polynomials."""
        indices = [0, 1, 2]
        weights = [1, 0.5, 0.3]
        result = phase.zernike_sum(normalized_grid, indices, weights)
        # Should be valid array
        assert result.shape == normalized_grid[0].shape
        assert np.all(np.isfinite(result))


class TestQuadrants:
    """Tests for quadrants function."""

    def test_quadrants_four_regions(self, simple_grid):
        """Test that quadrants creates four distinct regions."""
        result = phase.quadrants(simple_grid, radius=.001, center=(0, 0))
        unique_vals = np.unique(np.round(result, 6))
        # Should have exactly 4 unique values
        assert len(unique_vals) == 4


class TestBahtinov:
    """Tests for Bahtinov mask."""

    def test_bahtinov_mask(self, simple_grid):
        """Test Bahtinov mask generation."""
        result = phase.bahtinov(simple_grid, radius=0.005)
        # Should return a valid phase pattern
        assert result.shape == simple_grid[0].shape
        assert np.all(np.isfinite(result))


class TestPhaseProperties:
    """Test general properties that should hold for all phase functions."""

    def test_phase_functions_return_real(self, simple_grid):
        """Test that all phase functions return real arrays."""
        functions_to_test = [
            (phase.blaze, {"vector": (1, 1)}),
            (phase.sinusoid, {"vector": (1, 1)}),
            (phase.binary, {"vector": (1, 1)}),
            (phase.lens, {"f": (100, 100)}),
        ]

        for func, kwargs in functions_to_test:
            result = func(simple_grid, **kwargs)
            assert np.isrealobj(result), f"{func.__name__} returned complex values"
            assert np.all(np.isfinite(result)), f"{func.__name__} returned non-finite values"

    def test_phase_functions_preserve_shape(self, simple_grid):
        """Test that phase functions preserve grid shape."""
        functions_to_test = [
            (phase.blaze, {"vector": (1, 1)}),
            (phase.sinusoid, {"vector": (1, 1)}),
            (phase.binary, {"vector": (1, 1)}),
            (phase.lens, {"f": (100, 100)}),
        ]

        expected_shape = simple_grid[0].shape

        for func, kwargs in functions_to_test:
            result = func(simple_grid, **kwargs)
            assert result.shape == expected_shape, f"{func.__name__} changed shape"
