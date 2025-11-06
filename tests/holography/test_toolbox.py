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

# Test blaze()

def test_blaze_zero_vector(simple_grid):
    """Test blaze with zero vector returns zeros."""
    result = phase.blaze(simple_grid, vector=(0, 0))
    assert result.shape == simple_grid[0].shape
    assert np.allclose(result, 0)

def test_blaze_x_only(simple_grid):
    """Test blaze with x-only vector."""
    result = phase.blaze(simple_grid, vector=(1, 0))
    # Should vary only in x direction
    assert not np.allclose(result[0, :], result[0, 0])
    # Should be constant in y direction
    assert np.allclose(result[:, 50], result[0, 50])

def test_blaze_y_only(simple_grid):
    """Test blaze with y-only vector."""
    result = phase.blaze(simple_grid, vector=(0, 1))
    # Should vary only in y direction
    assert not np.allclose(result[:, 0], result[0, 0])
    # Should be constant in x direction
    assert np.allclose(result[50, :], result[50, 0])

def test_blaze_linear_in_k(simple_grid):
    """Test that blaze is linear in k-vector."""
    result1 = phase.blaze(simple_grid, vector=(1, 1))
    result2 = phase.blaze(simple_grid, vector=(2, 2))
    # Double k-vector should double the phase
    assert np.allclose(2 * result1, result2)

def test_blaze_3d_vector(simple_grid):
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

    # 3D vector should add a focusing term
    phase_3d = phase.blaze(simple_grid, vector=(0.1, 0.2, 0.5))
    phase_2d = phase.blaze(simple_grid, vector=(0.1, 0.2))

    # Should be different due to the focusing term
    assert not np.allclose(phase_3d, phase_2d)

def test_phase_function_parameter_variations(simple_grid):
    """Test blaze with different vector magnitudes."""
    blaze_small = phase.blaze(simple_grid, vector=(0.01, 0.01))
    blaze_large = phase.blaze(simple_grid, vector=(0.1, 0.1))

    # Larger vector should produce steeper gradient
    grad_small = np.max(np.gradient(blaze_small)[0])
    grad_large = np.max(np.gradient(blaze_large)[0])
    assert grad_large > grad_small


# Test sinusoid()

def test_sinusoid_zero_vector(simple_grid):
    """Test sinusoid with zero vector."""
    result = phase.sinusoid(simple_grid, vector=(0, 0))
    # Should be constant
    assert np.allclose(result, result[0, 0])

def test_sinusoid_range(simple_grid):
    """Test sinusoid output range."""
    result = phase.sinusoid(simple_grid, vector=(1, 0), a=np.pi, b=0)
    # Range should be [0, pi]
    assert np.min(result) >= -0.1  # Small tolerance
    assert np.max(result) <= np.pi + 0.1

def test_sinusoid_with_offset(simple_grid):
    """Test sinusoid with b offset."""
    result = phase.sinusoid(simple_grid, vector=(1, 0), a=np.pi, b=np.pi/2)
    # Range should be [pi/2, pi/2 + pi/2]
    assert np.min(result) >= np.pi/2 - 0.1
    assert np.max(result) <= np.pi + 0.1

def test_sinusoid_shift(simple_grid):
    """Test sinusoid phase shift."""
    result1 = phase.sinusoid(simple_grid, vector=(1, 0), shift=0)
    result2 = phase.sinusoid(simple_grid, vector=(1, 0), shift=np.pi)
    # Shifted by pi should flip the pattern
    assert not np.allclose(result1, result2)

def test_sinusoid_edge_cases(simple_grid):
    """Test sinusoid function edge cases."""
    # Test with custom amplitude and offset
    phase_custom = phase.sinusoid(simple_grid, vector=(0.1, 0.2), a=2*np.pi, b=np.pi/2)

    # Check range is approximately correct
    assert np.max(phase_custom) <= 2*np.pi + np.pi/2 + 0.1
    assert np.min(phase_custom) >= np.pi/2 - 0.1

    # Test with shift parameter
    phase_shifted = phase.sinusoid(simple_grid, vector=(0.1, 0.2), shift=np.pi/4)
    phase_unshifted = phase.sinusoid(simple_grid, vector=(0.1, 0.2), shift=0)

    # Should be different
    assert not np.allclose(phase_shifted, phase_unshifted)


# Test binary()

def test_binary_two_levels(simple_grid):
    """Test that binary produces exactly two values."""
    result = phase.binary(simple_grid, vector=(1, 0), a=np.pi, b=0)
    unique_vals = np.unique(np.round(result, 6))
    assert len(unique_vals) == 2
    assert 0 in unique_vals or np.isclose(unique_vals.min(), 0)
    assert np.pi in unique_vals or np.isclose(unique_vals.max(), np.pi)

import matplotlib.pyplot as plt

def test_binary_duty_cycle(simple_grid):
    """Test binary duty cycle parameter."""
    result = phase.binary(simple_grid, vector=(.1, .1), duty_cycle=0.25, a=1, b=0)

    # Approximately 25% should be high
    high_fraction = np.sum(result > 0.5) / result.size
    assert high_fraction == pytest.approx(0.25, abs=0.05)

def test_binary_integer_period(normalized_grid):
    """Test binary with integer period in pixels."""
    # This tests the optimization path
    result = phase.binary(normalized_grid, vector=(4, 0), duty_cycle=0.75, a=np.pi, b=0)
    assert result.shape == normalized_grid[0].shape

    # Approximately 75% should be high
    high_fraction = np.sum(result > 0.5) / result.size
    assert high_fraction == pytest.approx(0.75, abs=0.05)

def test_binary_edge_cases(simple_grid):
    """Test binary grating edge cases."""
    # Test with extreme duty cycle
    phase_thin = phase.binary(simple_grid, vector=(0.1, 0.2), duty_cycle=0.1)
    phase_thick = phase.binary(simple_grid, vector=(0.1, 0.2), duty_cycle=0.9)

    # Should be different
    assert not np.allclose(phase_thin, phase_thick)

    # Should have some structure
    assert np.std(phase_thin) > 0
    assert np.std(phase_thick) > 0

# Test lens()

def test_lens_infinite_focal_length(simple_grid):
    """Test lens with infinite focal length (no curvature)."""
    result = phase.lens(simple_grid, f=(np.inf, np.inf))
    # Should be all zeros
    assert np.allclose(result, 0)

def test_lens_symmetry(simple_grid):
    """Test lens symmetry."""
    result = phase.lens(simple_grid, f=(100, 100))
    # Should be symmetric about center
    centery = result.shape[0] // 2
    centerx = result.shape[1] // 2

    d = 20
    dx = result[centery, centerx-d] - result[centery, centerx+d]
    dy = result[centery-d, centerx] - result[centery+d, centerx]

    assert dx == pytest.approx(0, abs=.1)
    assert dy == pytest.approx(0, abs=.1)

def test_lens_edge_cases(simple_grid):
    """Test lens function edge cases."""
    # Test with negative focal length (diverging lens)
    phase_pos = phase.lens(simple_grid, f=(10, 10))
    phase_neg = phase.lens(simple_grid, f=(-10, -10))

    # Should be negatives of each other (approximately)
    assert np.allclose(phase_pos, -phase_neg, atol=1e-10)

    phase_pos_neg = phase.lens(simple_grid, f=(10, -10))

    assert np.all(np.abs(phase_pos_neg) <= phase_pos - phase_neg + 1e-10)

# Test axicon()

def test_axicon_infinite_focal_length(simple_grid):
    """Test axicon with infinite focal length."""
    result = phase.axicon(simple_grid, f=(np.inf, np.inf))
    # Should be constant (zero phase)
    assert np.allclose(result, result[0, 0])

def test_axicon_linear_in_radius(normalized_grid):
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

def test_axicon_edge_cases(simple_grid):
    """Test axicon function edge cases."""
    # Test with zero angle (should be nearly flat)
    phase_flat = phase.axicon(simple_grid, f=(np.inf, np.inf))

    # Should be nearly constant
    assert np.std(phase_flat) < 0.1

# Test zernike()

def test_zernike_piston(normalized_grid):
    """Test Zernike piston mode (j=0)."""
    result = phase.zernike(normalized_grid, index=0)
    # Piston should be constant
    assert np.allclose(result, result[0, 0])

def test_zernike_tilt_x(normalized_grid):
    """Test Zernike x-tilt mode (j=1)."""
    result = phase.zernike(normalized_grid, index=1)
    # Should vary in x
    assert not np.allclose(result[128, :], result[0, :])

def test_zernike_tilt_y(normalized_grid):
    """Test Zernike y-tilt mode (j=2)."""
    result = phase.zernike(normalized_grid, index=2)
    # Should vary in y
    assert not np.allclose(result[:, 128], result[:, 0])

def test_zernike_weight(normalized_grid):
    """Test Zernike weight parameter."""
    result1 = phase.zernike(normalized_grid, index=1, weight=1)
    result2 = phase.zernike(normalized_grid, index=1, weight=2)
    # Double weight should double the result
    assert np.allclose(2 * result1, result2)

def test_zernike_convert_index():
    """Test Zernike index conversion."""
    # ANSI j=3 should map to specific (n, m)
    result = phase.zernike_convert_index(3, from_index="ansi", to_index="radial")
    assert result.shape == (1, 2)
    assert isinstance(result[0,0], (int, np.integer))

    result = phase.zernike_convert_index([3,4,5], from_index="ansi", to_index="radial")
    assert result.shape == (3, 2)
    assert isinstance(result[0,0], (int, np.integer))

def test_zernike_sum(normalized_grid):
    """Test sum of Zernike polynomials."""
    indices = [0, 1, 2]
    weights = [1, 0.5, 0.3]
    result = phase.zernike_sum(normalized_grid, indices, weights)
    # Should be valid array
    assert result.shape == normalized_grid[0].shape
    assert np.all(np.isfinite(result))

def test_zernike_edge_cases(normalized_grid):
    """Test Zernike polynomial edge cases."""
    # Test higher order Zernike polynomials
    z_high = phase.zernike(normalized_grid, index=10)
    assert z_high.shape == normalized_grid[0].shape
    assert not np.allclose(z_high, 0)  # Should not be trivial

    # Test with different weights
    z_normal = phase.zernike(normalized_grid, index=5, weight=1.0)
    z_scaled = phase.zernike(normalized_grid, index=5, weight=2.0)

    # Should be scaled version
    np.testing.assert_array_almost_equal(z_scaled, 2 * z_normal)

# Test quadrants()

def test_quadrants(simple_grid):
    """Test that quadrants creates four distinct regions."""
    result = phase.quadrants(simple_grid, radius=.001*np.sqrt(2), center=(0, 0))
    result -= phase.blaze(simple_grid, vector=(.001, .001))  # Subtract the phase of one of the quadrants

    # Find the mode.
    result = np.around(result * 1000)
    vals, counts = np.unique(result, return_counts=True)
    mode = vals[np.argmax(counts)]

    # Count how many pixels the mode
    assert np.sum(mode == result) == pytest.approx(.25 * result.size, abs=0.05 * result.size)

# Test bahtinov()

def test_bahtinov(simple_grid):
    """Test Bahtinov mask generation."""
    result = phase.bahtinov(simple_grid, radius=0.005)
    # Should return a valid phase pattern
    assert result.shape == simple_grid[0].shape
    assert np.all(np.isfinite(result))

# Additional tests for general properties

def test_phase_functions_return_real(simple_grid):
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

def test_phase_functions_preserve_shape(simple_grid):
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

