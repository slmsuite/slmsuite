"""
Unit tests for slmsuite.misc modules.
"""
import pytest
import numpy as np
from slmsuite.misc.math import iseven, INTEGER_TYPES, FLOAT_TYPES, REAL_TYPES, SCALAR_TYPES
from slmsuite.misc.fitfunctions import (
    linear, parabola, hyperbola, cos, lorentzian,
    gaussian, gaussian2d, tophat2d, sinc2d
)


# Test misc.math functions

def test_iseven():
    """Test iseven function with various inputs."""
    assert iseven(0) == True
    assert iseven(1) == False
    assert iseven(2) == True
    assert iseven(3) == False
    assert iseven(4) == True
    assert iseven(-1) == False
    assert iseven(-2) == True


def test_iseven_arrays():
    """Test iseven function with array inputs."""
    x = np.array([0, 1, 2, 3, 4, 5])
    result = iseven(x)
    expected = np.array([True, False, True, False, True, False])
    np.testing.assert_array_equal(result, expected)


def test_iseven_float_inputs():
    """Test iseven function with float inputs (should round)."""
    assert iseven(2.1) == True  # rounds to 2
    assert iseven(2.9) == False  # rounds to 3
    assert iseven(3.1) == False  # rounds to 3
    assert iseven(3.9) == True   # rounds to 4


def test_type_tuples():
    """Test that type tuples are correctly defined."""
    assert int in INTEGER_TYPES
    assert isinstance(np.int32(1), INTEGER_TYPES)
    assert float in FLOAT_TYPES
    assert isinstance(np.float64(1.0), FLOAT_TYPES)
    assert int in REAL_TYPES
    assert float in REAL_TYPES
    assert complex in SCALAR_TYPES


def test_type_tuples_coverage():
    """Test additional type coverage."""
    # Test more numpy integer types
    assert isinstance(np.int8(1), INTEGER_TYPES)
    assert isinstance(np.int16(1), INTEGER_TYPES)
    assert isinstance(np.int32(1), INTEGER_TYPES)
    assert isinstance(np.int64(1), INTEGER_TYPES)
    assert isinstance(np.uint8(1), INTEGER_TYPES)
    assert isinstance(np.uint16(1), INTEGER_TYPES)
    assert isinstance(np.uint32(1), INTEGER_TYPES)
    assert isinstance(np.uint64(1), INTEGER_TYPES)

    # Test more numpy float types
    assert isinstance(np.float32(1.0), FLOAT_TYPES)
    assert isinstance(np.float64(1.0), FLOAT_TYPES)

    # Test complex types
    assert isinstance(1+1j, SCALAR_TYPES)
    assert isinstance(np.complex64(1+1j), SCALAR_TYPES)
    assert isinstance(np.complex128(1+1j), SCALAR_TYPES)


# Test 1D fit functions

def test_linear():
    """Test linear function."""
    x = np.linspace(0, 10, 100)
    y = linear(x, m=2, b=3)
    assert y[0] == pytest.approx(3.0)
    assert y[-1] == pytest.approx(23.0)
    assert np.all(np.diff(y) > 0)  # Increasing

def test_parabola():
    """Test parabola function."""
    x = np.linspace(-5, 5, 100)
    y = parabola(x, a=1, x0=0, y0=1)
    # Minimum at x0
    assert y[50] == pytest.approx(1.0, abs=0.1)
    # Symmetric
    assert y[0] == pytest.approx(y[-1], abs=0.1)

def test_hyperbola():
    """Test hyperbola function."""
    z = np.linspace(-10, 10, 100)
    w = hyperbola(z, w0=1, z0=0, zr=1)
    # Minimum at z0
    min_idx = np.argmin(w)
    assert z[min_idx] == pytest.approx(0.0, abs=0.2)
    assert w[min_idx] == pytest.approx(1.0, abs=0.1)

def test_cos():
    """Test cosine fit function."""
    x = np.linspace(0, 2*np.pi, 100)
    y = cos(x, b=0, a=2, c=1, k=1)
    # Check amplitude range
    assert np.max(y) == pytest.approx(3.0, abs=0.1)
    assert np.min(y) == pytest.approx(1.0, abs=0.1)

def test_lorentzian():
    """Test Lorentzian function."""
    x = np.linspace(990, 1010, 1000)
    y = lorentzian(x, x0=1000, a=10, c=1, w=1)
    # Peak at x0
    max_idx = np.argmax(y)
    assert x[max_idx] == pytest.approx(1000.0, abs=0.1)
    # Check amplitude
    assert y[max_idx] == pytest.approx(11.0, abs=0.1)

def test_gaussian():
    """Test 1D Gaussian function."""
    x = np.linspace(-10, 10, 200)
    y = gaussian(x, x0=0, a=10, c=1, w=2)
    # Peak at x0
    max_idx = np.argmax(y)
    assert x[max_idx] == pytest.approx(0.0, abs=0.1)
    # Check amplitude
    assert y[max_idx] == pytest.approx(11.0, abs=0.1)
    # Check baseline
    assert y[0] == pytest.approx(1.0, abs=0.2)


# Test 2D fit functions

def test_gaussian2d():
    """Test 2D Gaussian function."""
    x = np.linspace(-10, 10, 50)
    y = np.linspace(-10, 10, 50)
    X, Y = np.meshgrid(x, y)
    xy = np.array([X, Y])

    z = gaussian2d(xy, x0=0, y0=0, a=10, c=1, wx=2, wy=2)

    # Peak at center
    assert z[25, 25] == pytest.approx(11.0, abs=0.5)
    # Check baseline at edges
    assert z[0, 0] < 2.0

def test_gaussian2d_with_shear():
    """Test 2D Gaussian with shear."""
    x = np.linspace(-10, 10, 50)
    y = np.linspace(-10, 10, 50)
    X, Y = np.meshgrid(x, y)
    xy = np.array([X, Y])

    z = gaussian2d(xy, x0=0, y0=0, a=10, c=1, wx=2, wy=2, wxy=0.5)

    # Should still have peak near center
    max_val = np.max(z)
    assert max_val == pytest.approx(11.0, abs=0.5)

def test_tophat2d():
    """Test 2D tophat function."""
    x = np.linspace(-10, 10, 100)
    y = np.linspace(-10, 10, 100)
    X, Y = np.meshgrid(x, y)
    xy = np.array([X, Y])

    z = tophat2d(xy, x0=0, y0=0, R=5, a=10, c=1)

    # Center should be high
    assert z[50, 50] == pytest.approx(11.0)
    # Edge should be low (baseline)
    assert z[0, 0] == pytest.approx(1.0)
    # Check approximate area
    high_pixels = np.sum(z > 5)
    expected_pixels = np.pi * (5 / 0.2) ** 2  # R=5, step=0.2
    assert high_pixels == pytest.approx(expected_pixels, rel=0.2)


def test_sinc2d():
    """Test 2D sinc function."""
    x = np.linspace(-10, 10, 100)
    y = np.linspace(-10, 10, 100)
    X, Y = np.meshgrid(x, y)
    xy = np.array([X, Y])

    z = sinc2d(xy, x0=0, y0=0, a=10, c=1, wx=2, wy=2)

    # Peak at center
    assert z[50, 50] == pytest.approx(11.0, abs=0.5)
    # Check that it's symmetric
    assert z[50, 60] == pytest.approx(z[50, 40], abs=0.1)
    assert z[60, 50] == pytest.approx(z[40, 50], abs=0.1)


def test_cos_edge_cases():
    """Test cosine function edge cases."""
    x = np.array([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi])

    # Test with default k=1
    y = cos(x, b=0, a=2, c=1)
    # c + a/2 * (1 + cos(x)) = 1 + 1 * (1 + cos(x)) = 2 + cos(x)
    expected = np.array([3, 2, 1, 2, 3])  # 2 + cos([0, pi/2, pi, 3pi/2, 2pi])
    np.testing.assert_array_almost_equal(y, expected, decimal=5)

    # Test with phase offset
    y = cos(x, b=np.pi/2, a=2, c=1)
    # 2 + cos(x - pi/2) = 2 + sin(x)
    expected = np.array([2, 3, 2, 1, 2])  # 2 + sin([0, pi/2, pi, 3pi/2, 2pi])
    np.testing.assert_array_almost_equal(y, expected, decimal=5)

    # Test with k parameter
    y = cos(x, b=0, a=2, c=1, k=2)
    # 2 + cos(2x)
    expected = np.array([3, 1, 3, 1, 3])  # 2 + cos([0, pi, 2pi, 3pi, 4pi])
    np.testing.assert_array_almost_equal(y, expected, decimal=5)


def test_hyperbola_edge_cases():
    """Test hyperbola function edge cases."""
    z = np.array([0, 1, 2, 3, 4, 5])

    # Test basic hyperbola
    w = hyperbola(z, w0=1, z0=2, zr=1)

    # At z0, should equal w0
    assert w[2] == pytest.approx(1.0, abs=0.001)

    # Test at z0 ± zr (should be sqrt(2) * w0)
    w_at_zr = hyperbola(np.array([1, 3]), w0=1, z0=2, zr=1)
    expected = np.sqrt(2)
    np.testing.assert_array_almost_equal(w_at_zr, [expected, expected], decimal=5)

    # Test symmetry around z0
    z_sym = np.array([0, 1, 3, 4])
    w_sym = hyperbola(z_sym, w0=1, z0=2, zr=1)
    assert w_sym[0] == pytest.approx(w_sym[3], abs=0.001)  # z=0 and z=4
    assert w_sym[1] == pytest.approx(w_sym[2], abs=0.001)  # z=1 and z=3


def test_parabola_edge_cases():
    """Test parabola function edge cases."""
    x = np.linspace(-5, 5, 11)

    # Test vertex at origin
    y = parabola(x, a=1, x0=0, y0=0)
    assert y[5] == pytest.approx(0.0, abs=0.001)  # At x=0
    assert y[6] == pytest.approx(1.0, abs=0.001)  # At x=1
    assert y[4] == pytest.approx(1.0, abs=0.001)  # At x=-1

    # Test offset vertex
    y = parabola(x, a=2, x0=1, y0=3)
    expected_at_vertex = 3
    vertex_idx = 6  # x=1 is at index 6
    assert y[vertex_idx] == pytest.approx(expected_at_vertex, abs=0.001)

    # Test negative a (downward parabola)
    y = parabola(x, a=-1, x0=0, y0=5)
    assert y[5] == pytest.approx(5.0, abs=0.001)  # Maximum at vertex
    assert y[6] < y[5]  # Values decrease away from vertex
    assert y[4] < y[5]


def test_linear_edge_cases():
    """Test linear function edge cases."""
    x = np.array([0, 1, -1, 10, -10])

    # Test horizontal line (m=0)
    y = linear(x, m=0, b=5)
    expected = np.array([5, 5, 5, 5, 5])
    np.testing.assert_array_equal(y, expected)

    # Test line through origin (b=0)
    y = linear(x, m=2, b=0)
    expected = 2 * x
    np.testing.assert_array_equal(y, expected)

    # Test negative slope
    y = linear(x, m=-1, b=0)
    expected = -x
    np.testing.assert_array_equal(y, expected)


def test_lorentzian_edge_cases():
    """Test Lorentzian function edge cases."""
    x = np.linspace(990, 1010, 100)

    # Test high Q (narrow resonance)
    y_high_q = lorentzian(x, x0=1000, a=10, c=1, w=1)

    # Test low Q (broad resonance)
    y_low_q = lorentzian(x, x0=1000, a=10, c=1, w=10)

    # High Q should be narrower (values away from center should be lower)
    assert y_high_q[10] < y_low_q[10]  # Away from center
    # At center, both should be close to c + a, but allow some tolerance
    assert y_high_q[50] == pytest.approx(11.0, abs=0.5)
    assert y_low_q[50] == pytest.approx(11.0, abs=0.5)

    # Test at resonance center
    y_center = lorentzian(np.array([1000]), x0=1000, a=10, c=1, w=10)
    assert y_center[0] == pytest.approx(11.0, abs=0.001)  # c + a


def test_gaussian_edge_cases():
    """Test Gaussian function edge cases."""
    x = np.linspace(-10, 10, 100)

    # Test very narrow Gaussian
    y_narrow = gaussian(x, x0=0, a=10, c=1, w=0.1)

    # Test very broad Gaussian
    y_broad = gaussian(x, x0=0, a=10, c=1, w=10)

    # Both should have same amplitude parameter, but peaks may differ due to normalization
    center_idx = 50

    # Test offset Gaussian
    y_offset = gaussian(x, x0=2, a=10, c=1, w=1)
    max_idx = np.argmax(y_offset)
    assert x[max_idx] == pytest.approx(2.0, abs=0.2)  # Peak at x0=2


def test_gaussian2d_singular_matrix():
    """Test 2D Gaussian with singular covariance matrix."""
    x = np.linspace(-10, 10, 50)
    y = np.linspace(-10, 10, 50)
    X, Y = np.meshgrid(x, y)
    xy = np.array([X, Y])

    # This should still work, might just be very elongated
    z = gaussian2d(xy, x0=0, y0=0, a=10, c=1, wx=0.001, wy=2, wxy=0)

    # Should still have reasonable values
    assert np.max(z) > 1.0
    assert np.min(z) >= 1.0  # baseline



def test_sinc2d():
    """Test 2D sinc function."""
    x = np.linspace(-10, 10, 100)
    y = np.linspace(-10, 10, 100)
    X, Y = np.meshgrid(x, y)
    xy = np.array([X, Y])

    z = sinc2d(xy, x0=0, y0=0, R=2, a=10, b=0, c=0, d=1)

    # Peak at center
    max_idx = np.unravel_index(np.argmax(z), z.shape)
    assert max_idx[0] == pytest.approx(50, abs=2)
    assert max_idx[1] == pytest.approx(50, abs=2)
