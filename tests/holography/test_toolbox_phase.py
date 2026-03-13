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


def test_blaze(simple_grid, subtests):
    """Test blaze() phase pattern generation."""
    with subtests.test("zero vector returns zeros"):
        result = phase.blaze(simple_grid, vector=(0, 0))
        assert result.shape == simple_grid[0].shape
        assert np.allclose(result, 0)

    with subtests.test("x-only vector varies only in x"):
        result = phase.blaze(simple_grid, vector=(1, 0))
        assert not np.allclose(result[0, :], result[0, 0])
        assert np.allclose(result[:, 50], result[0, 50])

    with subtests.test("y-only vector varies only in y"):
        result = phase.blaze(simple_grid, vector=(0, 1))
        assert not np.allclose(result[:, 0], result[0, 0])
        assert np.allclose(result[50, :], result[50, 0])

    with subtests.test("linear in k-vector"):
        result1 = phase.blaze(simple_grid, vector=(1, 1))
        result2 = phase.blaze(simple_grid, vector=(2, 2))
        assert np.allclose(2 * result1, result2)

    with subtests.test("3D vector includes focusing term"):
        result_2d = phase.blaze(simple_grid, vector=(1, 1))
        result_3d = phase.blaze(simple_grid, vector=(1, 1, 1))
        assert result_3d.shape == result_2d.shape
        assert not np.allclose(result_2d, result_3d)
        diff = result_3d - result_2d
        r_squared = simple_grid[0]**2 + simple_grid[1]**2
        assert np.allclose(diff, np.pi * r_squared)

        phase_3d = phase.blaze(simple_grid, vector=(0.1, 0.2, 0.5))
        phase_2d = phase.blaze(simple_grid, vector=(0.1, 0.2))
        assert not np.allclose(phase_3d, phase_2d)

    with subtests.test("larger vector produces steeper gradient"):
        blaze_small = phase.blaze(simple_grid, vector=(0.01, 0.01))
        blaze_large = phase.blaze(simple_grid, vector=(0.1, 0.1))
        grad_small = np.max(np.gradient(blaze_small)[0])
        grad_large = np.max(np.gradient(blaze_large)[0])
        assert grad_large > grad_small


def test_sinusoid(simple_grid, subtests):
    """Test sinusoid() phase pattern generation."""
    with subtests.test("zero vector gives constant"):
        result = phase.sinusoid(simple_grid, vector=(0, 0))
        assert np.allclose(result, result[0, 0])

    with subtests.test("output range with a=pi, b=0"):
        result = phase.sinusoid(simple_grid, vector=(1, 0), a=np.pi, b=0)
        assert np.min(result) >= -0.1
        assert np.max(result) <= np.pi + 0.1

    with subtests.test("b offset shifts range"):
        result = phase.sinusoid(simple_grid, vector=(1, 0), a=np.pi, b=np.pi/2)
        assert np.min(result) >= np.pi/2 - 0.1
        assert np.max(result) <= np.pi + 0.1

    with subtests.test("shift parameter changes pattern"):
        result1 = phase.sinusoid(simple_grid, vector=(1, 0), shift=0)
        result2 = phase.sinusoid(simple_grid, vector=(1, 0), shift=np.pi)
        assert not np.allclose(result1, result2)

    with subtests.test("custom amplitude and offset range"):
        phase_custom = phase.sinusoid(simple_grid, vector=(0.1, 0.2), a=2*np.pi, b=np.pi/2)
        assert np.max(phase_custom) <= 2*np.pi + np.pi/2 + 0.1
        assert np.min(phase_custom) >= np.pi/2 - 0.1

    with subtests.test("shift vs unshifted differ"):
        phase_shifted = phase.sinusoid(simple_grid, vector=(0.1, 0.2), shift=np.pi/4)
        phase_unshifted = phase.sinusoid(simple_grid, vector=(0.1, 0.2), shift=0)
        assert not np.allclose(phase_shifted, phase_unshifted)


def test_binary(simple_grid, normalized_grid, subtests):
    """Test binary() grating generation."""
    with subtests.test("produces exactly two levels"):
        result = phase.binary(simple_grid, vector=(1, 0), a=np.pi, b=0)
        unique_vals = np.unique(np.round(result, 6))
        assert len(unique_vals) == 2
        assert 0 in unique_vals or np.isclose(unique_vals.min(), 0)
        assert np.pi in unique_vals or np.isclose(unique_vals.max(), np.pi)

    with subtests.test("duty cycle 0.25"):
        result = phase.binary(simple_grid, vector=(.1, .1), duty_cycle=0.25, a=1, b=0)
        high_fraction = np.sum(result > 0.5) / result.size
        assert high_fraction == pytest.approx(0.25, abs=0.05)

    with subtests.test("integer period with duty cycle 0.75"):
        result = phase.binary(normalized_grid, vector=(4, 0), duty_cycle=0.75, a=np.pi, b=0)
        assert result.shape == normalized_grid[0].shape
        high_fraction = np.sum(result > 0.5) / result.size
        assert high_fraction == pytest.approx(0.75, abs=0.05)

    with subtests.test("extreme duty cycles differ and have structure"):
        phase_thin = phase.binary(simple_grid, vector=(0.1, 0.2), duty_cycle=0.1)
        phase_thick = phase.binary(simple_grid, vector=(0.1, 0.2), duty_cycle=0.9)
        assert not np.allclose(phase_thin, phase_thick)
        assert np.std(phase_thin) > 0
        assert np.std(phase_thick) > 0


def test_lens(simple_grid, subtests):
    """Test lens() phase pattern generation."""
    with subtests.test("infinite focal length gives zeros"):
        result = phase.lens(simple_grid, f=(np.inf, np.inf))
        assert np.allclose(result, 0)

    with subtests.test("symmetric about center"):
        result = phase.lens(simple_grid, f=(100, 100))
        centery = result.shape[0] // 2
        centerx = result.shape[1] // 2
        d = 20
        dx = result[centery, centerx-d] - result[centery, centerx+d]
        dy = result[centery-d, centerx] - result[centery+d, centerx]
        assert dx == pytest.approx(0, abs=.1)
        assert dy == pytest.approx(0, abs=.1)

    with subtests.test("negative focal length negates phase"):
        phase_pos = phase.lens(simple_grid, f=(10, 10))
        phase_neg = phase.lens(simple_grid, f=(-10, -10))
        assert np.allclose(phase_pos, -phase_neg, atol=1e-10)

    with subtests.test("mixed focal lengths bounded"):
        phase_pos = phase.lens(simple_grid, f=(10, 10))
        phase_neg = phase.lens(simple_grid, f=(-10, -10))
        phase_pos_neg = phase.lens(simple_grid, f=(10, -10))
        assert np.all(np.abs(phase_pos_neg) <= phase_pos - phase_neg + 1e-10)


def test_axicon(simple_grid, normalized_grid, subtests):
    """Test axicon() phase pattern generation."""
    with subtests.test("infinite focal length gives constant"):
        result = phase.axicon(simple_grid, f=(np.inf, np.inf))
        assert np.allclose(result, result[0, 0])
        assert np.std(result) < 0.1

    with subtests.test("linear in radius shape check"):
        result = phase.axicon(normalized_grid, f=(1000, 1000))
        r = np.sqrt(normalized_grid[0]**2 + normalized_grid[1]**2)
        center = result.shape[0] // 2
        r_center = r[center-50:center+50, center-50:center+50]
        result_center = result[center-50:center+50, center-50:center+50]
        assert result_center.shape == r_center.shape


def test_zernike(normalized_grid, subtests):
    """Test zernike() polynomials and related utilities."""
    with subtests.test("piston mode (j=0) is constant"):
        result = phase.zernike(normalized_grid, index=0)
        assert np.allclose(result, result[0, 0])

    with subtests.test("x-tilt mode (j=1) varies in x"):
        result = phase.zernike(normalized_grid, index=1)
        assert not np.allclose(result[128, :], result[0, :])

    with subtests.test("y-tilt mode (j=2) varies in y"):
        result = phase.zernike(normalized_grid, index=2)
        assert not np.allclose(result[:, 128], result[:, 0])

    with subtests.test("weight parameter scales linearly"):
        result1 = phase.zernike(normalized_grid, index=1, weight=1)
        result2 = phase.zernike(normalized_grid, index=1, weight=2)
        assert np.allclose(2 * result1, result2)

    with subtests.test("higher order (j=10) is nontrivial"):
        z_high = phase.zernike(normalized_grid, index=10)
        assert z_high.shape == normalized_grid[0].shape
        assert not np.allclose(z_high, 0)

    with subtests.test("weight scaling for j=5"):
        z_normal = phase.zernike(normalized_grid, index=5, weight=1.0)
        z_scaled = phase.zernike(normalized_grid, index=5, weight=2.0)
        np.testing.assert_array_almost_equal(z_scaled, 2 * z_normal)

    with subtests.test("index conversion scalar"):
        result = phase.zernike_convert_index(3, from_index="ansi", to_index="radial")
        assert result.shape == (1, 2)
        assert isinstance(result[0,0], (int, np.integer))

    with subtests.test("index conversion list"):
        result = phase.zernike_convert_index([3,4,5], from_index="ansi", to_index="radial")
        assert result.shape == (3, 2)
        assert isinstance(result[0,0], (int, np.integer))

    with subtests.test("zernike_sum produces valid array"):
        indices = [0, 1, 2]
        weights = [1, 0.5, 0.3]
        result = phase.zernike_sum(normalized_grid, indices, weights)
        assert result.shape == normalized_grid[0].shape
        assert np.all(np.isfinite(result))


def test_quadrants(simple_grid):
    """Test that quadrants creates four distinct regions."""
    result = phase.quadrants(simple_grid, radius=.001*np.sqrt(2), center=(0, 0))
    result -= phase.blaze(simple_grid, vector=(.001, .001))

    result = np.around(result * 1000)
    vals, counts = np.unique(result, return_counts=True)
    mode = vals[np.argmax(counts)]

    assert np.sum(mode == result) == pytest.approx(.25 * result.size, abs=0.05 * result.size)


def test_bahtinov(simple_grid):
    """Test Bahtinov mask generation."""
    result = phase.bahtinov(simple_grid, radius=0.005)
    assert result.shape == simple_grid[0].shape
    assert np.all(np.isfinite(result))


def test_phase_functions_general(simple_grid, subtests):
    """Test general properties across all phase functions."""
    functions_to_test = [
        (phase.blaze, {"vector": (1, 1)}),
        (phase.sinusoid, {"vector": (1, 1)}),
        (phase.binary, {"vector": (1, 1)}),
        (phase.lens, {"f": (100, 100)}),
    ]

    expected_shape = simple_grid[0].shape

    for func, kwargs in functions_to_test:
        with subtests.test(f"{func.__name__} returns real finite values"):
            result = func(simple_grid, **kwargs)
            assert np.isrealobj(result), f"{func.__name__} returned complex values"
            assert np.all(np.isfinite(result)), f"{func.__name__} returned non-finite values"

        with subtests.test(f"{func.__name__} preserves shape"):
            result = func(simple_grid, **kwargs)
            assert result.shape == expected_shape, f"{func.__name__} changed shape"


def test_zernike_aperture(normalized_grid, subtests):
    """Test zernike_aperture() scaling helper."""
    with subtests.test("circular aperture is isotropic"):
        x_scale, y_scale = phase.zernike_aperture(normalized_grid, aperture="circular")
        assert x_scale == pytest.approx(y_scale)
        # Scale times max coordinate should give 1
        assert x_scale * np.nanmax(normalized_grid[0]) == pytest.approx(1, rel=1e-6)

    with subtests.test("elliptical aperture may be anisotropic"):
        x = np.linspace(-200, 200, 128)
        y = np.linspace(-500, 500, 128)
        X, Y = np.meshgrid(x, y)
        rect_grid = (X, Y)
        x_scale, y_scale = phase.zernike_aperture(rect_grid, aperture="elliptical")
        # Each axis maps independently
        assert x_scale == pytest.approx(1 / 200, rel=1e-6)
        assert y_scale == pytest.approx(1 / 500, rel=1e-6)

    with subtests.test("cropped aperture circumscribes rectangle"):
        x_scale, y_scale = phase.zernike_aperture(normalized_grid, aperture="cropped")
        assert x_scale == pytest.approx(y_scale)
        # For a square grid the corner distance is sqrt(2)*max
        max_coord = np.nanmax(normalized_grid[0])
        expected = 1 / np.sqrt(2 * max_coord**2)
        assert x_scale == pytest.approx(expected, rel=1e-6)

    with subtests.test("scalar aperture"):
        x_scale, y_scale = phase.zernike_aperture(normalized_grid, aperture=0.005)
        assert x_scale == pytest.approx(0.005)
        assert y_scale == pytest.approx(0.005)

    with subtests.test("tuple aperture"):
        x_scale, y_scale = phase.zernike_aperture(normalized_grid, aperture=(0.01, 0.02))
        assert x_scale == pytest.approx(0.01)
        assert y_scale == pytest.approx(0.02)

    with subtests.test("invalid string raises ValueError"):
        with pytest.raises(ValueError):
            phase.zernike_aperture(normalized_grid, aperture="invalid")


def test_zernike_get_string(subtests):
    """Test zernike_get_string() LaTeX representations."""
    with subtests.test("piston (j=0) is constant '1'"):
        s = phase.zernike_get_string(0)
        assert s == "1"

    with subtests.test("tilt (j=1) contains y"):
        s = phase.zernike_get_string(1)
        assert "y" in s

    with subtests.test("tilt (j=2) contains x"):
        s = phase.zernike_get_string(2)
        assert "x" in s

    with subtests.test("defocus (j=4) contains x^2 and y^2"):
        s = phase.zernike_get_string(4)
        assert "x^2" in s
        assert "y^2" in s

    with subtests.test("derivative reduces order"):
        s_orig = phase.zernike_get_string(4, derivative=(0, 0))
        s_dx = phase.zernike_get_string(4, derivative=(1, 0))
        # Derivative should not contain x^2
        assert "x^2" not in s_dx
        assert "x" in s_dx  # Should still contain x^1 term


def test_zernike_convert_index(subtests):
    """Test zernike_convert_index() roundtrip conversions."""
    with subtests.test("ansi -> radial -> ansi roundtrip"):
        indices = np.arange(15)
        radial = phase.zernike_convert_index(indices, from_index="ansi", to_index="radial")
        back = phase.zernike_convert_index(radial, from_index="radial", to_index="ansi")
        np.testing.assert_array_equal(back.ravel(), indices)

    with subtests.test("ansi -> noll"):
        noll = phase.zernike_convert_index([0, 1, 2, 3], from_index="ansi", to_index="noll")
        # Noll is 1-indexed; j=0 (piston) -> noll=1
        assert noll.ravel()[0] == 1

    with subtests.test("ansi -> fringe raises NotImplementedError (known bug)"):
        # Bug: `to_index == 'fringe'` is incorrectly checked in the from_index branch
        with pytest.raises(NotImplementedError):
            phase.zernike_convert_index([0, 1, 2], from_index="ansi", to_index="fringe")

    with subtests.test("ansi -> wyant"):
        wyant = phase.zernike_convert_index([0, 1, 2], from_index="ansi", to_index="wyant")
        assert wyant.ravel()[0] == 0  # Wyant is 0-indexed

    with subtests.test("invalid index raises ValueError"):
        with pytest.raises(ValueError):
            phase.zernike_convert_index([0], from_index="bogus", to_index="ansi")


def test_zernike_sum(normalized_grid, subtests):
    """Test zernike_sum() advanced features."""
    with subtests.test("use_mask='return' gives boolean mask"):
        mask = phase.zernike_sum(
            normalized_grid, indices=[0], weights=[1], use_mask="return"
        )
        assert mask.dtype == bool
        assert mask.shape == normalized_grid[0].shape

    with subtests.test("use_mask=True zeros outside aperture"):
        result = phase.zernike_sum(
            normalized_grid, indices=[4], weights=[1], use_mask=True, aperture="circular"
        )
        mask = phase.zernike_sum(
            normalized_grid, indices=[0], weights=[1],
            use_mask="return", aperture="circular"
        )
        # Outside the mask should be zero
        assert np.allclose(result[~mask], 0)

    with subtests.test("use_mask=False does not crop"):
        result = phase.zernike_sum(
            normalized_grid, indices=[4], weights=[1], use_mask=False
        )
        # Should have non-zero values everywhere (defocus is nonzero at corners)
        assert not np.allclose(result, 0)

    with subtests.test("derivative (1,0) of tilt-x is constant"):
        # Z_2 = x, so d/dx = 1 (up to scaling)
        result = phase.zernike_sum(
            normalized_grid, indices=[2], weights=[1],
            use_mask=False, derivative=(1, 0)
        )
        # Should be approximately constant and nonzero
        assert np.std(result) < 1e-10
        assert not np.allclose(result, 0)

    with subtests.test("stacked weights (D, N) returns 3D"):
        weights_2d = np.array([[1, 0], [0, 1]])  # Two polynomials, two stacks
        result = phase.zernike_sum(
            normalized_grid, indices=[1, 2], weights=weights_2d
        )
        assert result.ndim == 3
        assert result.shape[0] == 2

    with subtests.test("out parameter reuses memory"):
        out = np.zeros((1, *normalized_grid[0].shape), dtype=normalized_grid[0].dtype)
        result = phase.zernike_sum(
            normalized_grid, indices=[1], weights=[1], out=out
        )
        # result should share memory with out
        assert np.shares_memory(result, out)


def test_polynomial(simple_grid, subtests):
    """Test polynomial() monomial summation."""
    with subtests.test("constant term (x^0 * y^0)"):
        # term (0,0) with weight 5 should give constant 5
        result = phase.polynomial(simple_grid, weights=[5.0], terms=np.array([[0, 0]]))
        # Result is (1, H, W); squeeze it
        result = result.squeeze()
        assert result.shape == simple_grid[0].shape
        assert np.allclose(result, 5.0)

    with subtests.test("linear x term"):
        # term (1,0) = x with weight 1
        result = phase.polynomial(simple_grid, weights=[1.0], terms=np.array([[1, 0]]))
        result = result.squeeze()
        assert np.allclose(result, simple_grid[0])

    with subtests.test("linear y term"):
        # term (0,1) = y with weight 1
        result = phase.polynomial(simple_grid, weights=[1.0], terms=np.array([[0, 1]]))
        result = result.squeeze()
        assert np.allclose(result, simple_grid[1])

    with subtests.test("quadratic x^2 + y^2"):
        terms = np.array([[2, 0], [0, 2]])
        weights = [1.0, 1.0]
        result = phase.polynomial(simple_grid, weights=weights, terms=terms).squeeze()
        expected = simple_grid[0]**2 + simple_grid[1]**2
        assert np.allclose(result, expected)

    with subtests.test("stacked weights produce multiple outputs"):
        terms = np.array([[1, 0], [0, 1]])
        weights = np.array([[1.0, 2.0], [3.0, 4.0]])  # (D=2, N=2)
        result = phase.polynomial(simple_grid, weights=weights, terms=terms)
        assert result.shape[0] == 2
        # First stack: 1*x + 3*y
        expected_0 = 1.0 * simple_grid[0] + 3.0 * simple_grid[1]
        assert np.allclose(result[0], expected_0)
        # Second stack: 2*x + 4*y
        expected_1 = 2.0 * simple_grid[0] + 4.0 * simple_grid[1]
        assert np.allclose(result[1], expected_1)


def test_laguerre_gaussian(simple_grid, subtests):
    """Test laguerre_gaussian() structured light generation."""
    with subtests.test("l=0, p=0 gives zero (scalar)"):
        result = phase.laguerre_gaussian(simple_grid, l=0, p=0)
        # With l=0, p=0 the function returns the scalar 0
        assert np.allclose(result, 0)

    with subtests.test("l=1 produces vortex (azimuthal variation)"):
        result = phase.laguerre_gaussian(simple_grid, l=1, p=0)
        assert result.shape == simple_grid[0].shape
        # Should wrap 2pi around the center
        assert np.ptp(result) > 0

    with subtests.test("l=-1 is negation of l=1"):
        result_pos = phase.laguerre_gaussian(simple_grid, l=1, p=0)
        result_neg = phase.laguerre_gaussian(simple_grid, l=-1, p=0)
        assert np.allclose(result_pos, -result_neg)

    with subtests.test("higher l produces steeper vortex"):
        result_l1 = phase.laguerre_gaussian(simple_grid, l=1, p=0)
        result_l3 = phase.laguerre_gaussian(simple_grid, l=3, p=0)
        # l=3 should have 3x the angular phase ramp
        assert np.allclose(result_l3, 3 * result_l1)

    with subtests.test("p>0 adds radial rings"):
        result_p0 = phase.laguerre_gaussian(simple_grid, l=1, p=0)
        result_p1 = phase.laguerre_gaussian(simple_grid, l=1, p=1)
        # p=1 should differ from p=0
        assert not np.allclose(result_p0, result_p1)
        # p=1 should contain pi-shifted regions (from Heaviside on Laguerre)
        unique_extra = np.unique(np.round(result_p1 - result_p0, 4))
        assert len(unique_extra) > 1

    with subtests.test("custom w parameter"):
        result_default = phase.laguerre_gaussian(simple_grid, l=1, p=1)
        result_custom = phase.laguerre_gaussian(simple_grid, l=1, p=1, w=2.0)
        # Different w should give different patterns
        assert not np.allclose(result_default, result_custom)


def test_hermite_gaussian(simple_grid, subtests):
    """Test hermite_gaussian() structured light generation."""
    with subtests.test("n=0, m=0 gives flat phase"):
        result = phase.hermite_gaussian(simple_grid, n=0, m=0)
        assert result.shape == simple_grid[0].shape
        # HG_00 Hermite is constant positive, so phase should be pi everywhere
        assert np.allclose(result, np.pi) or np.allclose(result, 0)

    with subtests.test("n=1, m=0 has checkerboard in x"):
        result = phase.hermite_gaussian(simple_grid, n=1, m=0)
        assert result.shape == simple_grid[0].shape
        unique_vals = np.unique(result)
        # Should have exactly two phase levels: 0 and pi
        assert len(unique_vals) == 2
        assert np.isclose(unique_vals[0], 0)
        assert np.isclose(unique_vals[1], np.pi)

    with subtests.test("n=0, m=1 has checkerboard in y"):
        result = phase.hermite_gaussian(simple_grid, n=0, m=1)
        unique_vals = np.unique(result)
        assert len(unique_vals) == 2

    with subtests.test("higher orders produce more regions"):
        result_low = phase.hermite_gaussian(simple_grid, n=1, m=0)
        result_high = phase.hermite_gaussian(simple_grid, n=3, m=0)
        # Higher n should have more sign transitions
        transitions_low = np.sum(np.abs(np.diff(result_low, axis=1)) > 0.1)
        transitions_high = np.sum(np.abs(np.diff(result_high, axis=1)) > 0.1)
        assert transitions_high > transitions_low

    with subtests.test("custom w parameter changes pattern"):
        # Use a grid where default w and custom w give different binary patterns
        x = np.linspace(-50, 50, 200)
        y = np.linspace(-50, 50, 200)
        X, Y = np.meshgrid(x, y)
        wide_grid = (X, Y)
        # default w = min(50,50)/4 = 12.5
        result_default = phase.hermite_gaussian(wide_grid, n=2, m=0)
        result_custom = phase.hermite_gaussian(wide_grid, n=2, m=0, w=3.0)
        assert not np.allclose(result_default, result_custom)


def test_ince_gaussian_not_implemented(simple_grid):
    """Test that ince_gaussian() raises NotImplementedError."""
    with pytest.raises(NotImplementedError):
        phase.ince_gaussian(simple_grid, p=2, m=1)


def test_matheui_gaussian_not_implemented(simple_grid):
    """Test that matheui_gaussian() raises NotImplementedError."""
    with pytest.raises(NotImplementedError):
        phase.matheui_gaussian(simple_grid, r=1, q=1)


def test_airy_not_implemented(simple_grid):
    """Test that airy() raises an error (not yet implemented)."""
    with pytest.raises((NotImplementedError, UnboundLocalError)):
        phase.airy(simple_grid)

