"""
Unit tests for slmsuite.holography.analysis module.
"""
import pytest
import numpy as np
from slmsuite.holography import analysis
from slmsuite.holography.analysis.fitfunctions import gaussian2d


def test_image_centroids():
    """Tests for image centroid calculation."""

    # Test centroid of a single spot
    image = np.zeros((100, 100))
    image[45:55, 45:55] = 1  # 10x10 square at position (50, 50)
    image = image[np.newaxis, :, :]  # Add batch dimension for shape (1, 100, 100)

    centroids = analysis.image_centroids(image)

    assert centroids.shape == (2, 1)
    # Centroid is relative to image center (0,0)
    assert centroids[0, 0] == pytest.approx(0, abs=1)
    assert centroids[1, 0] == pytest.approx(0, abs=1)

    # Test centroid of off-center spot
    image = np.zeros((100, 100))
    image[20:30, 70:80] = 1
    image = image[np.newaxis, :, :]  # Add batch dimension

    centroids = analysis.image_centroids(image)

    assert centroids[0, 0] == pytest.approx(25, abs=1)
    assert centroids[1, 0] == pytest.approx(-25, abs=1)

    # Test centroid calculation with custom grid
    image = np.zeros((50, 50))
    image[20:30, 20:30] = 1
    image = image[np.newaxis, :, :]  # Add batch dimension

    # Create custom grid
    x = np.arange(50)
    y = np.arange(50)
    X, Y = np.meshgrid(x, y)
    grid = (X, Y)

    centroids = analysis.image_centroids(image, grid=grid)

    assert centroids[0, 0] == pytest.approx(25, abs=1)
    assert centroids[1, 0] == pytest.approx(25, abs=1)


def test_image_moments():
    """Tests for image moment calculation."""

    # Test zeroth order moment (total intensity)
    image = np.ones((50, 50)) * 0.5
    image = image[np.newaxis, :, :]  # Add batch dimension

    # Should be sum of all pixels (returns array of shape (1,))
    moment = analysis.image_moment(image, moment=(0, 0), normalize=False)
    assert moment[0] == pytest.approx(50 * 50 * 0.5)

    # Should be 1 if normalized
    moment = analysis.image_moment(image, moment=(0, 0), normalize=True)
    assert moment[0] == pytest.approx(1)

    # Test first order moments
    image = np.zeros((100, 100))
    image[45:55, 45:55] = 1
    image = image[np.newaxis, :, :]  # Add batch dimension

    moment_x = analysis.image_moment(image, moment=(1, 0))
    moment_y = analysis.image_moment(image, moment=(0, 1))

    # Returns arrays of shape (1,)
    assert isinstance(moment_x, np.ndarray)
    assert isinstance(moment_y, np.ndarray)
    assert moment_x.shape == (1,)
    assert moment_y.shape == (1,)


def test_image_variances():
    """Tests for image variance calculation."""

    # Test variance of uniform circular spot
    image = np.zeros((100, 100))
    Y, X = np.ogrid[:100, :100]
    mask = (X - 50)**2 + (Y - 50)**2 <= 10**2
    image[mask] = 1
    image = image[np.newaxis, :, :]  # Add batch dimension

    variances = analysis.image_variances(image)

    # Returns (M20=Mxx, M02=Myy, M11=Mxy) with shape (3, 1)
    assert variances.shape == (3, 1)
    # For circular spot, Mxx ~ Myy
    assert variances[0, 0] == pytest.approx(variances[1, 0], rel=0.1)
    # Shear should be small for symmetric spot
    assert abs(variances[2, 0]) < variances[0, 0] * 0.1

    # Test variance of elliptical spot
    image = np.zeros((100, 100))
    Y, X = np.ogrid[:100, :100]
    mask = ((X - 50)/20)**2 + ((Y - 50)/10)**2 <= 1
    image[mask] = 1
    image = image[np.newaxis, :, :]  # Add batch dimension

    variances = analysis.image_variances(image)

    # For ellipse, Mxx != Myy (returns shape (3, 1))
    assert variances[0, 0] != pytest.approx(variances[1, 0], rel=0.1)


def test_image_ellipticity():
    """Tests for ellipticity calculations."""

    # Test ellipticity of circular spot
    # Circular: Mxx = Myy, Mxy = 0
    # image_ellipticity expects shape (3, N) with order (M20=Mxx, M02=Myy, M11=Mxy)
    variances = np.array([[100], [100], [0]])

    ellipticity = analysis.image_ellipticity(variances)

    assert ellipticity[0] == pytest.approx(0.0, abs=0.01)

    # Test ellipticity of elongated spot
    # Elongated in x: Mxx > Myy
    # Order is (M20=Mxx, M02=Myy, M11=Mxy)
    variances = np.array([[200], [100], [0]])

    ellipticity = analysis.image_ellipticity(variances)

    # Should not be 0 (ellipticity = 1 - lambda_min/lambda_max)
    assert ellipticity[0] > 0.1


def test_image_normalization():
    """Tests for image normalization functions."""

    # Test image normalization
    image = np.random.rand(50, 50) * 100 + 50
    image = image[np.newaxis, :, :]  # Add batch dimension

    normalized = analysis.image_normalize(image)

    # Should sum to 1
    assert np.sum(normalized) == pytest.approx(1.0)
    # Should preserve relative intensities
    assert np.argmax(normalized) == np.argmax(image)

    # Test image_normalization returns sum
    image = np.ones((50, 50)) * 2
    image = image[np.newaxis, :, :]  # Add batch dimension

    norm_value = analysis.image_normalization(image)

    assert norm_value == pytest.approx(50 * 50 * 2)


def test_image_remove_field():
    """Tests for background field removal."""

    # Test background removal
    # Image with uniform background
    image = np.ones((100, 100)) * 10
    # Add a bright spot
    image[45:55, 45:55] = 100
    image = image[np.newaxis, :, :]  # Add batch dimension

    cleaned = analysis.image_remove_field(image, deviations=2)

    # Background should be reduced/removed
    # For single 2D image input, output is still 3D with shape (1, 100, 100)
    assert cleaned.shape == (1, 100, 100)
    assert np.median(cleaned) < np.median(image)
    # Peak should remain (but will be reduced by threshold subtraction)
    assert np.max(cleaned) > 10


def test_image_relative_strehl():
    """Tests for relative Strehl ratio calculation."""

    # Test relative Strehl ratio
    # Perfect spot (high peak)
    image = np.zeros((50, 50))
    image[24, 24] = 100
    image = image[np.newaxis, :, :]  # Add batch dimension

    strehl = analysis.image_relative_strehl(image)

    # Should be high for concentrated spot
    assert strehl > 0.5


def test_image_fit():
    """Tests for image fitting functions."""

    # Test fitting a Gaussian to image data
    # Create synthetic Gaussian
    x = np.linspace(-10, 10, 50)
    y = np.linspace(-10, 10, 50)
    X, Y = np.meshgrid(x, y)
    grid = (X, Y)

    # Generate Gaussian image
    image = gaussian2d((X, Y), x0=2, y0=-1, a=10, c=1, wx=2, wy=2)
    image = image[np.newaxis, :, :]  # Add batch dimension

    # Fit it - returns array of shape (1, result_count) where result_count = 2*param_count + 1
    result = analysis.image_fit(image, grid=grid, function=gaussian2d, plot=False)

    # result[0, 0] is r-squared, result[0, 1:param_count+1] are parameters
    assert result.shape[0] == 1  # Single image
    assert result[0, 1] == pytest.approx(2, abs=0.5)  # x0
    assert result[0, 2] == pytest.approx(-1, abs=0.5)  # y0
    assert result[0, 3] == pytest.approx(10, abs=2)  # a


def test_take():
    """Tests for take function."""

    # Test take function with single integration region
    image = np.random.rand(100, 100)
    result = analysis.take(image, vectors=[50, 50], size=10, centered=True)
    assert result.shape == (1, 10, 10)  # Single region returns (1, h, w)

    # Take 3 regions at different positions
    vectors = np.array([[25, 75], [50, 50], [75, 25]]).T
    result = analysis.take(image, vectors=vectors, size=10, centered=True)
    assert result.shape == (3, 10, 10)

    # Test take function with integration enabled
    image = np.ones((100, 100))
    result = analysis.take(image, vectors=[50, 50], size=10, centered=True, integrate=True)

    # Should return a single value (sum of 10x10 ones = 100)
    assert result.shape == ()
    assert float(result) == pytest.approx(100)


def test_image_positions():
    """Tests for image_positions function."""

    # Test position calculation for single spot
    image = np.zeros((100, 100))
    image[30:40, 60:70] = 1
    image = image[np.newaxis, :, :]  # Add batch dimension

    positions = analysis.image_positions(image)

    assert len(positions) == 2  # (x, y)


def test_image_std():
    """Tests for image standard deviation calculation."""

    # Test image standard deviation calculation
    image = np.zeros((100, 100))
    # Gaussian-like spot
    Y, X = np.ogrid[:100, :100]
    image = np.exp(-((X-50)**2 + (Y-50)**2) / (2 * 10**2))
    image = image[np.newaxis, :, :]  # Add batch dimension

    std = analysis.image_std(image)

    # Should return std values
    assert len(std) == 2
    assert all(s > 0 for s in std)


def test_edge_cases():
    """Test edge cases and error handling."""

    # Test with all-zero image
    image = np.zeros((50, 50))
    image = image[np.newaxis, :, :]  # Add batch dimension

    # Should handle gracefully
    centroids = analysis.image_centroids(image)
    assert len(centroids) == 2

    # Test with single bright pixel
    image = np.zeros((50, 50))
    image[25, 25] = 1
    image = image[np.newaxis, :, :]  # Add batch dimension

    centroids = analysis.image_centroids(image)

    # Returns shape (2, 1)
    assert centroids[0, 0] == pytest.approx(0, abs=1)
    assert centroids[1, 0] == pytest.approx(0, abs=1)

    # Test handling of NaN values
    image = np.ones((50, 50))
    image[10:20, 10:20] = np.nan
    image = image[np.newaxis, :, :]  # Add batch dimension

    # With nansum=True, should ignore NaN
    norm_value = analysis.image_normalization(image, nansum=True)
    assert np.isfinite(norm_value)
