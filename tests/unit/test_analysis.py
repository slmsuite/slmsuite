"""
Unit tests for slmsuite.holography.analysis module.
"""
import pytest
import numpy as np
from slmsuite.holography import analysis
from slmsuite.holography.analysis.fitfunctions import gaussian2d


class TestImageCentroids:
    """Tests for image centroid calculation."""

    def test_centroid_single_spot(self):
        """Test centroid of a single spot."""
        image = np.zeros((100, 100))
        image[45:55, 45:55] = 1  # 10x10 square at position (50, 50)

        centroids = analysis.image_centroids(image)

        assert centroids.shape == (2, 1)
        # Centroid should be near (50, 50) given 0-indexed center
        assert centroids[0, 0] == pytest.approx(49.5, abs=1)
        assert centroids[1, 0] == pytest.approx(49.5, abs=1)

    def test_centroid_off_center(self):
        """Test centroid of off-center spot."""
        image = np.zeros((100, 100))
        image[20:30, 70:80] = 1

        centroids = analysis.image_centroids(image)

        assert centroids[0, 0] == pytest.approx(74.5, abs=1)
        assert centroids[1, 0] == pytest.approx(24.5, abs=1)

    def test_centroid_with_grid(self):
        """Test centroid calculation with custom grid."""
        image = np.zeros((50, 50))
        image[20:30, 20:30] = 1

        # Create custom grid
        x = np.arange(50)
        y = np.arange(50)
        X, Y = np.meshgrid(x, y)
        grid = (X, Y)

        centroids = analysis.image_centroids(image, grid=grid)

        assert len(centroids) == 2


class TestImageMoments:
    """Tests for image moment calculation."""

    def test_moment_zeroth_order(self):
        """Test zeroth order moment (total intensity)."""
        image = np.ones((50, 50)) * 0.5

        moment = analysis.image_moment(image, moment=(0, 0))

        # Should be sum of all pixels (returns array of shape (1,))
        assert moment[0] == pytest.approx(50 * 50 * 0.5)

    def test_moment_first_order(self):
        """Test first order moments."""
        image = np.zeros((100, 100))
        image[45:55, 45:55] = 1

        moment_x = analysis.image_moment(image, moment=(1, 0))
        moment_y = analysis.image_moment(image, moment=(0, 1))

        # Returns arrays of shape (1,)
        assert isinstance(moment_x, np.ndarray)
        assert isinstance(moment_y, np.ndarray)
        assert moment_x.shape == (1,)
        assert moment_y.shape == (1,)


class TestImageVariances:
    """Tests for image variance calculation."""

    def test_variance_uniform_spot(self):
        """Test variance of uniform circular spot."""
        image = np.zeros((100, 100))
        Y, X = np.ogrid[:100, :100]
        mask = (X - 50)**2 + (Y - 50)**2 <= 10**2
        image[mask] = 1

        variances = analysis.image_variances(image)

        # Returns (M20=Mxx, M02=Myy, M11=Mxy) with shape (3, 1)
        assert variances.shape == (3, 1)
        # For circular spot, Mxx ~ Myy
        assert variances[0, 0] == pytest.approx(variances[1, 0], rel=0.1)
        # Shear should be small for symmetric spot
        assert abs(variances[2, 0]) < variances[0, 0] * 0.1

    def test_variance_elliptical_spot(self):
        """Test variance of elliptical spot."""
        image = np.zeros((100, 100))
        Y, X = np.ogrid[:100, :100]
        mask = ((X - 50)/20)**2 + ((Y - 50)/10)**2 <= 1
        image[mask] = 1

        variances = analysis.image_variances(image)

        # For ellipse, Mxx != Myy (returns shape (3, 1))
        assert variances[0, 0] != pytest.approx(variances[1, 0], rel=0.1)


class TestImageEllipticity:
    """Tests for ellipticity calculations."""

    def test_ellipticity_circular(self):
        """Test ellipticity of circular spot."""
        # Circular: Mxx = Myy, Mxy = 0
        # image_ellipticity expects shape (3, N) with order (M20=Mxx, M02=Myy, M11=Mxy)
        variances = np.array([[100], [100], [0]])

        ellipticity = analysis.image_ellipticity(variances)

        assert ellipticity[0] == pytest.approx(0.0, abs=0.01)

    def test_ellipticity_elongated(self):
        """Test ellipticity of elongated spot."""
        # Elongated in x: Mxx > Myy
        # Order is (M20=Mxx, M02=Myy, M11=Mxy)
        variances = np.array([[200], [100], [0]])

        ellipticity = analysis.image_ellipticity(variances)

        # Should not be 0 (ellipticity = 1 - lambda_min/lambda_max)
        assert ellipticity[0] > 0.1


class TestImageNormalization:
    """Tests for image normalization functions."""

    def test_image_normalize(self):
        """Test image normalization."""
        image = np.random.rand(50, 50) * 100 + 50

        normalized = analysis.image_normalize(image)

        # Should sum to 1
        assert np.sum(normalized) == pytest.approx(1.0)
        # Should preserve relative intensities
        assert np.argmax(normalized) == np.argmax(image)

    def test_image_normalization_value(self):
        """Test image_normalization returns sum."""
        image = np.ones((50, 50)) * 2

        norm_value = analysis.image_normalization(image)

        assert norm_value == pytest.approx(50 * 50 * 2)


class TestImageRemoveField:
    """Tests for background field removal."""

    def test_remove_field_with_background(self):
        """Test background removal."""
        # Image with uniform background
        image = np.ones((100, 100)) * 10
        # Add a bright spot
        image[45:55, 45:55] = 100

        cleaned = analysis.image_remove_field(image, deviations=2)

        # Background should be reduced/removed
        # For single 2D image input, output is still 2D
        assert cleaned.shape == (100, 100)
        assert np.median(cleaned) < np.median(image)
        # Peak should remain (but will be reduced by threshold subtraction)
        assert np.max(cleaned) > 10


class TestImageRelativeStrehl:
    """Tests for relative Strehl ratio calculation."""

    def test_relative_strehl(self):
        """Test relative Strehl ratio."""
        # Perfect spot (high peak)
        image = np.zeros((50, 50))
        image[24, 24] = 100

        strehl = analysis.image_relative_strehl(image)

        # Should be high for concentrated spot
        assert strehl > 0.5


class TestImageFit:
    """Tests for image fitting functions."""

    def test_image_fit_gaussian(self):
        """Test fitting a Gaussian to image data."""
        # Create synthetic Gaussian
        x = np.linspace(-10, 10, 50)
        y = np.linspace(-10, 10, 50)
        X, Y = np.meshgrid(x, y)
        grid = (X, Y)

        # Generate Gaussian image
        image = gaussian2d((X, Y), x0=2, y0=-1, a=10, c=1, wx=2, wy=2)

        # Fit it - returns array of shape (1, result_count) where result_count = 2*param_count + 1
        result = analysis.image_fit(image, grid=grid, function=gaussian2d, plot=False)

        # result[0, 0] is r-squared, result[0, 1:param_count+1] are parameters
        assert result.shape[0] == 1  # Single image
        assert result[0, 1] == pytest.approx(2, abs=0.5)  # x0
        assert result[0, 2] == pytest.approx(-1, abs=0.5)  # y0
        assert result[0, 3] == pytest.approx(10, abs=2)  # a


class TestTakeFunctions:
    """Tests for take and take_plot functions."""

    def test_take_single_region(self):
        """Test take function with single integration region."""
        image = np.random.rand(100, 100)
        # Take a 10x10 region centered at (50, 50)

        result = analysis.take(image, vectors=[50, 50], size=10, centered=True)

        assert result.shape == (10, 10)

    def test_take_multiple_regions(self):
        """Test take function with multiple integration regions."""
        image = np.random.rand(100, 100)
        # Take 3 regions at different positions
        vectors = [[25, 75], [50, 50], [75, 25]]

        result = analysis.take(image, vectors=vectors, size=10, centered=True)

        assert result.shape == (3, 10, 10)

    def test_take_with_integration(self):
        """Test take function with integration enabled."""
        image = np.ones((100, 100))
        # Take a 10x10 region and integrate

        result = analysis.take(image, vectors=[50, 50], size=10, centered=True, integrate=True)

        # Should return a single value (sum of 10x10 ones = 100)
        assert result.shape == (1,)
        assert result[0] == pytest.approx(100)


class TestImagePositions:
    """Tests for image_positions function."""

    def test_positions_single_image(self):
        """Test position calculation for single spot."""
        image = np.zeros((100, 100))
        image[30:40, 60:70] = 1

        positions = analysis.image_positions(image)

        assert len(positions) == 2  # (x, y)


class TestImageStd:
    """Tests for image standard deviation calculation."""

    def test_image_std(self):
        """Test image standard deviation calculation."""
        image = np.zeros((100, 100))
        # Gaussian-like spot
        Y, X = np.ogrid[:100, :100]
        image = np.exp(-((X-50)**2 + (Y-50)**2) / (2 * 10**2))

        std = analysis.image_std(image)

        # Should return std values
        assert len(std) == 2
        assert all(s > 0 for s in std)


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_image(self):
        """Test with all-zero image."""
        image = np.zeros((50, 50))

        # Should handle gracefully
        centroids = analysis.image_centroids(image)
        assert len(centroids) == 2

    def test_single_pixel(self):
        """Test with single bright pixel."""
        image = np.zeros((50, 50))
        image[25, 25] = 1

        centroids = analysis.image_centroids(image)

        # Returns shape (2, 1)
        assert centroids[0, 0] == pytest.approx(25, abs=0.1)
        assert centroids[1, 0] == pytest.approx(25, abs=0.1)

    def test_nan_handling(self):
        """Test handling of NaN values."""
        image = np.ones((50, 50))
        image[10:20, 10:20] = np.nan

        # With nansum=True, should ignore NaN
        norm_value = analysis.image_normalization(image, nansum=True)
        assert np.isfinite(norm_value)
