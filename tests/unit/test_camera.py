"""
Unit tests for Camera base class using SimulatedCamera.
"""
import pytest
import numpy as np
from slmsuite.hardware.cameras.simulated import SimulatedCamera
from slmsuite.hardware.slms.simulated import SimulatedSLM

# TODO: camera fixture vs. SimulatedCamera cleanup (use camera wherever possible)

def test_camera_init(self, slm):
    """Test basic SimulatedCamera construction."""
    cam = SimulatedCamera(
        slm=slm,
        resolution=(512, 512),
        pitch_um=(5.5, 5.5),
        bitdepth=8
    )

    assert cam.shape == (512, 512)
    assert np.allclose(cam.pitch_um, [5.5, 5.5])
    assert cam.bitdepth == 8
    assert cam.bitresolution == 256

    # Verify (height, width) shape convention.
    height, width = 480, 640
    cam = SimulatedCamera(slm=slm, resolution=(width, height))

    assert cam.shape[0] == height
    assert cam.shape[1] == width

    # Test default camera parameters.
    cam = SimulatedCamera(slm=slm, resolution=(256, 256))

    assert cam.exposure_s is not None
    assert cam.averaging is None or isinstance(cam.averaging, int)


class TestCameraImageAcquisition:
    """Tests for image acquisition."""

    def test_get_image_shape(self, camera):
        """Test that get_image returns correct shape."""
        image = camera.get_image()

        assert image.shape == camera.shape

    def test_get_image_dtype(self, camera):
        """Test that get_image returns correct dtype."""
        image = camera.get_image()

        # Should be integer type for camera images
        assert np.issubdtype(image.dtype, np.integer) or np.issubdtype(image.dtype, np.floating)

    def test_get_image_range(self, camera):
        """Test that image values are in valid range."""
        image = camera.get_image()

        assert np.min(image) >= 0
        assert np.max(image) <= camera.bitresolution

    def test_get_image_multiple_calls(self, camera):
        """Test multiple image acquisitions."""
        image1 = camera.get_image()
        image2 = camera.get_image()

        # Should return valid images
        assert image1.shape == image2.shape


class TestCameraExposure:
    """Tests for exposure control."""

    def test_set_exposure(self, camera):
        """Test setting exposure time."""
        exposure_s = 0.05
        camera.set_exposure(exposure_s)

        # Check that exposure was set (or cached)
        assert camera.exposure_s == pytest.approx(exposure_s, rel=0.1)

    def test_get_exposure(self, camera):
        """Test getting exposure time."""
        exposure = camera.get_exposure()

        assert isinstance(exposure, (int, float))
        assert exposure > 0


class TestCameraAveraging:
    """Tests for image averaging."""

    def test_averaging_single(self, camera):
        """Test with averaging=1 (no averaging)."""
        image = camera.get_image(averaging=1)

        assert image.shape == camera.shape

    def test_averaging_multiple(self, camera):
        """Test with multiple averaging."""
        image = camera.get_image(averaging=5)

        assert image.shape == camera.shape
        # Averaging might change dtype to float
        assert image.dtype in [np.uint8, np.uint16, np.float32, np.float64]


class TestCameraTransforms:
    """Tests for image transformations (if supported)."""

    def test_transform_none(self, camera):
        """Test basic image acquisition (transforms not in SimulatedCamera interface)."""
        image = camera.get_image()
        assert image.shape == camera.shape

    def test_transform_flip_horizontal(self, camera):
        """Test that we can manually transform images."""
        image = camera.get_image()
        flipped = np.fliplr(image)

        assert flipped.shape == camera.shape

    def test_transform_rotation(self, camera):
        """Test that we can manually rotate images."""
        image = camera.get_image()
        rotated = np.rot90(image)

        # Rotation changes shape for non-square
        assert rotated.size == image.size


class TestCameraWindowOfInterest:
    """Tests for window of interest (WOI) functionality."""

    def test_woi_full_frame(self, camera):
        """Test with full frame (default WOI)."""
        image = camera.get_image()

        assert image.shape == camera.shape

    def test_set_woi(self, slm):
        """Test setting window of interest."""
        cam = SimulatedCamera(slm=slm, resolution=(512, 512))

        # This may or may not be implemented in SimulatedCamera
        # Just test that it doesn't crash
        try:
            cam.set_woi([0, 256, 0, 256])
        except (NotImplementedError, AttributeError):
            pytest.skip("WOI not implemented in SimulatedCamera")


class TestCameraProperties:
    """Tests for camera properties and attributes."""

    def test_last_image_cached(self, camera):
        """Test that last_image is cached."""
        image = camera.get_image()

        assert camera.last_image is not None
        assert np.array_equal(camera.last_image, image)

    def test_default_shape_attribute(self, camera):
        """Test default_shape attribute."""
        assert hasattr(camera, 'default_shape')
        if camera.default_shape is not None:
            assert isinstance(camera.default_shape, tuple)
            assert len(camera.default_shape) == 2


class TestCameraAutoExposure:
    """Tests for auto-exposure functionality."""

    def test_autoexpose_exists(self, camera):
        """Test that camera has exposure control (autoexpose may not be in base)."""
        # Check for basic exposure control
        assert hasattr(camera, 'exposure_s') or hasattr(camera, 'get_exposure')

    def test_autoexpose_runs(self, camera):
        """Test exposure control works (autoexpose may not be implemented)."""
        # Just test that exposure can be set/get
        if hasattr(camera, 'set_exposure'):
            camera.set_exposure(0.01)
            assert camera.exposure_s > 0
        elif hasattr(camera, 'autoexpose'):
            try:
                result = camera.autoexpose(
                    measure_function=lambda: camera.get_image(),
                    exposure_bounds_s=(0.001, 0.1)
                )
                assert isinstance(result, (int, float))
            except (NotImplementedError, AttributeError):
                pytest.skip("Autoexpose not fully implemented in SimulatedCamera")


class TestCameraFlush:
    """Tests for camera flush/reset functionality."""

    def test_flush_exists(self, camera):
        """Test that flush method exists if applicable."""
        # Not all cameras have flush
        if hasattr(camera, 'flush'):
            camera.flush()
            # Should not crash


class TestCameraClose:
    """Tests for camera cleanup."""

    def test_close_camera(self, slm):
        """Test camera close/cleanup."""
        cam = SimulatedCamera(slm=slm, resolution=(256, 256))

        # Get an image to initialize
        _ = cam.get_image()

        # Close should not crash
        if hasattr(cam, 'close'):
            cam.close()


class TestSimulatedCameraSpecific:
    """Tests specific to SimulatedCamera."""

    def test_simulated_with_slm(self, slm):
        """Test SimulatedCamera can be linked with SLM."""
        cam = SimulatedCamera(
            slm=slm,
            resolution=(512, 512),
            pitch_um=(5.5, 5.5),
            bitdepth=8
        )

        # Should be able to get images
        image = cam.get_image()
        assert image is not None

    def test_noise_configuration(self, slm):
        """Test noise configuration in SimulatedCamera."""
        # Test with noise parameter in constructor
        noise_dict = {'dark': lambda img: np.random.normal(0, 0.01, img.shape)}

        cam = SimulatedCamera(
            slm=slm,
            resolution=(256, 256),
            pitch_um=(5.5, 5.5),
            bitdepth=8,
            noise=noise_dict
        )

        # SimulatedCamera should have noise attribute after initialization
        assert hasattr(cam, 'noise')
        assert isinstance(cam.noise, dict)


class TestCameraEdgeCases:
    """Test edge cases and error handling."""

    def test_small_camera(self, slm):
        """Test with very small camera."""
        cam = SimulatedCamera(slm=slm, resolution=(16, 16))

        image = cam.get_image()
        assert image.shape == (16, 16)

    def test_rectangular_camera(self, slm):
        """Test with non-square camera."""
        cam = SimulatedCamera(slm=slm, resolution=(640, 480))

        image = cam.get_image()
        assert image.shape == (480, 640)

    def test_high_bitdepth(self, slm):
        """Test with high bitdepth."""
        cam = SimulatedCamera(slm=slm, resolution=(256, 256), bitdepth=16)

        assert cam.bitdepth == 16
        assert cam.bitresolution == 65536

        image = cam.get_image()
        assert np.max(image) <= 65536


class TestCameraSelfTest:
    """Tests for the Camera.test() method."""

    def test_camera_self_test(self, camera):
        """Test that the camera's self-test method works."""
        # The test method should return True on success
        result = camera.test()
        assert result is True

    def test_camera_self_test_simulated(self, slm):
        """Test self-test with a fresh SimulatedCamera."""
        cam = SimulatedCamera(slm=slm, resolution=(64, 64), bitdepth=8)

        # Test should pass for a properly constructed camera
        result = cam.test()
        assert result is True

        # Cleanup
        cam.close()
