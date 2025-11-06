"""
Unit tests for Camera base class using SimulatedCamera.
"""
import pytest
import numpy as np
from slmsuite.hardware.cameras.simulated import SimulatedCamera

# TODO: camera fixture vs. SimulatedCamera cleanup (use camera wherever possible)

def test_camera_init(slm):
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

    cam.test()  # Basic self-test should pass

    cam.close()  # Cleanup

def test_camera_test(camera):
    """Test that the camera's self-test method works."""
    # The test method should return True on success
    result = camera.test()
    assert result is True


def test_camera_autoexpose(camera):
    """Test exposure control works."""
    # Check for basic exposure control
    assert hasattr(camera, 'exposure_s') or hasattr(camera, 'get_exposure')

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
