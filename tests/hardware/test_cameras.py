"""
Unit tests for Camera base class using SimulatedCamera.
"""
import pytest
import numpy as np
from slmsuite.hardware.cameras.simulated import SimulatedCamera
from slmsuite.hardware.cameraslms import FourierSLM
from slmsuite.holography.toolbox.phase import zernike

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

    # cam.test()  # Basic self-test should pass

    cam.close()  # Cleanup

def test_camera_test(camera):
    """Test that the camera's self-test method works."""
    # The test method should return True on success
    result = camera.test()
    assert result is True


def test_camera_autoexposure(slm):
    """Test exposure control works."""
    cam = SimulatedCamera(
        slm=slm,
        resolution=(512, 512),
        pitch_um=(5.5, 5.5),
        bitdepth=8
    )

    cam.set_exposure(0.01)
    result1 = cam.autoexposure(verbose=True)

    cam.set_exposure(1)
    result2 = cam.autoexposure(verbose=True)

    assert pytest.approx(result1, rel=0.1) == result2

def test_camera_autofocus(slm):
    cam = SimulatedCamera(
        slm=slm,
        resolution=(512, 512),
        pitch_um=(5.5, 5.5),
        bitdepth=8
    )

    slm.set_source_analytic()

    fs = FourierSLM(cam, slm)
    fs.fourier_calibrate(array_pitch=40, verbose=False)

    defocus_zernike = 1
    slm.source['phase_sim'] = zernike(slm, 4, -defocus_zernike, use_mask=False)

    defocus_opt = cam.autofocus(
        set_z=slm,
        verbose=True
    )

    assert pytest.approx(defocus_opt, rel=0.2) == defocus_zernike

