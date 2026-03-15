"""
Unit tests for Camera base class using SimulatedCamera.
"""
import warnings

import pytest
import numpy as np

from slmsuite.hardware.cameras.camera import Camera
from slmsuite.hardware.cameras.simulated import SimulatedCamera
from slmsuite.hardware.cameraslms import FourierSLM
from slmsuite.holography.toolbox.phase import zernike


class TestCamera:
    """Tests for the Camera base class via SimulatedCamera."""

    def test_selftest(self, camera, subtests):
        """camera.test() covers core properties, dtype, exposure, capture,
        averaging, HDR, WOI, and info."""
        assert camera.test() is True

    def test_init(self, slm, subtests):
        """Verify constructor sets shape, pitch, bitdepth, and resolution convention."""
        cam = SimulatedCamera(
            slm=slm, resolution=(512, 512), pitch_um=(5.5, 5.5), bitdepth=8
        )

        with subtests.test("shape"):
            assert cam.shape == (512, 512)

        with subtests.test("pitch_um"):
            np.testing.assert_allclose(cam.pitch_um, [5.5, 5.5])

        with subtests.test("bitdepth"):
            assert cam.bitdepth == 8

        with subtests.test("bitresolution"):
            assert cam.bitresolution == 256

        with subtests.test("height-width convention"):
            height, width = 480, 640
            cam2 = SimulatedCamera(slm=slm, resolution=(width, height))
            assert cam2.shape == (height, width)

        with subtests.test("defaults"):
            cam3 = SimulatedCamera(slm=slm, resolution=(256, 256))
            assert cam3.exposure_s is not None
            assert cam3.averaging is None or isinstance(cam3.averaging, int)

        with subtests.test("rotation swaps axes"):
            cam_rot = SimulatedCamera(
                slm=slm, resolution=(200, 100), rot="90"
            )
            assert cam_rot.shape == (200, 100)

        cam.close()

    def test_autoexposure(self, camera, subtests):
        """Autoexposure converges to same result from different starting points."""
        with subtests.test("convergence"):
            camera.set_exposure(0.01)
            result1 = camera.autoexposure(verbose=False)
            camera.set_exposure(1)
            result2 = camera.autoexposure(verbose=False)
            assert pytest.approx(result1, rel=0.15) == result2

        with subtests.test("custom set_fraction"):
            camera.set_exposure(0.01)
            result3 = camera.autoexposure(set_fraction=0.3, verbose=False)
            assert result3 > 0

    def test_autofocus(self, camera, slm, subtests):
        """Autofocus recovers known defocus applied via Zernike."""
        slm = slm
        slm.set_source_analytic()

        fs = FourierSLM(camera, slm)
        fs.fourier_calibrate(array_pitch=10, verbose=False)

        defocus_zernike = 1
        slm.source["phase_sim"] = zernike(slm, 4, -defocus_zernike, use_mask=False)

        with subtests.test("recovers defocus"):
            defocus_opt = camera.autofocus(set_z=slm, verbose=False)
            assert pytest.approx(defocus_opt, rel=0.25) == defocus_zernike

        with subtests.test("set_z validation"):
            with pytest.raises(ValueError, match="set_z must be"):
                camera.autofocus(set_z="not_callable")

    def test_plot(self, camera, mpl_test, subtests):
        """Camera plot method produces an axes."""
        import matplotlib.pyplot as plt

        with subtests.test("plot with captured image"):
            ax = camera.plot()
            assert ax is not None
            plt.close("all")

        with subtests.test("plot with last_image"):
            camera.get_image()
            ax = camera.plot(image=False)
            assert ax is not None
            plt.close("all")

        with subtests.test("plot with explicit image"):
            img = np.zeros(camera.shape, dtype=camera.dtype)
            ax = camera.plot(image=img, title="Test", limits=0.5)
            assert ax is not None
            plt.close("all")

    def test_info(self, camera, subtests):
        """Static info method returns a list."""
        with subtests.test("info returns list"):
            result = camera.__class__.info(verbose=False)
            assert isinstance(result, list)

