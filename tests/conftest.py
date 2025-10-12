"""
Pytest configuration and fixtures for slmsuite tests.

The fixtures in this file support testing with any SLM or Camera subclass.
By default, SimulatedSLM and SimulatedCamera are used for fast, hardware-free testing.

To test with real hardware, set environment variables:
    SLMSUITE_TEST_SLM_CLASS=slmsuite.hardware.slms.thorlabs.ThorlabsSLM
    SLMSUITE_TEST_SLM_ARGS='{"monitor_id": 1}'
    SLMSUITE_TEST_CAMERA_CLASS=slmsuite.hardware.cameras.thorlabs.ThorlabsCamera
    SLMSUITE_TEST_CAMERA_ARGS='{"serial": "12345"}'
"""
import pytest
import numpy as np
import tempfile
import os
import json
import importlib
import matplotlib
import matplotlib.pyplot as plt

try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    cp = np
    HAS_CUPY = False

from slmsuite.hardware.cameras.simulated import SimulatedCamera
from slmsuite.hardware.slms.simulated import SimulatedSLM


@pytest.fixture(scope="session")
def has_cupy():
    """Check if CuPy is available for GPU tests."""
    return HAS_CUPY


def _get_class_from_string(class_path):
    """
    Import and return a class from a module path string.

    Parameters
    ----------
    class_path : str
        Full path to class, e.g., 'slmsuite.hardware.slms.simulated.SimulatedSLM'

    Returns
    -------
    class
        The imported class
    """
    module_path, class_name = class_path.rsplit('.', 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


@pytest.fixture
def slm_class():
    """
    Return the SLM class to use for testing.

    By default returns SimulatedSLM. Can be overridden via environment variable:
    SLMSUITE_TEST_SLM_CLASS=slmsuite.hardware.slms.thorlabs.ThorlabsSLM

    Returns
    -------
    class
        SLM subclass to instantiate
    """
    class_path = os.environ.get('SLMSUITE_TEST_SLM_CLASS', None)
    if class_path:
        return _get_class_from_string(class_path)
    return SimulatedSLM


@pytest.fixture
def slm_kwargs():
    """
    Return keyword arguments for SLM instantiation.

    By default returns arguments for SimulatedSLM. Can be overridden via:
    SLMSUITE_TEST_SLM_ARGS='{"monitor_id": 1, "bitdepth": 8}'

    Returns
    -------
    dict
        Keyword arguments for SLM constructor
    """
    args_json = os.environ.get('SLMSUITE_TEST_SLM_ARGS', None)
    if args_json:
        return json.loads(args_json)

    # Default args for SimulatedSLM
    return {
        'resolution': (1920, 1080),
        'pitch_um': (8.0, 8.0),
        'bitdepth': 8,
        'wav_um': 0.78
    }


@pytest.fixture
def slm(slm_class, slm_kwargs):
    """
    Fixture providing an SLM instance for testing.

    By default returns SimulatedSLM, but can be configured to return any SLM subclass
    via environment variables SLMSUITE_TEST_SLM_CLASS and SLMSUITE_TEST_SLM_ARGS.
    """
    slm_instance = slm_class(**slm_kwargs)
    yield slm_instance
    # Cleanup
    if hasattr(slm_instance, 'close'):
        try:
            slm_instance.close()
        except:
            pass


@pytest.fixture
def camera_class():
    """
    Return the Camera class to use for testing.

    By default returns SimulatedCamera. Can be overridden via:
    SLMSUITE_TEST_CAMERA_CLASS=slmsuite.hardware.cameras.thorlabs.ThorlabsCamera

    Returns
    -------
    class
        Camera subclass to instantiate
    """
    class_path = os.environ.get('SLMSUITE_TEST_CAMERA_CLASS', None)
    if class_path:
        return _get_class_from_string(class_path)
    return SimulatedCamera


@pytest.fixture
def camera_kwargs(slm):
    """
    Return keyword arguments for Camera instantiation.

    By default returns arguments for SimulatedCamera. Can be overridden via:
    SLMSUITE_TEST_CAMERA_ARGS='{"serial": "12345", "bitdepth": 8}'

    Parameters
    ----------
    slm : SLM
        SLM instance (required for SimulatedCamera)

    Returns
    -------
    dict
        Keyword arguments for Camera constructor
    """
    args_json = os.environ.get('SLMSUITE_TEST_CAMERA_ARGS', None)
    if args_json:
        return json.loads(args_json)

    # Default args for SimulatedCamera
    return {
        'slm': slm,
        'resolution': (512, 512),
        'pitch_um': (5.5, 5.5),
        'bitdepth': 8
    }


@pytest.fixture
def camera(camera_class, camera_kwargs):
    """
    Fixture providing a Camera instance for testing.

    By default returns SimulatedCamera, but can be configured to return any Camera subclass
    via environment variables SLMSUITE_TEST_CAMERA_CLASS and SLMSUITE_TEST_CAMERA_ARGS.
    """
    cam = camera_class(**camera_kwargs)
    yield cam
    # Cleanup
    if hasattr(cam, 'close'):
        try:
            cam.close()
        except:
            pass

@pytest.fixture
def temp_dir():
    """Fixture providing a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir

@pytest.fixture
def random_phase():
    """Fixture providing random phase pattern for testing."""
    return np.random.rand(256, 256) * 2 * np.pi

@pytest.fixture
def random_amplitude():
    """Fixture providing random amplitude pattern for testing."""
    return np.random.rand(256, 256)


def pytest_configure(config):
    """Configure pytest with custom settings."""

    matplotlib.use("tkagg")
    plt.ion()
    return