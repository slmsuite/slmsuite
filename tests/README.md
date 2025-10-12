# slmsuite Testing Framework

This directory contains the unified testing framework for slmsuite, designed to work both with and without connected hardware.

## Quick Start

```bash
# Run all tests
pytest

# Run only unit tests
pytest tests/unit/

# Run integration tests
pytest tests/integration/

# Run with verbose output
pytest -v

# Run specific test file
pytest tests/unit/test_algorithms.py

# Run specific test class or function
pytest tests/unit/test_algorithms.py::TestHologramGS::test_gs_runs_without_error
```

## Test Structure

```
tests/
├── conftest.py          # Shared fixtures and configuration
├── pytest.ini           # Pytest configuration (in parent directory)
├── unit/                # Unit tests (fast, no hardware)
│   ├── test_misc.py              # Math utilities, fit functions
│   ├── test_phase_toolbox.py     # Phase patterns (blaze, Zernike, etc)
│   ├── test_algorithms.py        # Holography algorithms (GS, WGS)
│   ├── test_analysis.py          # Image analysis functions
│   ├── test_camera.py            # Camera base class tests
│   └── test_slm.py               # SLM base class tests
└── integration/         # Integration tests (simulated hardware)
    └── (to be added)
```

## Test Philosophy

### No Hardware Required by Default
All tests use simulated hardware (SimulatedCamera, SimulatedSLM) and synthetic data. This allows:
- Fast test execution in CI/CD
- Development without physical hardware
- Consistent test results across environments

### Test Categories

#### Unit Tests (`tests/unit/`)
- **Fast** (<10s total runtime)
- **Isolated**: Test individual functions/classes
- **No hardware**: Use SimulatedCamera/SimulatedSLM or mock objects
- **Comprehensive**: Cover edge cases, error handling, data validation

Modules covered:
- `test_misc.py`: Math functions, fit functions (Gaussian, Lorentzian, etc.)
- `test_phase_toolbox.py`: Phase pattern generation (blaze, sinusoid, binary, lens, Zernike)
- `test_algorithms.py`: Hologram optimization (GS, WGS variants, MRAF)
- `test_analysis.py`: Image analysis (centroids, moments, variances, fitting)
- `test_camera.py`: Camera interface (using SimulatedCamera)
- `test_slm.py`: SLM interface (using SimulatedSLM)

#### Integration Tests (`tests/integration/`)
- **Multi-component**: Test interactions between Camera, SLM, and algorithms
- **Simulated hardware**: Use SimulatedCamera + SimulatedSLM
- **Realistic workflows**: End-to-end calibration and optimization

#### Hardware Tests (future, opt-in)
- Tests requiring physical hardware will be marked with `@pytest.mark.hardware`
- Skip by default: `pytest -m "not hardware"`
- Run when available: `pytest -m hardware`

## Fixtures

### Available Fixtures (from `conftest.py`)

The test fixtures are designed to work with **any** SLM or Camera subclass, not just simulated hardware.

#### Hardware Fixtures

```python
@pytest.fixture
def slm():
    """
    Provides an SLM instance for testing.

    By default: SimulatedSLM (1920x1080, 8-bit, 780nm)

    Can be configured via environment variables:
        SLMSUITE_TEST_SLM_CLASS=slmsuite.hardware.slms.thorlabs.ThorlabsSLM
        SLMSUITE_TEST_SLM_ARGS='{"monitor_id": 1, "bitdepth": 8}'
    """

@pytest.fixture
def camera(slm):
    """
    Provides a Camera instance for testing.

    By default: SimulatedCamera (512x512, 8-bit)

    Can be configured via environment variables:
        SLMSUITE_TEST_CAMERA_CLASS=slmsuite.hardware.cameras.thorlabs.ThorlabsCamera
        SLMSUITE_TEST_CAMERA_ARGS='{"serial": "12345", "bitdepth": 8}'
    """

@pytest.fixture
def temp_dir():
    """Provides a temporary directory for test file I/O"""

@pytest.fixture
def random_phase():
    """Provides a 256x256 random phase pattern"""

@pytest.fixture
def random_amplitude():
    """Provides a 256x256 random amplitude pattern"""
```

### Using Fixtures

#### Basic Usage (Simulated Hardware - Default)

```python
def test_example(camera, temp_dir):
    """Test using camera and temp directory."""
    image = camera.get_image()
    assert image.shape == (512, 512)

    # Save to temp directory
    np.save(f"{temp_dir}/image.npy", image)
```

#### Testing with Real Hardware

To run tests with real hardware instead of simulated:

```bash
# Linux/Mac
export SLMSUITE_TEST_SLM_CLASS=slmsuite.hardware.slms.thorlabs.ThorlabsSLM
export SLMSUITE_TEST_SLM_ARGS='{"monitor_id": 1}'
export SLMSUITE_TEST_CAMERA_CLASS=slmsuite.hardware.cameras.thorlabs.ThorlabsCamera
export SLMSUITE_TEST_CAMERA_ARGS='{"serial": "12345"}'
pytest tests/unit/

# Windows (PowerShell)
$env:SLMSUITE_TEST_SLM_CLASS="slmsuite.hardware.slms.thorlabs.ThorlabsSLM"
$env:SLMSUITE_TEST_SLM_ARGS='{"monitor_id": 1}'
pytest tests/unit/

# Windows (CMD)
set SLMSUITE_TEST_SLM_CLASS=slmsuite.hardware.slms.thorlabs.ThorlabsSLM
set SLMSUITE_TEST_SLM_ARGS={"monitor_id": 1}
pytest tests/unit/
```

The test fixtures automatically handle:
- Instantiation with appropriate arguments
- Cleanup and resource management
- Compatibility with any SLM/Camera subclass

## Test Markers

Markers are defined in `pytest.ini` and can be used to categorize tests:

```python
@pytest.mark.gpu
def test_cupy_algorithm():
    """Test requiring CuPy/GPU."""
    if not HAS_CUPY:
        pytest.skip("CuPy not available")
    # ... GPU test code

@pytest.mark.slow
def test_long_optimization():
    """Test that takes >5 seconds."""
    # ... long-running test code
```

Run tests by marker:
```bash
# Skip GPU tests
pytest -m "not gpu"

# Run only slow tests
pytest -m slow

# Skip both GPU and slow tests
pytest -m "not gpu and not slow"
```

## Test Results

Current status: **93 passed / 144 total (65% pass rate)**

### Passing Test Categories:
- ✅ Fit functions (1D and 2D)
- ✅ Phase toolbox (blaze, sinusoid, binary, lens, axicon)
- ✅ Hologram construction and basic optimization
- ✅ SLM base class (MockSLM implementation)
- ✅ Image normalization and analysis basics

### Known Issues:
- Some Hologram.optimize() tests need return value adjustments
- Camera tests need SimulatedCamera signature updates
- Some phase toolbox function signatures differ from tests (quadrants, bahtinov)
- Zernike index conversion uses different naming convention

## Writing New Tests

### Test Class Structure
```python
class TestFeatureName:
    """Tests for specific feature."""

    def test_basic_functionality(self):
        """Test the happy path."""
        result = my_function(input_data)
        assert result == expected_output

    def test_edge_case(self):
        """Test boundary conditions."""
        assert my_function(edge_input) handles correctly

    def test_error_handling(self):
        """Test that errors are raised appropriately."""
        with pytest.raises(ValueError):
            my_function(invalid_input)
```

### Testing Best Practices

1. **Use descriptive test names**: `test_gs_improves_efficiency` not `test_gs_1`
2. **Test one thing**: Each test should verify a single behavior
3. **Use fixtures**: Avoid code duplication with shared fixtures
4. **Test edge cases**: Empty arrays, zero values, NaN, extreme sizes
5. **Use pytest.approx()**: For floating-point comparisons
6. **Parametrize**: Test multiple inputs with `@pytest.mark.parametrize`

### Example: Parametrized Test
```python
@pytest.mark.parametrize("bitdepth,expected", [
    (8, 256),
    (10, 1024),
    (12, 4096),
])
def test_bitresolution(bitdepth, expected):
    """Test bitresolution calculation for various bitdepths."""
    slm = SimulatedSLM(bitdepth=bitdepth)
    assert slm.bitresolution == expected
```

## GPU Testing

Tests can run on both CPU (NumPy) and GPU (CuPy) when available:

```python
@pytest.mark.skipif(not HAS_CUPY, reason="CuPy not available")
def test_gpu_acceleration():
    """Test that runs only if CuPy is installed."""
    import cupy as cp
    target = cp.zeros((128, 128))
    # ... GPU test code
```

## Debugging Failed Tests

```bash
# Show full traceback
pytest -v --tb=long

# Stop at first failure
pytest -x

# Show local variables in traceback
pytest -l

# Run last failed tests
pytest --lf

# Enter debugger on failure
pytest --pdb
```

## TODOs

- [ ] Integration tests for CameraSLM workflows
- [ ] Integration tests for FourierSLM calibration
- [ ] Integration tests for SpotHologram optimization
- [ ] CPU/GPU parametrization for all algorithm tests
- [ ] Performance benchmarking tests
- [ ] Coverage reporting (pytest-cov)
- [ ] Hardware detection and optional hardware tests
- [ ] Test data fixtures for common calibration scenarios