# slmsuite Testing Framework

This directory contains the unified testing framework for slmsuite, designed to work both with and without connected hardware.

## Quick Start

```bash
# Run all tests
pytest

# Run tests for specific module
pytest tests/holography/
pytest tests/hardware/
pytest tests/misc/

# Run with verbose output
pytest -v

# Run specific test file
pytest tests/holography/test_algorithms.py

# Run specific test class or function
pytest tests/holography/test_algorithms.py::TestHologram::test_gs_converges
```

## Test Structure

The test directory structure mirrors the package structure:

```
tests/
├── conftest.py              # Shared fixtures and configuration
├── README.md                # This file
├── pytest.ini               # Pytest configuration (in parent directory)
├── hardware/                # Hardware module tests
│   ├── test_camera.py       # Camera base class and implementations
│   └── test_slm.py          # SLM base class and implementations
├── holography/              # Holography module tests
│   ├── test_algorithms.py   # Hologram optimization (GS, WGS, feedback)
│   ├── test_analysis.py     # Image analysis and statistics
│   └── test_toolbox.py      # Phase patterns (blaze, Zernike, etc)
└── misc/                    # Misc utilities tests
    └── test_misc.py         # Math functions, fit functions, I/O
```

This structure mirrors `slmsuite/`:
- `tests/hardware/` → `slmsuite/hardware/`
- `tests/holography/` → `slmsuite/holography/`
- `tests/misc/` → `slmsuite/misc/`

## Test Philosophy

### No Hardware Required by Default
All tests use simulated hardware (SimulatedCamera, SimulatedSLM) and synthetic data. This allows:
- Fast test execution in CI/CD
- Development without physical hardware
- Consistent test results across environments

### Test Categories

All tests are designed to run without physical hardware using simulated devices:

#### Hardware Tests (`tests/hardware/`)
- **Camera tests** (`test_camera.py`): Camera base class interface, image acquisition, exposure control
- **SLM tests** (`test_slm.py`): SLM base class interface, phase display, calibration

#### Holography Tests (`tests/holography/`)
- **Algorithm tests** (`test_algorithms.py`): Hologram optimization (GS, WGS variants), feedback methods, convergence
- **Toolbox tests** (`test_toolbox.py`): Phase pattern generation (blaze, sinusoid, binary, lens, Zernike)
- **Analysis tests** (`test_analysis.py`): Image analysis (centroids, moments, variances, fitting)

#### Misc Tests (`tests/misc/`)
- **Math and fit functions** (`test_misc.py`): Utility functions, 1D/2D fitting (Gaussian, Lorentzian, etc.)

#### Test Characteristics
- **Fast**: Most tests complete in <1 second (mark long tests with `@pytest.mark.slow`)
- **Isolated**: Each test is independent and doesn't require external state
- **No hardware by default**: Use SimulatedCamera/SimulatedSLM
- **Hardware-ready**: All fixtures support real hardware via environment variables

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
pytest

# Windows (PowerShell)
$env:SLMSUITE_TEST_SLM_CLASS="slmsuite.hardware.slms.thorlabs.ThorlabsSLM"
$env:SLMSUITE_TEST_SLM_ARGS='{"monitor_id": 1}'
pytest

# Windows (CMD)
set SLMSUITE_TEST_SLM_CLASS=slmsuite.hardware.slms.thorlabs.ThorlabsSLM
set SLMSUITE_TEST_SLM_ARGS={"monitor_id": 1}
pytest
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

## Logging and Plot Capture

### Output Directory Structure

Each pytest run creates a **timestamped output directory**:

```
tests/output/
├── 20250126_143015/         # First test run (YYYYMMDD_HHMMSS)
│   ├── pytest.log
│   └── test_*.png
├── 20250126_154522/         # Second test run
│   ├── pytest.log
│   └── test_*.png
└── latest -> 20250126_154522/  # Symlink to latest (if supported)
```

This allows you to:
- Compare outputs from different test runs
- Keep history of test results
- Isolate each run's artifacts

### Test Logging

All tests **automatically log** to `tests/output/{timestamp}/pytest.log` with start/end markers:

```bash
# View logs in terminal during test run
pytest --log-cli --log-cli-level=INFO

# View logs after test run (latest)
cat tests/output/latest/pytest.log
tail -f tests/output/latest/pytest.log

# View logs from specific run
cat tests/output/20250126_143015/pytest.log
```

**Automatic Logging**: Every test automatically gets a logger (via `autouse=True` fixture). No need to add `test_logger` parameter unless you want to use it explicitly:

```python
# Option 1: Automatic logging (most tests)
def test_example():
    # Test automatically logs start/end
    result = my_function()
    assert result == expected

# Option 2: Explicit logging (when needed)
def test_complex_operation(test_logger):
    test_logger.info("Starting complex operation...")
    test_logger.debug(f"Parameters: {params}")
    result = complex_function(params)
    test_logger.info(f"Result: {result}")
    assert result > 0
```

**Third-party Logging**: All external packages are automatically set to WARNING level. Only `slmsuite` package logs at INFO level.

**Log Format:**
```
2025-01-26 14:30:15 [    INFO] [slmsuite.tests.test_algorithms.TestHologram.test_gs_converges] === START: holography/test_algorithms.py::TestHologram::test_gs_converges ===
2025-01-26 14:30:16 [    INFO] [slmsuite.holography.algorithms] Optimizing with method=GS, maxiter=20
2025-01-26 14:30:17 [    INFO] [slmsuite.tests.test_algorithms.TestHologram.test_gs_converges] === PASSED: holography/test_algorithms.py::TestHologram::test_gs_converges ===
```

### Matplotlib Plot Capture

Plots are **automatically saved** to the current run's directory:

```bash
# Enable plot saving (default)
pytest

# Disable plot saving
pytest --no-save-plots

# Example output:
# 📁 Test output directory: tests/output/20250126_143015
# 📊 Saved plot: tests/output/20250126_143015/test_algorithms_TestHologram_test_hologram_construction_fig1.png
```

**Filename Format:** `{module}_{class}_{function}_fig{N}.png`

Examples:
- `test_algorithms_TestHologram_test_gs_converges_fig1.png`
- `test_toolbox_test_blaze_basic_fig1.png` (no class)

Existing tests using `plt.show()` work without modification - plots are automatically saved instead of displayed.

**View saved plots:**
```bash
# Latest run
ls tests/output/latest/*.png
open tests/output/latest/*.png

# Specific run
ls tests/output/20250126_143015/*.png

# Compare two runs
diff tests/output/20250126_143015/pytest.log tests/output/20250126_154522/pytest.log
```

### Cleanup

```bash
# Remove old test runs (keep last 5)
cd tests/output && ls -t | tail -n +6 | xargs rm -rf

# Remove all output
rm -rf tests/output/
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