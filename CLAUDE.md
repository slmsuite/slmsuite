# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## About slmsuite

`slmsuite` is a Python package for high-performance spatial light modulator (SLM) control and holography. It combines GPU-accelerated beamforming algorithms with optimized hardware control, automated calibration, and user-friendly scripting to enable programmable optics.

## Development Commands

### Installation
```bash
# Install from local source
pip install -e .

# Install with IPython/Jupyter support
pip install -r requirements_ipython.txt
```

### Documentation
```bash
# Build documentation locally
cd docs
make html
# View at docs/_build/html/index.html
```

### Testing
```bash
# Run all tests (from slmsuite/ directory)
pytest

# Run tests for specific module
pytest tests/holography/
pytest tests/hardware/
pytest tests/misc/

# Run specific test file or function
pytest tests/holography/test_algorithms.py
pytest tests/holography/test_algorithms.py::TestHologram::test_gs_converges

# Run with options
pytest -v              # Verbose output
pytest -x              # Stop at first failure
pytest --lf            # Run last failed tests
pytest -m "not gpu"    # Skip GPU tests
pytest -m "not slow"   # Skip slow tests
```

**Test Structure**: Tests mirror the package structure with directories for `hardware/`, `holography/`, and `misc/`. All tests use simulated hardware by default (no physical SLM/camera required) via fixtures defined in `tests/conftest.py`.

**Key Fixtures** (from `conftest.py`):
- `slm`: Provides SimulatedSLM (configurable via env vars for real hardware)
- `camera`: Provides SimulatedCamera (configurable via env vars for real hardware)
- `temp_dir`: Temporary directory for test file I/O
- `random_phase`, `random_amplitude`: Test data patterns
- `has_cupy`: Boolean indicating GPU availability
- `test_logger`: Automatic per-test logger (autouse=True, optional parameter)
- `mpl_test`: Matplotlib fixture with automatic cleanup

**Test Markers**:
- `@pytest.mark.gpu`: Tests requiring CuPy/CUDA
- `@pytest.mark.slow`: Long-running tests (>5 seconds)

**Hardware Testing**: Fixtures support testing with real hardware via environment variables:
```bash
# Example: Test with real Thorlabs hardware
export SLMSUITE_TEST_SLM_CLASS=slmsuite.hardware.slms.thorlabs.ThorlabsSLM
export SLMSUITE_TEST_SLM_ARGS='{"monitor_id": 1}'
pytest
```

**Output Directory**: Each pytest run creates `tests/output/YYYYMMDD_HHMMSS/` with logs and plots. Latest run: `tests/output/latest/`

**Logging**: All tests automatically log start/end to `tests/output/latest/pytest.log`. Only slmsuite package (INFO+) and external packages (WARNING+) are logged. View during tests: `pytest --log-cli --log-cli-level=INFO`

**Plot Capture**: Matplotlib plots auto-saved to current run directory as `{test_name}_fig{N}.png` (disable: `--no-save-plots`)

See `tests/README.md` for comprehensive testing documentation.

## Architecture Overview

### Module Structure

The codebase is organized into three main modules:

1. **`slmsuite.hardware`** - Hardware interfaces for SLMs and cameras
2. **`slmsuite.holography`** - GPU-accelerated holography algorithms and analysis
3. **`slmsuite.misc`** - Utility functions (math, file I/O, fit functions)

### Hardware Module (`slmsuite/hardware/`)

Provides abstract base classes and vendor-specific implementations:

- **`slms/slm.py`**: Abstract `SLM` class defining the interface for all spatial light modulators
  - Key attributes: `shape`, `bitdepth`, `pitch_um`, `wav_um`, `phase`, `display`
  - Key methods: `set_phase()`, `display_phase()`, calibration routines
  - Subclasses implement vendor-specific drivers (Thorlabs, Holoeye, Hamamatsu, Texas Instruments, etc.)

- **`cameras/camera.py`**: Abstract `Camera` class for all camera interfaces
  - Key attributes: `shape`, `bitdepth`, `exposure_s`, `averaging`, `hdr`
  - Key methods: `get_image()`, `set_exposure()`, `autoexpose()`
  - Subclasses implement vendor-specific drivers (Basler, FLIR, Thorlabs, etc.)

- **`cameraslms.py`**: Combines SLMs with cameras for closed-loop feedback
  - `CameraSLM`: Base class linking an SLM with a camera
  - `FourierSLM`: Adds Fourier-space calibration and coordinate transformations
  - Enables automated wavefront calibration and camera-in-the-loop optimization

- **`_pyglet.py`**: Window management using pyglet for SLM display
  - Supports GPU-accelerated transfers with CuPy
  - Allocates pinned memory for optimized GPU→CPU DMA transfers
  - Used by `screenmirrored.py` and `texasinstruments.py` SLM implementations

### Holography Module (`slmsuite/holography/`)

Contains GPU-accelerated phase retrieval and holography algorithms:

- **`algorithms/`**: Core holography algorithms
  - Split across multiple internal files for clarity:
    - `_header.py`: Common imports, handles CuPy/NumPy fallback
    - `_hologram.py`: Base `Hologram` class with Gerchberg-Saxton (GS) and Weighted GS algorithms
    - `_feedback.py`: `FeedbackHologram` for camera-in-the-loop optimization
    - `_spots.py`: `SpotHologram` and `CompressedSpotHologram` for optical focus arrays
    - `_multiplane.py`: `MultiplaneHologram` for 3D holography
  - Algorithms include: GS, WGS variants (Leonardo, Kim, Nogrette, Wu, tanh), gradient descent
  - Supports Mixed Region Amplitude Freedom (MRAF) for ignoring unused regions

- **`toolbox/phase.py`**: Analytic phase pattern library
  - Functions: `blaze()`, `sinusoid()`, `zernike()`, `binary()`, etc.
  - Contains CUDA kernels loaded from `cuda.cu` for custom GPU operations
  - Used to generate gratings, lenses, aberrations, and structured light

- **`analysis/`**: Image analysis and statistics
  - Functions for centroid detection, spot analysis, Fourier transforms
  - File I/O utilities (HDF5 save/load)

### GPU Acceleration Strategy

The codebase uses **CuPy** for GPU acceleration with NumPy fallback:
- Import pattern: `try: import cupy as cp` with `except: cp = np`
- Algorithms automatically detect and use GPU when available
- Critical for performance in iterative phase retrieval (100x+ speedup)
- Pinned memory allocation (`cupyx.zeros_pinned`) used for efficient GPU↔CPU transfers

## Key Design Patterns

### Abstract Base Classes
All hardware drivers inherit from abstract base classes (`SLM`, `Camera`) that define standard interfaces. This allows algorithms to work with any vendor's hardware through a unified API.

### Shape Convention
Arrays follow NumPy's `(height, width)` convention. The holography algorithms distinguish between:
- `slm_shape`: Physical SLM dimensions
- `shape`: Computational space (often larger due to zero-padding for improved FFT resolution)

### Phase vs. Display
SLM classes maintain two representations:
- `phase`: Continuous phase in radians (what the user sets)
- `display`: Discrete gray levels (what hardware actually displays)
- Conversion handles `bitdepth`, `wav_um`, `wav_design_um`, and wavefront correction

### Calibration Data Storage
Calibration results (wavefront, Fourier transforms, etc.) are stored in `source` dictionaries and saved/loaded via HDF5 for reproducibility.

## Code Style

- Follows **Black** code formatting
- Type hints are used selectively, primarily for documentation
- Extensive docstrings with NumPy/Google style
- Abstract methods marked with `@abstractmethod` decorator
- Private/internal methods prefixed with `_`

## Working with Examples

Examples are maintained in a separate repository (`slmsuite-examples`) and can be viewed:
- In documentation at https://slmsuite.readthedocs.io/en/latest/examples.html

## Important Notes

- This is research-grade code used in quantum optics, atomic physics, and optical computing
- Performance is critical - prefer GPU-accelerated operations when possible
- Hardware compatibility varies - check vendor-specific driver requirements
- Wavefront calibration is essential for high-quality holography
- Zero-padding in Fourier space improves resolution but increases memory usage
