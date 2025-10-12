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
Test files are located in the root directory with the pattern `test_*.py`. These are typically used to verify GPU acceleration, display functionality, and performance benchmarks:

```bash
# Run individual test files
python test_gpu_display.py
python test_gpu_vs_cpu.py
python test_pinned_memory.py
python test_cpu_plm.py
```

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
- Live through nbviewer
- In documentation at https://slmsuite.readthedocs.io/en/latest/examples.html

## Important Notes

- This is research-grade code used in quantum optics, atomic physics, and optical computing
- Performance is critical - prefer GPU-accelerated operations when possible
- Hardware compatibility varies - check vendor-specific driver requirements
- Wavefront calibration is essential for high-quality holography
- Zero-padding in Fourier space improves resolution but increases memory usage
