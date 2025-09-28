# Technical Implementation Guide for slmsuite Improvements

This document provides specific technical details and implementation strategies for the 10 major improvements identified for the slmsuite package.

## 1. Comprehensive Testing Infrastructure and CI/CD Pipeline

### Current Issues Identified:
- Only 1 test file exists: `testing/hardware/cameras/alliedvision.py`
- No pytest configuration or CI/CD setup
- Missing test fixtures for hardware simulation
- No performance benchmarking or regression tests

### Detailed Implementation Plan:

#### Phase 1: Basic Testing Framework
```yaml
# .github/workflows/tests.yml
name: Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ${{ matrix.os }}
    strategy:
      matrix:
        os: [ubuntu-latest, windows-latest, macos-latest]
        python-version: [3.8, 3.9, "3.10", "3.11"]
    steps:
      - uses: actions/checkout@v4
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: ${{ matrix.python-version }}
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -e .[test]
      - name: Run tests
        run: pytest --cov=slmsuite --cov-report=xml
      - name: Upload coverage
        uses: codecov/codecov-action@v3
```

#### Phase 2: Hardware Simulation Framework
```python
# tests/fixtures/hardware.py
import pytest
import numpy as np
from slmsuite.hardware.slms.simulated import SimulatedSLM
from slmsuite.hardware.cameras.simulated import SimulatedCamera

@pytest.fixture
def mock_slm():
    """Fixture providing a simulated SLM for testing."""
    return SimulatedSLM(
        resolution=(1920, 1080),
        pitch_um=(6.4, 6.4),
        bitdepth=8
    )

@pytest.fixture
def mock_camera():
    """Fixture providing a simulated camera for testing."""
    return SimulatedCamera(
        resolution=(1024, 1024),
        pitch_um=(5.2, 5.2),
        bitdepth=12
    )

@pytest.fixture
def hologram_test_data():
    """Generate test data for holography algorithms."""
    target = np.zeros((100, 100))
    target[40:60, 40:60] = 1  # Square target
    return {"target": target, "source": np.ones((100, 100))}
```

#### Phase 3: Algorithm Testing
```python
# tests/test_holography.py
import pytest
import numpy as np
from slmsuite.holography.algorithms import Hologram

class TestHolographyAlgorithms:
    def test_gs_convergence(self, hologram_test_data):
        """Test Gerchberg-Saxton convergence."""
        hologram = Hologram(
            target=hologram_test_data["target"],
            source=hologram_test_data["source"]
        )
        
        initial_efficiency = hologram.stats["efficiency"]
        hologram.optimize(method="GS", maxiter=10)
        final_efficiency = hologram.stats["efficiency"]
        
        assert final_efficiency > initial_efficiency
        assert hologram.phase.shape == hologram_test_data["source"].shape

    @pytest.mark.parametrize("method", ["GS", "WGS", "WGS-Kim"])
    def test_optimization_methods(self, hologram_test_data, method):
        """Test different optimization methods."""
        hologram = Hologram(
            target=hologram_test_data["target"],
            source=hologram_test_data["source"]
        )
        
        hologram.optimize(method=method, maxiter=5)
        
        assert hasattr(hologram, "phase")
        assert hologram.stats["efficiency"] > 0
        assert hologram.stats["efficiency"] <= 1
```

### Testing Metrics to Track:
- Code coverage (target: >80%)
- Algorithm convergence rates
- Memory usage patterns
- GPU utilization efficiency
- Hardware communication reliability

---

## 2. Modern Python Packaging and Development Standards

### Current Issues:
- Legacy `setup.py` configuration
- Missing development dependencies
- No code formatting standards
- No type hints or static analysis

### Implementation Strategy:

#### Complete pyproject.toml Configuration:
```toml
[build-system]
requires = ["setuptools>=61.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "slmsuite"
version = "0.4.0"
description = "High-performance spatial light modulator control and holography"
readme = "README.md"
license = {file = "LICENSE"}
authors = [
    {name = "slmsuite developers", email = "qp-slm@mit.edu"}
]
classifiers = [
    "Development Status :: 4 - Beta",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Topic :: Scientific/Engineering :: Physics",
]
requires-python = ">=3.8"
dependencies = [
    "numpy>=1.20.0",
    "scipy>=1.7.0",
    "opencv-python>=4.5.0",
    "matplotlib>=3.5.0",
    "h5py>=3.0.0",
    "tqdm>=4.60.0",
]

[project.optional-dependencies]
gpu = ["cupy>=10.0.0"]
hardware = [
    "pyglet>=1.5.0",
    "imageio>=2.9.0",
    "pyav>=8.0.0",
]
dev = [
    "pytest>=7.0.0",
    "pytest-cov>=4.0.0",
    "pytest-xdist>=3.0.0",
    "black>=22.0.0",
    "isort>=5.10.0",
    "flake8>=5.0.0",
    "mypy>=1.0.0",
    "pre-commit>=2.20.0",
]
docs = [
    "sphinx>=5.0.0",
    "sphinx-autodoc-typehints>=1.19.0",
    "sphinx-design>=0.3.0",
    "nbsphinx>=0.8.0",
]
all = ["slmsuite[gpu,hardware,dev,docs]"]

[project.urls]
Homepage = "https://github.com/slmsuite/slmsuite"
Documentation = "https://slmsuite.readthedocs.io/"
Repository = "https://github.com/slmsuite/slmsuite.git"
Issues = "https://github.com/slmsuite/slmsuite/issues"

# Tool configurations
[tool.black]
line-length = 88
target-version = ['py38', 'py39', 'py310', 'py311']
include = '\.pyi?$'
extend-exclude = '''
/(
  # directories
  \.eggs
  | \.git
  | \.mypy_cache
  | \.tox
  | \.venv
  | build
  | dist
)/
'''

[tool.isort]
profile = "black"
multi_line_output = 3
line_length = 88
known_first_party = ["slmsuite"]

[tool.mypy]
python_version = "3.8"
warn_return_any = true
warn_unused_configs = true
ignore_missing_imports = true
disallow_untyped_defs = false  # Start permissive, gradually tighten

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py", "*_test.py"]
python_functions = ["test_*"]
addopts = [
    "--strict-markers",
    "--disable-warnings",
    "--cov=slmsuite",
    "--cov-report=term-missing",
    "--cov-report=html",
    "--cov-report=xml"
]
markers = [
    "slow: marks tests as slow (deselect with '-m \"not slow\"')",
    "gpu: marks tests requiring GPU acceleration",
    "hardware: marks tests requiring physical hardware",
]
```

#### Pre-commit Configuration:
```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.4.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-added-large-files
      - id: check-merge-conflict

  - repo: https://github.com/psf/black
    rev: 22.12.0
    hooks:
      - id: black
        language_version: python3

  - repo: https://github.com/pycqa/isort
    rev: 5.11.4
    hooks:
      - id: isort
        args: ["--profile", "black"]

  - repo: https://github.com/pycqa/flake8
    rev: 6.0.0
    hooks:
      - id: flake8
        args: [--max-line-length=88, --extend-ignore=E203,W503]

  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v0.991
    hooks:
      - id: mypy
        additional_dependencies: [types-all]
```

---

## 3. Enhanced Error Handling and Logging System

### Current Issues:
- Inconsistent use of print statements vs proper logging
- Limited error context in hardware failures
- No structured error recovery mechanisms

### Implementation Strategy:

#### Logging System Architecture:
```python
# slmsuite/misc/logging.py
import logging
import sys
from typing import Optional
from pathlib import Path

class SLMSuiteFormatter(logging.Formatter):
    """Custom formatter for slmsuite logging."""
    
    COLORS = {
        logging.DEBUG: '\033[36m',    # Cyan
        logging.INFO: '\033[32m',     # Green  
        logging.WARNING: '\033[33m',  # Yellow
        logging.ERROR: '\033[31m',    # Red
        logging.CRITICAL: '\033[35m', # Magenta
    }
    
    def format(self, record):
        if sys.stderr.isatty():  # Only colorize if terminal
            color = self.COLORS.get(record.levelno, '')
            reset = '\033[0m' if color else ''
            record.levelname = f"{color}{record.levelname}{reset}"
        
        return super().format(record)

def setup_logging(level: str = "INFO", log_file: Optional[Path] = None):
    """Configure logging for slmsuite."""
    logger = logging.getLogger("slmsuite")
    logger.setLevel(getattr(logging, level.upper()))
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(
        SLMSuiteFormatter(
            fmt='[%(asctime)s] %(name)s.%(levelname)s: %(message)s',
            datefmt='%H:%M:%S'
        )
    )
    logger.addHandler(console_handler)
    
    # File handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(
            logging.Formatter(
                fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
        )
        logger.addHandler(file_handler)
    
    return logger
```

#### Hardware Error Classes:
```python
# slmsuite/hardware/exceptions.py
class SLMSuiteHardwareError(Exception):
    """Base exception for hardware-related errors."""
    pass

class SLMConnectionError(SLMSuiteHardwareError):
    """SLM connection or communication error."""
    
    def __init__(self, device_id: str, message: str, recovery_hint: str = None):
        self.device_id = device_id
        self.recovery_hint = recovery_hint
        
        full_message = f"SLM {device_id}: {message}"
        if recovery_hint:
            full_message += f"\nSuggested fix: {recovery_hint}"
        
        super().__init__(full_message)

class CameraError(SLMSuiteHardwareError):
    """Camera operation error."""
    
    def __init__(self, camera_type: str, error_code: int = None, message: str = ""):
        self.camera_type = camera_type
        self.error_code = error_code
        
        full_message = f"{camera_type} camera error"
        if error_code:
            full_message += f" (code {error_code})"
        if message:
            full_message += f": {message}"
            
        super().__init__(full_message)

class CalibrationError(SLMSuiteHardwareError):
    """Calibration procedure error."""
    pass
```

#### Error Recovery Mechanisms:
```python
# slmsuite/hardware/recovery.py
import logging
import time
from typing import Callable, Any, Optional
from functools import wraps

logger = logging.getLogger(__name__)

def with_retry(max_attempts: int = 3, delay: float = 1.0, 
               exceptions: tuple = (Exception,)):
    """Decorator for automatic retry of failed operations."""
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            last_exception = None
            
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_attempts - 1:
                        logger.warning(
                            f"{func.__name__} failed (attempt {attempt + 1}): {e}. "
                            f"Retrying in {delay}s..."
                        )
                        time.sleep(delay)
                    else:
                        logger.error(
                            f"{func.__name__} failed after {max_attempts} attempts"
                        )
            
            raise last_exception
        return wrapper
    return decorator

class HardwareRecovery:
    """Handles hardware connection recovery."""
    
    @staticmethod
    @with_retry(max_attempts=3, delay=2.0)
    def reconnect_slm(slm_instance):
        """Attempt to reconnect to SLM."""
        logger.info(f"Attempting to reconnect SLM {slm_instance.name}")
        slm_instance._initialize_connection()
        slm_instance._verify_connection()
        logger.info("SLM reconnection successful")
        
    @staticmethod
    def graceful_shutdown(hardware_instances: list):
        """Safely shut down hardware connections."""
        for hw in hardware_instances:
            try:
                if hasattr(hw, 'close'):
                    hw.close()
                logger.info(f"Successfully closed {hw.__class__.__name__}")
            except Exception as e:
                logger.error(f"Error closing {hw.__class__.__name__}: {e}")
```

---

## 4. Performance Optimization and Memory Management

### Current Issues:
- GPU memory not properly managed in holography algorithms
- Large arrays not cleaned up after processing
- No memory usage monitoring or optimization

### Implementation Strategy:

#### GPU Memory Management:
```python
# slmsuite/holography/gpu_memory.py
import gc
import contextlib
from typing import Optional, Generator
import logging

logger = logging.getLogger(__name__)

try:
    import cupy as cp
    import cupyx
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False
    cp = None

class GPUMemoryManager:
    """Manages GPU memory allocation and cleanup."""
    
    def __init__(self, memory_pool_size: Optional[int] = None):
        if not HAS_CUPY:
            raise ImportError("CuPy is required for GPU memory management")
        
        self.memory_pool = cp.get_default_memory_pool()
        self.pinned_memory_pool = cp.get_default_pinned_memory_pool()
        
        if memory_pool_size:
            self.memory_pool.set_limit(size=memory_pool_size)
    
    @contextlib.contextmanager
    def managed_memory(self) -> Generator[None, None, None]:
        """Context manager for automatic GPU memory cleanup."""
        initial_memory = self.memory_pool.used_bytes()
        
        try:
            yield
        finally:
            # Force cleanup
            cp._default_memory_pool.free_all_blocks()
            cp._default_pinned_memory_pool.free_all_blocks()
            gc.collect()
            
            final_memory = self.memory_pool.used_bytes()
            logger.debug(
                f"GPU memory: {initial_memory / 1e6:.1f} MB -> "
                f"{final_memory / 1e6:.1f} MB"
            )
    
    def get_memory_info(self) -> dict:
        """Get current GPU memory usage statistics."""
        if not HAS_CUPY:
            return {"available": False}
            
        meminfo = cp.cuda.runtime.memGetInfo()
        total_memory = meminfo[1]
        free_memory = meminfo[0]
        used_memory = total_memory - free_memory
        
        return {
            "available": True,
            "total_gb": total_memory / 1e9,
            "used_gb": used_memory / 1e9,
            "free_gb": free_memory / 1e9,
            "utilization": used_memory / total_memory,
            "pool_used_gb": self.memory_pool.used_bytes() / 1e9,
        }

# Enhanced holography algorithms with memory management
class MemoryOptimizedHologram:
    """Hologram class with optimized memory usage."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gpu_manager = GPUMemoryManager() if HAS_CUPY else None
        self._chunked_processing = True
        self._max_chunk_size = 512  # MB
    
    def optimize_chunked(self, method="GS", maxiter=20, **kwargs):
        """Memory-efficient optimization using chunked processing."""
        if not self.gpu_manager:
            return self.optimize(method=method, maxiter=maxiter, **kwargs)
        
        # Estimate memory requirements
        array_size = self.nearfield.nbytes
        if array_size > self._max_chunk_size * 1e6:
            return self._optimize_with_chunks(method, maxiter, **kwargs)
        else:
            with self.gpu_manager.managed_memory():
                return self.optimize(method=method, maxiter=maxiter, **kwargs)
    
    def _optimize_with_chunks(self, method, maxiter, **kwargs):
        """Process large arrays in chunks to manage memory usage."""
        # Implementation for chunked processing
        pass
```

#### Performance Monitoring:
```python
# slmsuite/misc/profiling.py
import time
import psutil
import functools
from typing import Dict, Any, Callable
import logging

logger = logging.getLogger(__name__)

class PerformanceMonitor:
    """Monitor performance metrics for optimization algorithms."""
    
    def __init__(self):
        self.metrics = {}
        self.start_time = None
        self.peak_memory = 0
    
    def start_monitoring(self):
        """Start performance monitoring."""
        self.start_time = time.time()
        self.peak_memory = psutil.Process().memory_info().rss
        
    def record_iteration(self, iteration: int, efficiency: float, **kwargs):
        """Record metrics for an optimization iteration."""
        current_memory = psutil.Process().memory_info().rss
        self.peak_memory = max(self.peak_memory, current_memory)
        
        if iteration not in self.metrics:
            self.metrics[iteration] = {}
            
        self.metrics[iteration].update({
            'efficiency': efficiency,
            'memory_mb': current_memory / 1e6,
            'time_s': time.time() - self.start_time,
            **kwargs
        })
    
    def get_summary(self) -> Dict[str, Any]:
        """Get performance summary."""
        if not self.metrics:
            return {}
            
        efficiencies = [m['efficiency'] for m in self.metrics.values()]
        times = [m['time_s'] for m in self.metrics.values()]
        
        return {
            'total_iterations': len(self.metrics),
            'final_efficiency': efficiencies[-1],
            'efficiency_improvement': efficiencies[-1] - efficiencies[0],
            'total_time_s': times[-1],
            'peak_memory_mb': self.peak_memory / 1e6,
            'avg_iteration_time_s': times[-1] / len(self.metrics),
        }

def profile_function(func: Callable) -> Callable:
    """Decorator to profile function performance."""
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss
        
        try:
            result = func(*args, **kwargs)
            return result
        finally:
            end_time = time.time()
            end_memory = psutil.Process().memory_info().rss
            
            logger.debug(
                f"{func.__name__} completed in {end_time - start_time:.3f}s, "
                f"memory delta: {(end_memory - start_memory) / 1e6:.1f} MB"
            )
    
    return wrapper
```

---

## 5. Extensible Plugin Architecture for Hardware

### Current Issues:
- Hardware classes are tightly coupled
- No standardized interface for new hardware
- Difficult to add support for new devices

### Implementation Strategy:

#### Plugin Interface Design:
```python
# slmsuite/hardware/plugin_system.py
import abc
import importlib
import pkgutil
from typing import Dict, List, Type, Optional, Any
import logging

logger = logging.getLogger(__name__)

class HardwarePlugin(abc.ABC):
    """Abstract base class for hardware plugins."""
    
    # Plugin metadata
    NAME: str = ""
    MANUFACTURER: str = ""
    SUPPORTED_MODELS: List[str] = []
    REQUIRES_SDK: bool = False
    SDK_REQUIREMENTS: List[str] = []
    
    @abc.abstractmethod
    def __init__(self, **config):
        """Initialize hardware plugin with configuration."""
        pass
    
    @abc.abstractmethod
    def connect(self) -> bool:
        """Connect to hardware device."""
        pass
    
    @abc.abstractmethod
    def disconnect(self) -> bool:
        """Disconnect from hardware device."""
        pass
    
    @abc.abstractmethod
    def is_connected(self) -> bool:
        """Check if device is connected."""
        pass
    
    @abc.abstractmethod
    def get_capabilities(self) -> Dict[str, Any]:
        """Return device capabilities."""
        pass
    
    @classmethod
    def check_requirements(cls) -> bool:
        """Check if plugin requirements are met."""
        if not cls.REQUIRES_SDK:
            return True
            
        for requirement in cls.SDK_REQUIREMENTS:
            try:
                importlib.import_module(requirement)
            except ImportError:
                logger.warning(f"Missing requirement for {cls.NAME}: {requirement}")
                return False
        return True

class SLMPlugin(HardwarePlugin):
    """Base class for SLM plugins."""
    
    @abc.abstractmethod
    def write_phase(self, phase: np.ndarray) -> bool:
        """Write phase pattern to SLM."""
        pass
    
    @abc.abstractmethod
    def get_resolution(self) -> tuple:
        """Get SLM resolution."""
        pass
    
    @abc.abstractmethod
    def get_pixel_pitch(self) -> tuple:
        """Get SLM pixel pitch in micrometers."""
        pass

class CameraPlugin(HardwarePlugin):
    """Base class for camera plugins."""
    
    @abc.abstractmethod
    def capture_image(self) -> np.ndarray:
        """Capture single image."""
        pass
    
    @abc.abstractmethod
    def start_continuous(self, callback=None):
        """Start continuous image capture."""
        pass
    
    @abc.abstractmethod
    def stop_continuous(self):
        """Stop continuous image capture."""
        pass
    
    @abc.abstractmethod
    def set_exposure(self, exposure_ms: float):
        """Set camera exposure time."""
        pass

class PluginRegistry:
    """Registry for hardware plugins."""
    
    def __init__(self):
        self.slm_plugins: Dict[str, Type[SLMPlugin]] = {}
        self.camera_plugins: Dict[str, Type[CameraPlugin]] = {}
        self._discovered = False
    
    def discover_plugins(self, package_paths: List[str] = None):
        """Discover available hardware plugins."""
        if package_paths is None:
            package_paths = [
                'slmsuite.hardware.slms',
                'slmsuite.hardware.cameras'
            ]
        
        for package_path in package_paths:
            try:
                package = importlib.import_module(package_path)
                for _, name, _ in pkgutil.iter_modules(package.__path__):
                    module_name = f"{package_path}.{name}"
                    try:
                        module = importlib.import_module(module_name)
                        self._register_plugins_from_module(module)
                    except Exception as e:
                        logger.debug(f"Could not load plugin module {module_name}: {e}")
            except ImportError:
                logger.debug(f"Package {package_path} not found")
        
        self._discovered = True
        logger.info(f"Discovered {len(self.slm_plugins)} SLM and "
                   f"{len(self.camera_plugins)} camera plugins")
    
    def _register_plugins_from_module(self, module):
        """Register plugins from a module."""
        for attr_name in dir(module):
            attr = getattr(module, attr_name)
            if (isinstance(attr, type) and 
                issubclass(attr, HardwarePlugin) and 
                attr != HardwarePlugin):
                
                if issubclass(attr, SLMPlugin) and attr.check_requirements():
                    self.slm_plugins[attr.NAME] = attr
                elif issubclass(attr, CameraPlugin) and attr.check_requirements():
                    self.camera_plugins[attr.NAME] = attr
    
    def get_available_slms(self) -> Dict[str, Type[SLMPlugin]]:
        """Get available SLM plugins."""
        if not self._discovered:
            self.discover_plugins()
        return self.slm_plugins.copy()
    
    def get_available_cameras(self) -> Dict[str, Type[CameraPlugin]]:
        """Get available camera plugins."""
        if not self._discovered:
            self.discover_plugins()
        return self.camera_plugins.copy()
    
    def create_slm(self, plugin_name: str, **config) -> Optional[SLMPlugin]:
        """Create SLM instance from plugin."""
        if plugin_name in self.slm_plugins:
            return self.slm_plugins[plugin_name](**config)
        else:
            raise ValueError(f"Unknown SLM plugin: {plugin_name}")
    
    def create_camera(self, plugin_name: str, **config) -> Optional[CameraPlugin]:
        """Create camera instance from plugin."""
        if plugin_name in self.camera_plugins:
            return self.camera_plugins[plugin_name](**config)
        else:
            raise ValueError(f"Unknown camera plugin: {plugin_name}")

# Global plugin registry
hardware_registry = PluginRegistry()
```

#### Configuration System:
```yaml
# Example hardware_config.yaml
slms:
  primary:
    plugin: "santec_slm200"
    config:
      slm_number: 1
      display_number: 2
      wav_um: 1.064
      pitch_um: [8.0, 8.0]
  
  secondary:
    plugin: "holoeye_leto"
    config:
      display_id: "LETO-12345"
      resolution: [1920, 1080]

cameras:
  imaging:
    plugin: "allied_vision_mako"
    config:
      serial_number: "DEV_000F31234567"
      pixel_format: "Mono12"
      exposure_us: 10000
  
  wavefront:
    plugin: "thorlabs_zelux"
    config:
      serial_number: "12345678"
      roi: [100, 100, 400, 400]
```

---

*[Document continues with implementations for remaining improvements...]*

## Summary of Implementation Benefits

These technical implementations provide:

1. **Robust Foundation**: Comprehensive testing and modern packaging ensure reliability
2. **Developer Productivity**: Pre-commit hooks, type checking, and structured logging improve development experience  
3. **Performance**: GPU memory management and profiling tools enable high-performance computing
4. **Extensibility**: Plugin architecture allows easy addition of new hardware support
5. **User Experience**: Better error messages, configuration systems, and documentation

Each improvement builds upon the previous ones, creating a cohesive upgrade path that transforms slmsuite into a professional, maintainable, and extensible package suitable for both research and industrial applications.