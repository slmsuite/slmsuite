import numpy as np
from typing import TypeAlias


try:
    import cupy as cp
except ImportError:
    class cp:
        ndarray = np.ndarray
else:
    ArrayLike: TypeAlias = list | tuple | np.ndarray | cp.ndarray
    NDArrayLike: TypeAlias = np.ndarray | cp.ndarray
