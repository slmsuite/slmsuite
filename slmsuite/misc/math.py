"""
Common definitions.
"""

import numpy as np

INTEGER_TYPES = (
    int,
    np.integer,
)

FLOAT_TYPES = (
    float,
    np.floating,
)

REAL_TYPES = (
    *INTEGER_TYPES,
    *FLOAT_TYPES,
)

SCALAR_TYPES =  (
    *REAL_TYPES,
    complex
)

def iseven(x):
    """
    Test if an integer is even.

    Parameters
    ----------
    x : int
        The integer to test.

    Returns
    -------
    bool
        Whether or not `x` is even.
    """
    return bool(~(x & 0x1))
