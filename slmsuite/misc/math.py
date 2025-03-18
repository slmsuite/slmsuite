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

SCALAR_TYPES = (*REAL_TYPES, complex)


def iseven(x: int) -> bool:
    """
    Test if an integer is even.

    :param int x: The integer to test.
    :returns: Whether or not `x` is even.
    :rtype: bool
    """
    return bool(~(x & 0x1))
