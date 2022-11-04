"""
Common definitions.
"""

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