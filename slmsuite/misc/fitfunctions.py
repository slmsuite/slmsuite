"""
Common fit functions.
"""

import numpy as np


def cos_fitfun(x, a, b, c, k=1):
    r"""Offset sinusoid for fitting:  :math:`y(x)=\frac{a}{2} \left[1+\cos(kx+b) \right]+c`.
    Parameters
    ----------
    x : numpy.ndarray
        Phase in radians.
    a : float
        Peak amplitude.
    b : float
        Phase offset.
    c : float
        Amplitude offset.
    k : float
        Phase scale factor. Default is 1.

    Returns
    -------
    y : numpy.ndarray
        Cosine fit evaluated at all `x`.
    """
    return a * 0.5 * (1 + np.cos(k * x - b)) + c


def lorentzian_fitfun(x, x0, a, Q, c):
    r"""
    For fitting:  :math:`y(x)=\frac{a - c}{1 + [\frac{x - x_0}{x_0/2Q}]^2} + c`.

    Parameters
    ----------
    x : numpy.ndarray
        Wavelength.
    x0 : float
        Center wavelength.
    a : float
        Amplitude.
    Q : float
        Quality factor.
    c : float
        constant offset.

    Returns
    -------
    y : numpy.ndarray
        Lorentzian fit evaluated at all `x`.
    """
    return (a - c) / (1 + ((x - x0) / (x0 / Q / 2)) ** 2) + c


def lorentzian_jacobian(x, x0, a, Q, c):
    """
    Jacobian of :meth:`lorentzian_fitfun`.

    Parameters
    ----------
    x : numpy.ndarray
        Wavelength.
    x0 : float
        Center wavelength.
    a : float
        Amplitude.
    Q : float
        Quality factor.
    c : float
        constant offset.

    Returns
    -------
    gradf : numpy.ndarray
        Jacobian of Lorentzian fit evaluated at all `x`.
    """
    return np.array(
        [
            (a - c)
            * 8
            * Q ** 2
            * (x - x0)
            / x0 ** 2
            * (1 + (x - x0) / x0)
            / (1 + 4 * (Q * (x - x0) / x0) ** 2) ** 2,
            1 / (1 + 4 * (Q * (x - x0) / x0) ** 2),
            -8
            * Q
            * (a - c)
            * ((x - x0) / x0) ** 2
            / (1 + 4 * (Q * (x - x0) / x0) ** 2) ** 2,
            1 - 1 / (1 + 4 * (Q * (x - x0) / x0) ** 2),
        ]
    ).T
