"""
Common fit functions.
"""

import numpy as np


def cos_fitfun(x, b, a, c, k=1):
    r"""For fitting an offset sinusoid.

    .. math:: y(x) = c + \frac{a}{2} \left[1+\cos(kx+b) \right].

    Parameters
    ----------
    x : numpy.ndarray
        Phase in radians.
    b : float
        Phase offset.
    a : float
        Peak amplitude.
    c : float
        Amplitude offset.
    k : float
        Phase scale factor. Default is 1.

    Returns
    -------
    y : numpy.ndarray
        Cosine fit evaluated at all ``x``.
    """
    return a * 0.5 * (1 + np.cos(k * x - b)) + c


def lorentzian_fitfun(x, x0, a, c, Q):
    r"""
    For fitting an offset resonance.

    .. math:: y(x) = c + \frac{a}{1 + \left[\frac{x - x_0}{x_0/2Q}\right]^2}.

    :math:`Q` is the quality factor of the resonance.

    Parameters
    ----------
    x : numpy.ndarray
        Wavelength.
    x0 : float
        Center wavelength.
    a : float
        Amplitude.
    c : float
        constant offset.
    Q : float
        Quality factor.

    Returns
    -------
    y : numpy.ndarray
        Lorentzian fit evaluated at all ``x``.
    """
    return a / (1 + ((x - x0) / (x0 / Q / 2)) ** 2) + c

def lorentzian_jacobian(x, x0, a, c, Q):
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
    c : float
        constant offset.
    Q : float
        Quality factor.

    Returns
    -------
    gradf : numpy.ndarray
        Jacobian of Lorentzian fit evaluated at all ``x``.
    """
    return np.array(
        [
            a
            * 8
            * Q ** 2
            * (x - x0)
            / x0 ** 2
            * (1 + (x - x0) / x0)
            / (1 + 4 * (Q * (x - x0) / x0) ** 2) ** 2,
            1 / (1 + 4 * (Q * (x - x0) / x0) ** 2),
            -8
            * Q
            * a
            * ((x - x0) / x0) ** 2
            / (1 + 4 * (Q * (x - x0) / x0) ** 2) ** 2,
            1 - 1 / (1 + 4 * (Q * (x - x0) / x0) ** 2),
        ]
    ).T


def gaussian_fitfun(x, x0, a, c, w):
    r"""
    For fitting a 1d Gaussian.

    .. math:: y(x) = c + a \exp \left[\frac{(x-x_0)^2}{w^2}\right].

    w :float
        The standard deviation of the normal distribution. This is related to the
        full width at half maximum (FWHM) by a factor of :math:`2\sqrt{2\ln{2}}`.
    """
    return c + a * np.exp(-.5 * np.square((x - x0) * (1/w)))

def gaussian2d_fitfun(xy, x0, y0, a, c, wx, wy, wxy=0):
    r""""
    For fitting a 2d Gaussian. (Unfinished) Shear equation

    .. math:: z(x,y) = c + a \exp \left[
                                \frac{(x-x_0)^2}{w_x^2} +
                                \frac{(x-x_0)(y-y_0)}{w_{xy}} +
                                \frac{(y-y_0)^2}{w_y^2}
                                \right].

    Caution
    ~~~~~~~
    The shear variance :math:`w_{xy}` does not have the same units as the widths
    :math:`w_x` and :math:`w_y`.


    """
    x = xy[0] - x0
    y = xy[1] - y0

    # if wxy != 0:
    #     shear = -2 * wxy / (wx * wx * wy * wy - wxy * wxy)
    # else:
    #     shear = 0

    M = np.linalg.inv([[wx*wx, wxy], [wxy, wy*wy]])

    argument = np.square(x) * M[0,0] + np.square(y) * M[1,1] + 2 * x * y * M[1,0]

    return c + a * np.exp(-.5 * argument)