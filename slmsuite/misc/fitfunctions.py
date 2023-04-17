"""
Common fit functions.
"""

import numpy as np

def linear(x, m, b):
    r"""
    For fitting a line.

    .. math:: y(x) = mx + b

    Parameters
    ----------
    x : numpy.ndarray
        Some independent variable.
    m : float
        Slope of the line.
    b : float
        y-intercept of the line.

    Returns
    -------
    y : numpy.ndarray
        Line evaluated at all ``x``.
    """
    return m * x + b


def hyperbola(z, w0, z0, zr):
    r"""
    For fitting a hyperbola.

    .. math:: w(z) = w_0 \sqrt{1 + \left[\frac{z - z_0}{z_R}\right]^2}.

    Parameters
    ----------
    z : numpy.ndarray
        Distance.
    w0 : float
        Beamradius at :math:`z = z_0`.
    z0 : float
        Plane of focus, the center of the hyperbola.
    zr : float
        Rayleigh length, the depth of focus.

    Returns
    -------
    w : numpy.ndarray
        Hyperbola evaluated at all ``z``.
    """
    return w0 * np.sqrt(1 + np.square((z - z0) / zr))


def cos(x, b, a, c, k=1):
    r"""
    For fitting an offset sinusoid.

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


def lorentzian(x, x0, a, c, Q):
    r"""
    For fitting an offset resonance.

    .. math:: y(x) = c + \frac{a}{1 + \left[\frac{x - x_0}{x_0/2Q}\right]^2}.

    :math:`Q` is the quality factor of the resonance.

    Parameters
    ----------
    x : numpy.ndarray
        Points to fit upon.
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
    Jacobian of :meth:`lorentzian`.

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


def gaussian(x, x0, a, c, w):
    r"""
    For fitting a 1D Gaussian.

    .. math:: y(x) = c + a \exp \left[\frac{(x-x_0)^2}{2w^2}\right].

    Parameters
    ----------
    x : numpy.ndarray
        Points to fit upon.
    x0 : float
        Positional offset.
    a : float
        Amplitude.
    c : float
        constant offset.
    w : float
        The standard deviation of the normal distribution.
        Equivalent to the :math:`1/e` radius.
        This is related to the full width at half maximum (FWHM)
        by a factor of :math:`2\sqrt{2\ln{2}}`.

    Returns
    -------
    y : numpy.ndarray
        Gaussian fit evaluated at all ``x``.
    """
    return c + a * np.exp(-.5 * np.square((x - x0) * (1/w)))


def gaussian2d(xy, x0, y0, a, c, wx, wy, wxy=0):
    r"""
    For fitting a 2D Gaussian.

    When the shear variance ``wxy`` (equivalent to :math:`M_{11}`;
    see :meth:`~slmsuite.holography.analysis.image_moment()`) is zero,

    .. math:: z(x,y) = c + a \exp \left[
                                \frac{(x-x_0)^2}{2w_x^2} +
                                \frac{(y-y_0)^2}{2w_y^2}
                                \right].

    When ``wxy`` is nonzero, we want to find the Gaussian which will have second
    order central moments (equivalent to variance;
    see :meth:`~slmsuite.holography.analysis.image_variances()`) satisfying:

    .. math::   M =
                \begin{bmatrix}
                    M_{20} & M_{11} \\
                    M_{11} & M_{02}
                \end{bmatrix}
                =
                \begin{bmatrix}
                    w_x^2 & w_{xy} \\
                    w_{xy} & w_y^2
                \end{bmatrix}.

    The equation satisfying this condition is:

    .. math:: z(x,y) = c + a \exp \left[
                                -\frac{1}{2}\left(
                                K_{00}(x-x_0)^2 +
                                2*K_{10}(x-x_0)(y-y_0) +
                                K_{11}(y-y_0)^2
                                \right)
                                \right].

    Where

    .. math:: K =
                \begin{bmatrix}
                    K_{00} & K_{10} \\
                    K_{10} & K_{11}
                \end{bmatrix}
                = M^{-1}.

    Caution
    ~~~~~~~
    The shear variance :math:`w_{xy}` does not have the same units as the widths
    :math:`w_x` and :math:`w_y`.

    Note
    ~~~~
    The shear variance ``wxy`` is currently bounded to magnitudes below ``wx*wy``.
    Higher values lead to solutions which cannot be normalized.
    When ``wxy = wx*wy``, this distribution is a line (an ellipse with zeroed minor
    axis).

    Parameters
    ----------
    xy : numpy.ndarray
        Points to fit upon. ``(x, y)`` form.
    x0, y0 : float
        Vector offset.
    a : float
        Amplitude.
    c : float
        constant offset.
    wx, wy : float
        The standard deviation of the normal distribution.
        Equivalent to the :math:`1/e` radius.
        This is related to the full width at half maximum (FWHM)
        by a factor of :math:`2\sqrt{2\ln{2}}`.
    wxy : float
        Shear variance. See above.

    Returns
    -------
    z : numpy.ndarray
        Gaussian fit evaluated at all ``(x,y)`` in ``xy``.
    """
    x = xy[0] - x0
    y = xy[1] - y0

    wxy = np.sign(wxy) * np.min([np.abs(wxy), wx*wy])

    try:
        M = np.linalg.inv([[wx*wx, wxy], [wxy, wy*wy]])
    except np.linalg.LinAlgError:
        M = np.array([[1/wx/wx, 0], [0, 1/wy/wy]])

    argument = np.square(x) * M[0,0] + np.square(y) * M[1,1] + 2 * x * y * M[1,0]

    return c + a * np.exp(-.5 * argument)


def tophat2d(xy, x0, y0, r, a=1):
    r"""
    For fitting a 2D tophat distribution.

    Parameters
    ----------
    xy : numpy.ndarray
        Points to fit upon (x, y).
    x0, y0 : float
        Vector offset.
    r : float
        Active radius of the tophat.
    a : float
        Amplitude.

    Returns
    -------
    z : numpy.ndarray
        Tophat fit evaluated at all ``(x,y)`` in ``xy``.
    """
    x = xy[0] - x0
    y = xy[1] - y0
    return np.where(x ** 2 + y ** 2 <= r ** 2, a, 0)

