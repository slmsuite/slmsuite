"""
Repository of common analytic phase patterns.
"""
import os
import numpy as np
try:
    import cupy as cp
except ImportError:
    cp = np
from scipy import special
from math import factorial
import matplotlib.pyplot as plt

from slmsuite.misc.math import REAL_TYPES
from slmsuite.holography.toolbox import _process_grid

# Load CUDA code. This is used for cupy.RawKernels in this file and elsewhere.

with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "cuda.cu"), 'r') as file:
    CUDA_KERNELS = file.read()

# Basic gratings.

def blaze(grid, vector=(0, 0), offset=0):
    r"""
    Returns a simple `blazed grating <https://en.wikipedia.org/wiki/Blazed_grating>`_,
    a linear phase ramp, toward a given vector in :math:`k`-space.

    .. math:: \phi(\vec{x}) = 2\pi \cdot \vec{k}_{norm} \cdot \vec{x}_{norm} + o

    Parameters
    ----------
    grid : (array_like, array_like) OR :class:`~slmsuite.hardware.slms.slm.SLM`
        :math:`\vec{x}_{norm}`. Meshgrids of normalized :math:`\frac{x}{\lambda}` coordinates
        corresponding to SLM pixels, in ``(x_grid, y_grid)`` form.
        These are precalculated and stored in any :class:`~slmsuite.hardware.slms.slm.SLM`, so
        such a class can be passed instead of the grids directly.
    vector : (float, float)
        :math:`\vec{k}_{norm}`. Blaze vector in normalized :math:`\frac{k_x}{k}` units.
        See :meth:`~slmsuite.holography.toolbox.convert_blaze_vector()`
    offset : float
        Phase offset for this blaze.

    Returns
    -------
    numpy.ndarray
        The phase for this function.
    """
    (x_grid, y_grid) = _process_grid(grid)

    # Optimize phase construction based on context.
    if vector[0] == 0 and vector[1] == 0:
        result = np.zeros_like(x_grid)
    elif vector[1] == 0:
        result = (2 * np.pi * vector[0]) * x_grid
    elif vector[0] == 0:
        result = (2 * np.pi * vector[1]) * y_grid
    else:
        result = (2 * np.pi * vector[0]) * x_grid + (2 * np.pi * vector[1]) * y_grid

    # Add offset if provided.
    if offset != 0:
        result += offset

    return result


def sinusoid(grid, vector=(0, 0), offset=0, amplitude=np.pi, outer_offset=np.pi):
    r"""
    Returns a simple `holographic grating
    <https://en.wikipedia.org/wiki/Diffraction_grating#SR_(Surface_Relief)_gratings>`_,
    a sinusoidal grating, toward a given vector in :math:`k`-space.

    .. math:: \phi(\vec{x}) = a \sin(\vec{k}_{norm} \cdot \vec{x}_{norm} + o) + b

    Important
    ---------
    Half the power will be deflected toward the -1st order at :math:`-\vec{k}_{norm}`.

    Parameters
    ----------
    grid : (array_like, array_like) OR :class:`~slmsuite.hardware.slms.slm.SLM`
        :math:`\vec{x}_{norm}`. Meshgrids of normalized :math:`\frac{x}{\lambda}` coordinates
        corresponding to SLM pixels, in ``(x_grid, y_grid)`` form.
        These are precalculated and stored in any :class:`~slmsuite.hardware.slms.slm.SLM`, so
        such a class can be passed instead of the grids directly.
    vector : (float, float)
        :math:`\vec{k}_{norm}`. Blaze vector in normalized :math:`\frac{k_x}{k}` units.
        See :meth:`~slmsuite.holography.toolbox.convert_blaze_vector()`
    offset : float
        Grating phase offset.
    amplitude : float
        Amplitude of the sinusoid.
        The 0th order will be minimized when this is equal to :math:`\pi`.
    outer_offset : float
        Phase offset for this grating.

    Returns
    -------
    numpy.ndarray
        The phase for this function.
    """
    if vector[0] == 0 and vector[1] == 0:
        (x_grid, _) = _process_grid(grid)
        result = np.full_like(x_grid, np.sin(offset))
    else:
        result = np.sin(blaze(grid, vector, offset))

    # Add offset if provided.
    if outer_offset != 0:
        result += outer_offset

    return result


def binary(grid, vector=(0, 0), offset=0, amplitude=np.pi, outer_offset=0, duty_cycle=.5):
    r"""
    Returns a simple binary grating toward a given vector in :math:`k`-space.

    .. math:: \phi(\vec{x}) =   \left\{
                                    \begin{array}{ll}
                                        a+b, & (\vec{k}_{norm} \cdot \vec{x}_{norm} + o) \text{ mod } 2\pi < 2\pi*d \\
                                        b, & \text{ otherwise}.
                                    \end{array}
                                \right.

    Note
    ----
    When parameters are chosen to produce integer period,
    this function uses speed optimizations **(implementation incomplete)**.
    Otherwise, this function uses ``np.mod`` on top of
    :meth:`~slmsuite.holography.toolbox.phase.blaze()` to compute gratings.

    Parameters
    ----------
    grid : (array_like, array_like) OR :class:`~slmsuite.hardware.slms.slm.SLM`
        :math:`\vec{x}_{norm}`. Meshgrids of normalized :math:`\frac{x}{\lambda}` coordinates
        corresponding to SLM pixels, in ``(x_grid, y_grid)`` form.
        These are precalculated and stored in any :class:`~slmsuite.hardware.slms.slm.SLM`, so
        such a class can be passed instead of the grids directly.
    vector : (float, float)
        :math:`\vec{k}_{norm}`. Blaze vector in normalized :math:`\frac{k_x}{k}` units.
        See :meth:`~slmsuite.holography.toolbox.convert_blaze_vector()`
    offset : float
        Grating phase offset.
    amplitude : float
        Amplitude of the sinusoid.
        The 0th order will be minimized when this is equal to :math:`\pi`.
    outer_offset : float
        Phase offset for this grating.
    duty_cycle : float
        Ratio of the period which is 'on'.

    Returns
    -------
    numpy.ndarray
        The phase for this function.
    """
    result = None

    if vector[0] == 0 and vector[1] == 0:
        (x_grid, _) = _process_grid(grid)
        phase = outer_offset
        if offset != 0:
            if np.mod(offset, 2*np.pi) < (2 * np.pi * duty_cycle):
                phase += amplitude
        result = np.full_like(x_grid, phase)
    elif vector[0] != 0 and vector[1] != 0:
        pass    # xor the next case.
    elif vector[0] == 0 or vector[1] == 0:
        period = 1/np.sum(vector)
        duty = period*duty_cycle

        period_int = np.rint(period)
        duty_int = np.rint(duty)

        if np.all(np.isclose(period, period_int)) and np.all(np.isclose(duty, duty_int)):
            pass    # TODO

    # If we have not set result, then we have to use the slow np.mod option.
    if result is None:
        result = np.full_like(x_grid, outer_offset)
        result += amplitude * (
            np.mod(blaze(grid, vector, offset), 2*np.pi) < (2 * np.pi * duty_cycle)
        )

    return result


# Basic lenses.

def lens(grid, f=(np.inf, np.inf)):
    r"""
    Returns a simple
    `thin parabolic lens <https://en.wikipedia.org/wiki/Thin_lens#Physical_optics>`_.

    When the focal length :math:`f` is isotropic,

    .. math:: \phi(\vec{x}) = \frac{\pi}{f}|\vec{x}|^2

    Otherwise :math:`\vec{f}` represents an elliptical lens,

    .. math:: \phi(x, y) = \pi \left[\frac{x^2}{f_x} + \frac{y^2}{f_y} \right]

    Parameters
    ----------
    grid : (array_like, array_like) OR :class:`~slmsuite.hardware.slms.slm.SLM`
        Meshgrids of normalized :math:`\frac{x}{\lambda}` coordinates
        corresponding to SLM pixels, in ``(x_grid, y_grid)`` form.
        These are precalculated and stored in any :class:`~slmsuite.hardware.slms.slm.SLM`, so
        such a class can be passed instead of the grids directly.
    f : float OR (float, float)
        Focus in normalized :math:`\frac{x}{\lambda}` units.
        Defaults to infinity (no lens).
        Scalars are interpreted as a non-cylindrical isotropic lens.
        Future: add a ``convert_focal_length`` method to parallel
        :meth:`.convert_blaze_vector()`.

    Returns
    -------
    numpy.ndarray
        The phase for this function.
    """
    (x_grid, y_grid) = _process_grid(grid)

    # Parse focal length.
    if isinstance(f, REAL_TYPES):
        f = [f, f]
    if isinstance(f, (list, tuple, np.ndarray)):
        f = np.squeeze(f)

        assert f.shape == (2,)
        assert not np.any(f == 0), "Cannot interpret a focal length of zero."

    # Optimize phase construction based on context (for speed, to avoid square, etc).
    if np.isfinite(f[0]) and np.isfinite(f[1]):
        return (np.pi / f[0]) * np.square(x_grid) + (np.pi / f[1]) * np.square(y_grid)
    elif np.isfinite(f[0]) and np.isfinite(f[1]):
        return (np.pi / f[0]) * np.square(x_grid)
    elif np.isfinite(f[1]):
        return (np.pi / f[1]) * np.square(y_grid)
    else:
        return np.zeros_like(x_grid)


def axicon(grid, f=(np.inf, np.inf), w=None):
    r"""
    Returns an `axicon <https://en.wikipedia.org/wiki/Axicon>`_ lens, the phase farfield for a Bessel beam.
    A (elliptically)-cylindrical axicon blazes according to :math:`\vec{k}_g = w / \vec{f} / 2` where
    :math:`w` is the radius of the axicon. With a flat input amplitude over
    :math:`[-w, w]`, this will produce a Bessel beam centered at :math:`z = \vec{f}`.

    .. math:: \phi(\vec{x}) = 2\pi \cdot \vec{k}_g \cdot |\vec{x}|

    Parameters
    ----------
    grid : (array_like, array_like) OR :class:`~slmsuite.hardware.slms.slm.SLM`
        Meshgrids of normalized :math:`\frac{x}{\lambda}` coordinates
        corresponding to SLM pixels, in ``(x_grid, y_grid)`` form.
        These are precalculated and stored in any :class:`~slmsuite.hardware.slms.slm.SLM`, so
        such a class can be passed instead of the grids directly.
    f : float OR (float, float)
        Focal length (center of the axicon diamond) in normalized :math:`\frac{x}{\lambda}` units.
        Scalars are interpreted as a non-cylindrical isotropic axicon.
        Defaults to infinity (no axicon).
    w : float OR None
        See :meth:`~slmsuite.holography.toolbox._determine_source_radius()`.

    Returns
    -------
    numpy.ndarray
        The phase for this function.
    """
    (x_grid, y_grid) = _process_grid(grid)

    w = _determine_source_radius(grid, w)

    if isinstance(f, REAL_TYPES):
        f = [f, f]
    if isinstance(f, (list, tuple, np.ndarray)):
        f = np.squeeze(f)

        assert f.shape == (2,)
        assert not np.any(f == 0), "Cannot interpret a focal length of zero."

    angle = [w / f[0] / 2, w / f[1] / 2]    # Notice that this fraction is in radians.

    # Optimize phase construction based on context (for speed, to avoid sqrt, etc).
    if angle[0] == 0 and angle[1] == 0:
        return 0 * x_grid
    elif angle[0] == 0:
        return (2 * np.pi * angle[1]) * np.abs(y_grid)
    elif angle[1] == 0:
        return (2 * np.pi * angle[0]) * np.abs(x_grid)
    else:
        return (2 * np.pi) * np.sqrt(np.square(x_grid * angle[0]) + np.square(y_grid * angle[1]))


# Zernike.

ZERNIKE_INDEXING_DIMENSION = {"ansi" : 1, "noll" : 1, "fringe" : 1, "wyant" : 1, "radial" : 2}
ZERNIKE_INDEXING = ZERNIKE_INDEXING_DIMENSION.keys()
ZERNIKE_NAMES = [
    # Oth order
    "Piston",

    # 1st order
    "Vertical tilt",
    "Horizontal tilt",

    # 2nd order
    "Oblique astigmatism",
    "Defocus",
    "Vertical astigmatism",

    # 3rd order
    "Vertical trefoil",
    "Vertical coma",
    "Horizontal coma",
    "Oblique trefoil",

    # 4th order
    "Oblique quadrafoil",
    "Oblique secondary astigmatism",
    "Primary spherical aberration",
    "Vertical secondary astigmatism",
    "Vertical quadrafoil",

    # 5th order
    "Vertical pentafoil",
    "Vertical secondary trefoil",
    "Vertical secondary coma",
    "Horizontal secondary coma",
    "Oblique secondary trefoil",
    "Oblique pentafoil",
]

def convert_zernike_index(indices, from_index="ansi", to_index="ansi"):
    """
    Helper function for converting between Zernike indexing conventions.

    Currently supported conventions:

     - ``"radial"``
        The standard 2-dimensional :math:`n,l` indexing for
        `Zernike polynomials <https://en.wikipedia.org/wiki/Zernike_polynomials>`_.,
        where :math:`n` is the radial index
        and :math:`l` is the azimuthal index.

     - ``"ansi"``
        1-dimensional (0-indexed) `ANSI indices
        <https://en.wikipedia.org/wiki/Zernike_polynomials#OSA/ANSI_standard_indices>`_.
        **This is the default** :mod:`slmsuite` **index.**

     - ``"noll"``
        1-dimensional (1-indexed) `Noll indices
        <https://en.wikipedia.org/wiki/Zernike_polynomials#Noll's_sequential_indices>`_.

     - ``"fringe"``
        1-dimensional (1-indexed) `Fringe indices
        <https://en.wikipedia.org/wiki/Zernike_polynomials#Fringe/University_of_Arizona_indices>`_.

     - ``"wyant"``
        1-dimensional (0-indexed) `Wyant indices
        <https://en.wikipedia.org/wiki/Zernike_polynomials#Wyant_indices>`_.
        Equivalent to ``"fringe"``, except with starting with zero instead of one.

    Parameters
    ----------
    indices : array_like
        List of indices of shape ``(N, D)`` where ``D`` is the dimension of the indexing
        (1, apart from ``"radial"`` indexing which has a dimension of 2).
    from_index, to_index : str
        Index convention. Must be supported.

    Returns
    -------
    indices_converted : numpy.ndarray
        List of indices of shape ``(N, D)`` where ``D`` is the dimension of the indexing
        (1, apart from ``"radial"`` indexing which has a dimension of 2).
    """
    # Parse arguments.
    if from_index not in ZERNIKE_INDEXING:
        raise ValueError(f"From index '{from_index}' not recognized as a valid unit. \
                         Options: {ZERNIKE_INDEXING}")
    if to_index not in ZERNIKE_INDEXING:
        raise ValueError(f"To index '{to_index}' not recognized as a valid unit. \
                         Options: {ZERNIKE_INDEXING}")

    dimension = ZERNIKE_INDEXING_DIMENSION[from_index]

    indices = np.array(indices, dtype=int, copy=False)
    if indices.size == dimension:
        indices = indices.reshape((dimension, 1))
    if dimension > 1 and indices.shape[1] != dimension:
        raise ValueError(f"Expected dimension ({dimension}, N); found {indices.shape}")

    if from_index == to_index:
        return indices

    # Convert all cases to radial indices n, l.
    if from_index == "radial":
        n = indices[:,0]
        l = indices[:,1]
    elif from_index == "noll" or to_index == "fringe" or from_index == "wyant":
        raise RuntimeError(f"from_index '{from_index}' is not supported currently")
    elif from_index == "ansi":
        n = np.floor(.5 * np.sqrt(8*indices + 1) - .5).astype(int)
        l = 2*indices - n*(n+2)

    # Convert to the desired indices.
    if to_index == "radial":
        result = np.vstack((n, l)).T
    elif to_index == "noll":
        result = (n * (n + 1)) // 2 + np.abs(l)
        result += np.logical_and(l >= 0, np.mod(n, 4) <= 1)
        result += np.logical_and(l <= 0, np.mod(n, 4) > 1)
    elif to_index == "wyant" or to_index == "fringe":
        result = (
            np.square(1 + (n + np.abs(l)) / 2).astype(int)
            - 2 * np.abs(l) + (l < 0)
            - (to_index == "wyant")
        )
    elif to_index == "ansi":
        result = (n * (n + 2) + l) // 2

    return result


def scale_zernike_aperture(grid, aperture=None):
    """

    Parameters
    ----------
    aperture : {"circular", "elliptical", "cropped"} OR (float, float) OR None
        How to scale the polynomials relative to the grid shape. This is relative
        to the :math:`R = 1` edge of a standard Zernike pupil.

        ``"circular"``, ``None``
          The circle is scaled isotropically until the pupil edge touches one set
          of opposite grid edges. This is the default aperture.

        ``"elliptical"``
          The circle is scaled anisotropically until each pupil edge touches a grid
          edge. Generally produces an ellipse.

        ``"cropped"``
          The circle is scaled isotropically until the rectangle of the grid is
          circumscribed by the circle.

        ``(float, float)``
          Custom scaling. These values are multiplied to the ``x_grid`` and ``y_grid``
          directly, respectively. The edge of the pupil corresponds to where
          ``x_grid**2 + y_grid**2 = 1``.
    """
    # Parse grid.
    (x_grid, y_grid) = _process_grid(grid)

    # Parse aperture.
    if aperture is None:
        aperture = "circular"

    if isinstance(aperture, str):
        if aperture == "elliptical":
            x_scale = 1 / np.nanmax(x_grid)
            y_scale = 1 / np.nanmax(y_grid)
        elif aperture == "circular":
            x_scale = y_scale = 1 / np.amin([np.nanmax(x_grid), np.nanmax(y_grid)])
        elif aperture == "cropped":
            x_scale = y_scale = 1 / np.sqrt(np.nanmax(np.square(x_grid) + np.square(y_grid)))
        else:
            raise ValueError("NotImplemented")
    elif isinstance(aperture, (list, tuple)) and len(aperture) == 2:
        x_scale = aperture[0]
        y_scale = aperture[1]
    else:
        raise ValueError("Aperture type {} not recognized.".format(type(aperture)))

    return (x_scale, y_scale)


def zernike(grid, i, weight=1, **kwargs):
    r"""
    Returns a single real `Zernike polynomial <https://en.wikipedia.org/wiki/Zernike_polynomials>`_.

    Parameters
    ----------
    grid : (array_like, array_like) OR :class:`~slmsuite.hardware.slms.slm.SLM`
        Meshgrids of normalized :math:`\frac{x}{\lambda}` coordinates
        corresponding to SLM pixels, in ``(x_grid, y_grid)`` form.
        These are precalculated and stored in any :class:`~slmsuite.hardware.slms.slm.SLM`, so
        such a class can be passed instead of the grids directly.
    i : int
        ANSI Zernike index defining the polynomial.
    weight : float
        Amplitude of the polynomial.
    **kwargs
        Passed to :meth:`.zernike_sum()`.

    Returns
    -------
    numpy.ndarray
        The phase for this function.
    """
    return zernike_sum(grid, ((i, weight), ), **kwargs)


def zernike_sum(grid, weights, aperture=None, use_mask=True, dx=0, dy=0, out=None):
    r"""
    Returns a summation of
    `Zernike polynomials <https://en.wikipedia.org/wiki/Zernike_polynomials>`_
    in a computationally-efficient manner. To improve performance, especially for higher
    order polynomials, we store a cache of Zernike coefficients to avoid regeneration.
    See the below example to generate
    :math:`Z_1 - Z_2 + Z_3 = Z_2^{-1} - Z_2^{-1} + Z_2^{-1}`,
    where :math:`Z_n^l` is represented by the ansi index :math:`i` as :math:`Z_i`.

    .. highlight:: python
    .. code-block:: python

        zernike_sum_phase = toolbox.phase.zernike_sum(
            grid=slm,
            weights=(
                (1,  1),       # Z_1
                (2, -1),       # Z_2
                (3,  1)        # Z_3
            ),
            aperture="circular"
        )

    Important
    ~~~~~~~~~
    Zernike polynomials are canonically defined on a circular aperture. However, we may
    want to use these polynomials on other apertures (e.g. a rectangular SLM).
    Cropping this aperture breaks the orthogonality and normalization of the set, but
    this is fine for many applications. While it is possible to orthonormalize the
    cropped set, we do not do so in :mod:`slmsuite`, as this is not critical for target
    applications such as aberration correction.

    Parameters
    ----------
    grid : (array_like, array_like) OR :class:`~slmsuite.hardware.slms.slm.SLM`
        Meshgrids of normalized :math:`\frac{x}{\lambda}` coordinates
        corresponding to SLM pixels, in ``(x_grid, y_grid)`` form.
        These are precalculated and stored in any :class:`~slmsuite.hardware.slms.slm.SLM`, so
        such a class can be passed instead of the grids directly.
    weights : list of (int, float)
        Which Zernike polynomials to sum.
        The ``int`` is the ANSI index ``i``.
        The ``float`` is the weight for the given index.
    aperture : {"circular", "elliptical", "cropped"} OR (float, float) OR None
        Passed to :meth:`.scale_zernike_aperture()`.
    use_mask : bool OR "return"
        If ``True``, sets the area where standard Zernike polynomials are undefined to zero.
        If ``False``, the polynomial is not cropped. This should be used carefully, as
        the wavefront correction outside the unit circle quickly explodes with
        :math:`r^O` for terms of high order :math:`O`.
        If ``"return"``, returns the 2D mask ``x_grid**2 + y_grid**2 <= 1``.
    dx, dy : int
        If non-zero, returns the Zernike derivative of the given order. For instance,
        ``dx = 1, dy = 0`` corresponds to the first derivative in the :math:`x` direction.
    out : array_like OR None
        Memory to be used for the phase output. Allocated separately if ``None``.

    Returns
    -------
    numpy.ndarray
        The phase for this function. Optionally returns the 2D Zernike mask.
    """
    # Parse passed values
    (x_grid, y_grid) = _process_grid(grid)
    (x_scale, y_scale) = scale_zernike_aperture(grid, aperture)

    # At the end, we're going to set the values outside the aperture to zero.
    # Make a mask for this if it's necessary.
    mask = np.square(x_grid * x_scale) + np.square(y_grid * y_scale) <= 1
    if use_mask == "return":
        return mask
    use_mask = use_mask and np.any(mask == 0)

    if use_mask:
        x_grid_scaled = x_grid[mask] * x_scale
        y_grid_scaled = y_grid[mask] * y_scale
    else:
        x_grid_scaled = x_grid * x_scale
        y_grid_scaled = y_grid * y_scale

    # Now find the coefficients for polynomial terms x^ay^b. We want to only compute
    # x^ay^b once because this is an operation on a large array. In contrast, summing
    # the coefficients of the same terms is simple and fast scalar operations.
    summed_coefficients = {}

    for (key, weight) in weights:
        coefficients = _zernike_coefficients(key)

        for power_key, factor in coefficients.items():
            power_factor = factor * weight

            if dx != 0 or dy != 0:
                # Apply the power rule to the coefficient.
                if dx == 1:
                    power_factor *= power_key[0]
                elif dx != 0:
                    if power_key[0] >= dx:
                        power_factor *= factorial(power_key[0]) / factorial(power_key[0] - dx)
                    else:
                        power_factor = 0

                if dy == 1:
                    power_factor *= power_key[1]
                elif dy != 0:
                    if power_key[1] >= dy:
                        power_factor *= factorial(power_key[1]) / factorial(power_key[1] - dy)
                    else:
                        power_factor = 0

                # Change the power key based on derivatives.
                power_key = (power_key[0] - dx, power_key[1] - dy)

            # Add the coefficient to the sum for the given monomial power.
            if power_key in summed_coefficients:
                summed_coefficients[power_key] += power_factor
            else:
                summed_coefficients[power_key] = power_factor

    # Finally, build the polynomial.
    if True:
        canvas = np.zeros(x_grid.shape)

        for power_key, factor in summed_coefficients.items():
            if factor != 0:
                if power_key == (0,0):
                    if use_mask:
                        canvas[mask] += factor
                    else:
                        canvas += factor
                else:
                    if use_mask:
                        canvas[mask] += factor * np.power(x_grid_scaled, power_key[0]) * np.power(y_grid_scaled, power_key[1])
                    else:
                        canvas += factor * np.power(x_grid_scaled, power_key[0]) * np.power(y_grid_scaled, power_key[1])
    else:
        pass


    return canvas


def _plot_zernike_pyramid(grid, order, scale=1, **kwargs):
    """
    Plots :meth:`.zernike()` on a pyramid of subplots corresponding to the radial and
    azimuthal order. The user can resize the figure with ``plt.figure()`` beforehand
    and force ``plt.show()`` afterward.

    Parameters
    ----------
    grid : (array_like, array_like) OR :class:`~slmsuite.hardware.slms.slm.SLM`
        Meshgrids of normalized :math:`\frac{x}{\lambda}` coordinates
        corresponding to SLM pixels, in ``(x_grid, y_grid)`` form.
        These are precalculated and stored in any :class:`~slmsuite.hardware.slms.slm.SLM`, so
        such a class can be passed instead of the grids directly.
    order : int
        Maximum radial order to plot.
    scale : float
        Scales the subplots to ``[-scale, scale]``.
    **kwargs
        Passed to :meth:`.zernike()`.
    """
    indices_ansi = np.arange((order * (order + 1)) // 2)
    indices_radial = convert_zernike_index(indices_ansi, from_index="ansi", to_index="radial")

    for i in indices_ansi:
        n, l = indices_radial[i, :]
        m = (n + l) // 2

        phase = zernike(grid, i, 1, **kwargs)

        plt.subplot(order, order, 1 + m + n*order)
        plt.imshow(phase)
        plt.clim([-scale, scale])
        plt.xticks([])
        plt.yticks([])


# Old style dictionary.
#   {(n,m) : {(nx, ny) : w, ... }, ... }
_zernike_cache = {}
# New style matrix.
#   N x M, N spans ansi Zernike indices, M spans cantor polynomial indices.
_zernike_cache_vectorized = np.array([[]], dtype=int)

def _zernike_build(n):
    """Pre-caches Zernike polynomials up to order :math:`n`."""
    N = (n+1) * (n+2) // 2
    for i in range(N):
        _zernike_coefficients(i)

def _zernike_coefficients(index):
    """
    Returns the coefficients for the :math:`x^ay^b` terms of the real Zernike polynomial
    of ANSI index ``i``. This is returned as a dictionary of form ``{(a,b) : coefficient}``.
    Uses `this algorithm <https://doi.org/10.1117/12.294412>`_.
    """
    index = int(index)

    # Generate coefficients only if we have not already generated.
    if not index in _zernike_cache:
        zernike_this = {}

        (n, l) = convert_zernike_index(index, to_index="radial")[0]

        # Define helper variables.
        if l % 2:   # If even
            q = int((abs(l) - 1) / 2)
        else:
            if l > 0:
                q = int(abs(l)/2 - 1)
            else:
                q = int(abs(l)/2)

        if l <= 0:
            p = 0
        else:
            p = 1

        l = abs(l)
        m = int((n-l)/2)

        # Helper function
        def comb(n, k):
            return factorial(n) / (factorial(k) * factorial(n-k))

        # Finding the coefficients is a summed combinatorial search.
        # This is why we cache: so we don't have to do this many times,
        # especially for higher order polynomials and the corresponding cubic scaling.
        for i in range(q+1):
            for j in range(m+1):
                for k in range(m-j+1):
                    factor = -1 if (i + j) % 2 else 1
                    factor *= comb(l, 2 * i + p)
                    factor *= comb(m - j, k)
                    factor *= (float(factorial(n - j))
                        / (factorial(j) * factorial(m - j) * factorial(n - m - j)))

                    power_key = (int(n - 2*(i + j + k) - p), int(2 * (i + k) + p))

                    # Add this coefficient to the element in the dictionary
                    # corresponding to the right power.
                    if power_key in zernike_this:
                        zernike_this[power_key] += int(factor)
                    else:
                        zernike_this[power_key] = int(factor)

        # Update the cache. Remove all factors that have cancelled out (== 0).
        _zernike_cache[index] = {
            power_key: factor
            for power_key, factor in zernike_this.items()
            if factor != 0
        }

        # If we need to, enlarge the vector cache.
        N = (n+1) * (n+2) // 2      # The Zernike order determines the size of the cache.
        global _zernike_cache_vectorized

        if _zernike_cache_vectorized.shape[1] < N:
            _zernike_cache_vectorized = np.pad(
                _zernike_cache_vectorized,
                (
                    (0, N - _zernike_cache_vectorized.shape[0]),
                    (0, N - _zernike_cache_vectorized.shape[1])
                ),
                constant_values=0
            )

        # Update the vectorized dict.
        for power_key, factor in _zernike_cache[index].items():
            cantor_index = _cantor_pairing(power_key)
            _zernike_cache_vectorized[index, cantor_index] = factor

    return _zernike_cache[index]

# Polynomials.

def _cantor_pairing(xy):
    """
    Converts a 2D index to a unique 1D index according to the
    `Cantor pairing function <https://en.wikipedia.org/wiki/Pairing_function>`.
    """
    xy = np.array(xy, dtype=int, copy=False).reshape((-1, 2))
    return (np.round(.5 * (xy[:,0] + xy[:,1]) * (xy[:,0] + xy[:,1] + 1) + xy[:,1])).astype(int)


def _inverse_cantor_pairing(z):
    """
    Converts a 1D index to a unique 2D index according to the
    `Cantor pairing function <https://en.wikipedia.org/wiki/Pairing_function>`.
    """
    z = np.array(z, dtype=int, copy=False)

    w = np.floor((np.sqrt(8*z + 1) - 1) // 2).astype(int)
    t = (w*w + w) // 2

    y = z-t
    x = w-y

    return np.vstack((x, y))


def _term_pathing(xy):
    """
    Returns the index for term sorting to minimize number of monomial multiplications when summing
    polynomials (with only one storage variable). This yields a provably-optimal set of paths.
    The proof is left as an exercise to the reader.

    It may be the case that division could yield a shorter path, but division is
    generally more expensive than multiplication so we omit this scenario.

    It may also be the case that optimizing for large-step multiplications can yield a
    speedup. (e.g. `x^5 = y * y * x` with `y = x * x` costs three multiplications instead
    of five) However, it is unlikely that users will need the very-high-order
    polynomials would would experience an appreciable speedup.
    """
    # Prepare helper variables.
    xy = np.array(xy, dtype=int, copy=False)

    order = np.sum(xy, axis=1)
    delta = np.diff(xy, axis=1)

    cantor = _cantor_pairing(xy)
    cantor_index = np.argsort(cantor)

    # Prepare the output data structure.
    I = np.zeros_like(order, dtype=int)

    # Helper function to recurse through pathing options.
    def recurse(i0, j0):
        # Fill in the current values.
        I[j0] = i0
        cantor[cantor_index[i0]] = -1

        # Figure out the distance between the current index and all other indices.
        dd = delta - delta[cantor_index[i0]]
        do = order[cantor_index[i0]] - order

        # Find the best candidate for the next index in the thread.
        nearest = -cantor + np.inf * ((np.abs(dd) >= do) + (do > 0) + cantor >= 0)
        i = np.argmin(nearest)

        # Either exit or continue this thread.
        if cantor[cantor_index[i]] == -1:
            return recurse(i, j0-1)
        else:
            return j0-1

    # Traverse backwards through the array,
    j = len(I)-1
    for i in range(len(order)):
        if cantor[cantor_index[i]] >= 0:
            j = recurse(i, j)

    return I


try:
    _polynomial_sum_kernel = cp.RawKernel(CUDA_KERNELS, 'polynomial_sum')
except:
    _polynomial_sum_kernel = None

def polynomial_sum(grid, weights, terms=None, pathing=None, out=None):
    """
    TODO

    Parameters
    ----------
    grid : (array_like, array_like) OR :class:`~slmsuite.hardware.slms.slm.SLM`
        Meshgrids of normalized :math:`\frac{x}{\lambda}` coordinates
        corresponding to SLM pixels, in ``(x_grid, y_grid)`` form.
        These are precalculated and stored in any :class:`~slmsuite.hardware.slms.slm.SLM`, so
        such a class can be passed instead of the grids directly.

    """
    # Parse terms
    if terms is None:
        terms = _inverse_cantor_pairing(np.arange(len(weights)))
    else:
        terms = np.squeeze(terms)

    if terms.ndim == 1: # TODO check corner case!
        terms = _inverse_cantor_pairing(terms)

    if not terms.shape[1] == 2:
        raise ValueError("TODO")

    # Parse pathing
    if pathing is False:
        pathing = np.arange(terms.shape[0])
    if pathing is None:
        pathing = _term_pathing(terms)

    # Prepare the grids and canvas.
    (x_grid, y_grid) = _process_grid(grid)
    if out is None:
        # Initialize out to zero.
        if cp == np:
            out = np.zeros_like(x_grid)
        else:
            out = cp.get_array_module(x_grid).zeros_like(x_grid)
    else:
        # Error check user-provided out.
        if out.shape != x_grid.shape:
            raise ValueError("TODO")
        if out.dtype != x_grid.dtype:
            raise ValueError("TODO")
        if cp != np and cp.get_array_module(x_grid) != cp.get_array_module(out):
            raise ValueError("TODO")

    # Decide whether to use numpy/cupy or CUDA
    if cp == np or _polynomial_sum_kernel is None or cp.get_array_module(x_grid) == np:  # numpy/cupy
        out.fill(0)
        nx0 = ny0 = 0
        if cp == np:
            monomial = np.ones_like(x_grid)
        else:
            monomial = cp.get_array_module(x_grid).ones_like(x_grid)

        # Sum the result.
        for index in pathing:
            if weights[index] != 0:
                (nx, ny) = terms[index, :]

                # Reset if we're starting a new path.
                if nx - nx0 < 0 or ny - ny0 < 0:
                    nx0 = ny0 = 0
                    monomial.fill(1)

                # Traverse the path in +x or +y.
                for _ in range(nx - nx0):
                    monomial *= x_grid
                for _ in range(ny - ny0):
                    monomial *= y_grid

                # Add the monomial to the result.
                out += weights[index] * monomial
    else:                               # CUDA
        N = int(terms.shape[0])
        WH = int(x_grid.size)

        threads_per_block = int(_polynomial_sum_kernel.max_threads_per_block)
        blocks = WH // threads_per_block

        # Call the RawKernel.
        _polynomial_sum_kernel(
            (blocks,),
            (threads_per_block,),
            (
                N,
                cp.array(pathing, copy=False),
                cp.array(weights, copy=False),
                cp.array(terms[:, 0], copy=False),
                cp.array(terms[:, 1], copy=False),
                WH,
                x_grid.ravel(),
                y_grid.ravel(),
                out.ravel()
            )
        )

    return out

# Structured light.

def _determine_source_radius(grid, w=None):
    r"""
    Helper function to determine the assumed Gaussian source radius for various
    structured light conversion functions. This is important because structured light
    conversions need knowledge of the size of the incident Gaussian beam.
    For example, see the ``w`` parameter in
    :meth:`~slmsuite.holography.toolbox.phase.laguerre_gaussian()`.

    Note
    ~~~~
    Future work: when ``grid`` is a :class:`~slmsuite.hardware.slms.slm.SLM` which has completed
    :meth:`~slmsuite.hardware.cameraslm.FourierSLM.fourier_calibration()`, this function should fit
    (and cache?) :attr:`~slmsuite.hardware.slms.slm.amplitude_measured` to a Gaussian
    and use the resulting width (and center?).

    Parameters
    ----------
    grid : (array_like, array_like) OR :class:`~slmsuite.hardware.slms.slm.SLM`
        Meshgrids of normalized :math:`\frac{x}{\lambda}` coordinates
        corresponding to SLM pixels, in ``(x_grid, y_grid)`` form.
        These are precalculated and stored in any :class:`~slmsuite.hardware.slms.slm.SLM`, so
        such a class can be passed instead of the grids directly.
    w : float OR None
        The radius of the phase pattern in normalized :math:`\frac{x}{\lambda}` units.
        To produce perfect structured beams, this radius is equal to the radius of
        the gaussian profile of the source (ideally not clipped by the SLM).
        If ``w`` is left as ``None``, ``w`` is set to a quarter of the smallest normalized screen dimension.

    Returns
    -------
    float
        Determined radius. In normalized units.
    """
    (x_grid, y_grid) = _process_grid(grid)

    if w is None:
        return np.min([np.amax(x_grid), np.amax(y_grid)]) / 4
    else:
        return w


def laguerre_gaussian(grid, l, p, w=None):
    r"""
    Returns the phase farfield for a
    `Laguerre-Gaussian <https://en.wikipedia.org/wiki/Gaussian_beam#Laguerre-Gaussian_modes>`_
    beam.

    This function is especially useful to hone and validate SLM alignment. Perfect alignment will
    result in concentric and uniform fringes for higher order beams. Focusing issues, aberration,
    or pointing misalignment will mitigate this.

    Parameters
    ----------
    grid : (array_like, array_like) OR :class:`~slmsuite.hardware.slms.slm.SLM`
        Meshgrids of normalized :math:`\frac{x}{\lambda}` coordinates
        corresponding to SLM pixels, in ``(x_grid, y_grid)`` form.
        These are precalculated and stored in any :class:`~slmsuite.hardware.slms.slm.SLM`, so
        such a class can be passed instead of the grids directly.
    l : int
        The azimuthal wavenumber, or orbital angular momentum. Can be negative.
    p : int
        The radial wavenumber. Should be non-negative.
    w : float OR None
        See :meth:`~slmsuite.holography.toolbox._determine_source_radius()`.

    Returns
    -------
    numpy.ndarray
        The phase for this function.
    """
    (x_grid, y_grid) = _process_grid(grid)

    w = _determine_source_radius(grid, w)

    theta_grid = np.arctan2(x_grid, y_grid)
    radius_grid = y_grid * y_grid + x_grid * x_grid

    return np.mod(
        l * theta_grid
        + np.pi
        * np.heaviside(-special.genlaguerre(p, np.abs(l))(2 * radius_grid / w / w), 0)
        + np.pi,
        2 * np.pi,
    )


def hermite_gaussian(grid, n, m, w=None):
    r"""
    Returns the phase farfield for a
    `Hermite-Gaussian <https://en.wikipedia.org/wiki/Gaussian_beam#Hermite-Gaussian_modes>`_
    beam. Uses the formalism described by `this paper <https://doi.org/10.1364/AO.54.008444>`_.

    Parameters
    ----------
    grid : (array_like, array_like) OR :class:`~slmsuite.hardware.slms.slm.SLM`
        Meshgrids of normalized :math:`\frac{x}{\lambda}` coordinates
        corresponding to SLM pixels, in ``(x_grid, y_grid)`` form.
        These are precalculated and stored in any :class:`~slmsuite.hardware.slms.slm.SLM`, so
        such a class can be passed instead of the grids directly.
    n, m : int
        The horizontal (``n``) and vertical (``m``) wavenumbers. ``n = m = 0`` yields a flat
        phase or a standard Gaussian beam.
    w : float
        See :meth:`~slmsuite.holography.toolbox._determine_source_radius()`.

    Returns
    -------
    numpy.ndarray
        The phase for this function.
    """
    (x_grid, y_grid) = _process_grid(grid)
    w = _determine_source_radius(grid, w)

    factor = np.sqrt(2) / w

    # Generate the amplitude of a Hermite-Gaussian mode.
    phase = special.hermite(n)(factor * x_grid) * special.hermite(m)(factor * y_grid)

    # This is real, so the phase is just the sign of the mode. This produces a
    # checkerboard pattern. Probably could make this faster by bitflipping rows and columns.
    phase[phase < 0] = 0
    phase[phase > 0] = np.pi

    return phase


def ince_gaussian(grid, p, m, parity=1, ellipticity=1, w=None):
    r"""
    **(NotImplemented)** Returns the phase farfield for an
    `Ince-Gaussian <https://en.wikipedia.org/wiki/Gaussian_beam#Ince-Gaussian_modes>`_
    beam.
    `Consider <https://doi.org/10.1364/OL.29.000144>`_
    `using <https://doi.org/10.1364/AO.54.008444>`_
    `these <https://doi.org/10.3390/jimaging8050144>`_
    `references <https://en.wikipedia.org/wiki/Elliptic_coordinate_system>`_.

    Parameters
    ----------
    grid : (array_like, array_like) OR :class:`~slmsuite.hardware.slms.slm.SLM`
        Meshgrids of normalized :math:`\frac{x}{\lambda}` coordinates
        corresponding to SLM pixels, in ``(x_grid, y_grid)`` form.
        These are precalculated and stored in any :class:`~slmsuite.hardware.slms.slm.SLM`, so
        such a class can be passed instead of the grids directly.
    p : int
        Ince polynomial order.
    m : int
        Ince polynomial degree.
    parity : {1, -1, 0}
        Whether to produce an even (1), odd (-1), or helical (0) Ince polynomial. A helical
        polynomial is the linear combination of even and odd polynomials.

        .. math:: IG^h_{p,m} = IG^e_{p,m} + iIG^o_{p,m}

    ellipticity : float
        Ellipticity of the beam. The semifocal distance is equal to ``ellipticity * w``,
        where the foci are the points which define the elliptical coordinate system.
    w : float
        See :meth:`~slmsuite.holography.toolbox._determine_source_radius()`.

    Returns
    -------
    numpy.ndarray
        The phase for this function.
    """
    (x_grid, y_grid) = _process_grid(grid)
    w = _determine_source_radius(grid, w)

    if parity == 1:
        assert 0 <= m <= p
    else:
        assert 1 <= m <= p

    complex_grid = x_grid + 1j * y_grid

    factor = 1 / (w * np.sqrt(ellipticity / 2))

    elliptic_grid = np.arccosh(complex_grid * factor)

    raise NotImplementedError()


def matheui_gaussian(grid, r, q, w=None):
    """
    **(NotImplemented)** Returns the phase farfield for a
    `Matheui-Gaussian <https://doi.org/10.1364/AO.49.006903>`_ beam.

    Returns
    -------
    numpy.ndarray
        The phase for this function.
    """
    (x_grid, y_grid) = _process_grid(grid)
    w = _determine_source_radius(grid, w)

    raise NotImplementedError()

