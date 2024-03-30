"""
Repository of common analytic phase patterns.
"""

import numpy as np
try:
    import cupy as cp
except ImportError:
    cp = np
from scipy import special
from math import factorial

from slmsuite.misc.math import REAL_TYPES
from slmsuite.holography.toolbox import _process_grid

# Basic functions

def blaze(grid, vector=(0, 0), offset=0):
    r"""
    Returns a simple `blaze <https://en.wikipedia.org/wiki/Blazed_grating>`_,
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
    offset :
        Phase offset for this blaze.

    Returns
    -------
    numpy.ndarray
        The phase for this function.
    """
    (x_grid, y_grid) = _process_grid(grid)

    # Optimize phase construction based on context.
    if vector[0] == 0 and vector[1] == 0:
        result = 0 * x_grid
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
        Scalars are interpreted as a non-cylindrical isotropic lens.
        Future: add a ``convert_focal_length`` method to parallel
        :meth:`.convert_blaze_vector()`
        Defaults to infinity (no lens).

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


# Zernike

ZERNIKE_INDEXING_DIMENSION = {"ansi" : 1, "noll" : 1, "fringe" : 1, "wyant" : 1, "radial" : 2, "cartesian" : 2}
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

     -  ``"radial"``
        The standard :math:`n,l` indexing for Zernike polynomials, where
        :math:`n` is the radial index and
        :math:`l` is the azimuthal index.

     -  ``"cartesian"``
        A modified
        The standard :math:`n,l`

     -  ``"ansi"``
        1-dimensional `ANSI indices
        https://en.wikipedia.org/wiki/Zernike_polynomials#OSA/ANSI_standard_indices>`_.
        **This is the default slmsuite unit.**

     -  ``"noll"``
        1-dimensional `Noll indices
        <https://en.wikipedia.org/wiki/Zernike_polynomials#Noll's_sequential_indices>`_.

     -  ``"fringe"``
        1-dimensional `Fringe indices
        <https://en.wikipedia.org/wiki/Zernike_polynomials#Fringe/University_of_Arizona_indices>`_.

     -  ``"wyant"``
        1-dimensional `Wyant indices
        <https://en.wikipedia.org/wiki/Zernike_polynomials#Wyant_indices>`_.
        Equivalent to ``"fringe"``, except with starting with zero instead of one.

    """
    # Parse arguments.
    if from_index not in ZERNIKE_INDEXING:
        raise ValueError(f"From index '{from_index}' not recognized as a valid unit. \
                         Options: {ZERNIKE_INDEXING}")
    if to_index not in ZERNIKE_INDEXING:
        raise ValueError(f"To index '{to_index}' not recognized as a valid unit. \
                         Options: {ZERNIKE_INDEXING}")

    dimension = ZERNIKE_INDEXING_DIMENSION[from_index]

    if indices.shape[0] != dimension:
        raise ValueError(f"Expected dimension ({dimension}, N); found {indices.shape}")

    indices = np.array(indices, dtype=int, copy=False)

    # Convert all cases to radial indices n, l.

    if from_index == "cartesian":
        n = indices[0, :]
        l = 2 * indices[1, :] - n   # 2m - n
    elif from_index == "radial":
        n = indices[0, :]
        l = indices[1, :]
    elif from_index == "noll" or to_index == "fringe" or from_index == "wyant":
        raise RuntimeError(f"from_index '{from_index}' is not supported currently")
    elif from_index == "ansi":
        w = np.floor((np.sqrt(8*indices - 1) - 1) // 2).astype(int)
        t = (w*w + w) // 2

        y = indices-t
        x = w-y

        n = x + y
        l = y - x

    # Convert to the desired indices.
    if to_index == "cartesian":
        result = np.vstack((n, (l - n) // 2))
    elif to_index == "radial":
        result = np.vstack((n, l))
    elif to_index == "noll":
        result = (n * (n + 1)) // 2 + np.abs(l)
        result += np.logical_or(l >= 0, np.mod(n, 4) <= 1)
        result += (l == 0)
    elif to_index == "wyant" or to_index == "fringe":
        result = (
            np.square(1 + (n + np.abs(l) // 2))
            - 2 * np.abs(l) + (l < 0)
            - (to_index == "wyant")
        )
    elif to_index == "ansi":
        result = (n * (n + 2) + l) // 2

    return result



def zernike(grid, i, aperture=None, return_mask=False):
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
    aperture : {"circular", "elliptical", "cropped"} OR (float, float) OR None
        See :meth:`.zernike_sum()`.
    return_mask : bool
        Whether or not to return the 2D mask showing where Zernikes are computed
        instead of the phase.

    Returns
    -------
    numpy.ndarray
        The phase for this function.
    """
    return zernike_sum(grid, ((i, 1), ), aperture=aperture, return_mask=return_mask)


def zernike_sum(grid, weights, aperture=None, return_mask=False, out=None, dx=0, dy=0):
    r"""
    Returns a summation of
    `Zernike polynomials <https://en.wikipedia.org/wiki/Zernike_polynomials>`_
    in a computationally-efficient manner. To improve performance, especially for higher
    order polynomials, we store a cache of Zernike coefficients to avoid regeneration.
    See the below example to generate
    :math:`Z_1 - Z_2 + Z_3 = Z_2^{-1} - Z_2^{-1} + Z_2^{-1}`.

    .. highlight:: python
    .. code-block:: python

        zernike_sum_phase = toolbox.phase.zernike_sum(
            grid=slm,
            weights=(   (1,  1),       # Z_20
                        (2, -1),       # Z_21
                        (3,  1)    ),  # Z_31
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
        How to scale the polynomials relative to the grid shape. This is relative
        to the :math:`R = 1` edge of a standard Zernike pupil.

        ``"circular"``, ``None``
          The circle is scaled isotropically until the pupil edge touches the grid edge.
          This is the default aperture.

        ``"elliptical"``
          The circle is scaled anisotropically until each pupil edge touches a grid
          edge. Generally produces and ellipse.

        ``"cropped"``
          The circle is scaled isotropically until the rectangle of the grid is
          circumscribed by the circle.

        ``(float, float)``
          Custom scaling. These values are multiplied to the ``x_grid`` and ``y_grid``
          directly, respectively. The edge of the pupil corresponds to where
          ``x_grid**2 + y_grid**2 = 1``.

    return_mask : bool
        Whether or not to return the 2D mask showing where Zernikes are computed
        instead of the phase.
    out : array_like

    dx, dy : int
        If non-zero, returns the Zernike derivative of the given order.
        For instance, `dx = 1, dy = 0` corresponds to the first derivative in the x direction.

    Returns
    -------
    numpy.ndarray
        The phase for this function.

    numpy.ndarray
        Optional return for the 2D Zernike mask.
    """
    # Parse passed values
    (x_grid, y_grid) = _process_grid(grid)

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

    # At the end, we're going to set the values outside the aperture to zero.
    # Make a mask for this if it's necessary.
    mask = np.square(x_grid * x_scale) + np.square(y_grid * y_scale) <= 1
    if return_mask:
        return mask
    use_mask = np.any(mask == 0)

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
                if dx != 0:
                    power_factor *= factorial(power_key[0]) / factorial(power_key[0] - dx)
                if dy != 0:
                    power_factor *= factorial(power_key[1]) / factorial(power_key[1] - dy)

                # Change the power key based on derivatives.
                power_key = (power_key[0] - dx, power_key[1] - dy)

            # Add the coefficient to the sum for the given monomial power.
            if power_key in summed_coefficients:
                summed_coefficients[power_key] += power_factor
            else:
                summed_coefficients[power_key] = power_factor

    # Finally, build the polynomial.
    if False:
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

# Old style dictionary.
#   {(n,m) : {(nx, ny) : w, ... }, ... }
_zernike_cache = {}
# New style matrix.
#   N x M, N spans cantor polynomial indices and M spans ansi Zernike indices.
_zernike_cache_vectorized = np.array([[]])

def _zernike_coefficients(i):
    """
    Returns the coefficients for the :math:`x^ay^b` terms of the real Zernike polynomial
    of ANSI index ``i``. This is returned as a dictionary of form ``{(a,b) : coefficient}``.
    Uses `this algorithm <https://doi.org/10.1117/12.294412>`_.
    """
    i = int(i)

    # Generate coefficients only if we have not already generated.
    if not i in _zernike_cache:
        zernike_this = {}

        (n, l) = convert_zernike_index(i, to_index="radial")

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

                    power_key = (n - 2*(i + j + k) - p, 2 * (i + k) + p)

                    # Add this coefficient to the element in the dictionary
                    # corresponding to the right power.
                    if power_key in zernike_this:
                        zernike_this[power_key] += factor
                    else:
                        zernike_this[power_key] = factor

        # Update the cache. Remove all factors that have cancelled out (== 0).
        _zernike_cache[i] = {power_key: factor for power_key, factor in zernike_this.items() if factor != 0}

        # If we need to, enlarge the vector cache.
        if _zernike_cache_vectorized.shape[1] < i+1:
            N = factorial(n)

            _zernike_cache_vectorized = np.pad(
                _zernike_cache_vectorized,
                (
                    (0, M + 1 - _zernike_cache_vectorized.shape[0]),
                    (0, N - _zernike_cache_vectorized.shape[1])
                ),
                constant_values=0
            )

            for power_key, factor in _zernike_cache[i].items():
                cantor_index = _cantor_pairing(power_key)
                _zernike_cache_vectorized[i, cantor_index] = factor

    return _zernike_cache[i]

def zernike_expand(grid, phase, phase_dx=None, phase_dy=None, rsquared=None, order=5, tolerance=.001):
    r"""
    Returns a single real `Zernike polynomial <https://en.wikipedia.org/wiki/Zernike_polynomials>`_.

    Parameters
    ----------
    grid : (array_like, array_like) OR :class:`~slmsuite.hardware.slms.slm.SLM`
        Meshgrids of normalized :math:`\frac{x}{\lambda}` coordinates
        corresponding to SLM pixels, in ``(x_grid, y_grid)`` form.
        These are precalculated and stored in any :class:`~slmsuite.hardware.slms.slm.SLM`, so
        such a class can be passed instead of the grids directly.
    phase : array_like
        Measured phase pattern of the same shape as the grids to expand as a summation
        of Zernike polynomials.
    phase_dx, phase_dy : array_like
        Measured derivatives in the x and y directions of the phase pattern.
        This helps to pin the accuracy of the fit. Of the same shape as the grids.
    rsquared : array_like
        Measured fit r-squared. This helps to ignore parts of the phase pattern (and
        derivates) which are invalid during measurement. Of the same shape as the grids.
    order : int
        TODO
    tolerance : int
        TODO

    Returns
    -------
    weights : list of ((int, int), float)
        The coefficients of the Zernike fit. The ``(int, int)`` is the index ``(n, m)``,
        which correspond to the azimuthal degree and order of the polynomial in ``"asci"``
        zernike indices.  The ``float`` is the weight for the given index.
    """
    # Parse passed values
    (x_grid, y_grid) = _process_grid(grid)



    return fit

# Polynomials

def _cantor_pairing(xy):
    """
    Converts a 2D index to a unique 1D index according to the
    `Cantor pairing function <https://en.wikipedia.org/wiki/Pairing_function>`.
    """
    return (np.round(.5 * (xy[0,:] + xy[1,:]) * (xy[0,:] + xy[1,:] + 1) + xy[1,:])).astype(int)


def _inverse_cantor_pairing(z):
    """
    Converts a 1D index to a unique 2D index according to the
    `Cantor pairing function <https://en.wikipedia.org/wiki/Pairing_function>`.
    """
    z = np.array(z, dtype=int, copy=False)

    w = np.floor((np.sqrt(8*z - 1) - 1) // 2).astype(int)
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

    cantor = _cantor_pairing(xy[:, 0], xy[:, 1])
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
    _polynomial_sum_kernel = cp.RawKernel(
        r'''
        #include <cupy/complex.cuh>
        extern "C"
        __global__ void polynomial_sum(
            const unsigned int N,           // Number of coefficients
            const int* pathing,             // Path order (1*N)
            const float* coefficients,      // Spot parameters (1*N)
            const float* px,                // Spot parameters (1*N)
            const float* py,                // Spot parameters (1*N)
            const unsigned int WH,          // Size of nearfield
            const float* X,                 // X grid (WH)
            const float* Y,                 // Y grid (WH)
            float* out                      // Output (WH)
        ) {
            // g is each pixel in the grid.
            int g = blockDim.x * blockIdx.x + threadIdx.x;

            if (g < WH) {
                // Make a local result variable to avoid talking with global memory.
                float result = 0;

                // Copy data that will be used multiple times per thread into local memory (this might not matter though).
                float local_X = X[g];
                float local_Y = Y[g];
                float coefficient;

                // Local helper variables.
                float monomial = 1;
                int nx, ny, nx0, ny0 = 0;
                int j, k = 0

                // Loop over all the spots (compiler should handle optimizing the trinary).
                for (int i = 0; i < N; i++) {
                    k = pathing[i];
                    coefficient = coefficients[k];

                    if (coefficient != 0) {
                        nx = px[k];
                        ny = py[k];

                        // Reset if we're starting a new path.
                        if (nx - nx0 < 0 || ny - ny0 < 0) {
                            nx0 = ny0 = 0;
                            monomial = 1;
                        }

                        // Traverse the path in +x or +y.
                        for (j = 0; j < nx - nx0; j++) {
                            monomial *= local_X;
                        }
                        for (j = 0; j < ny - ny0; j++) {
                            monomial *= local_Y;
                        }

                        // Add the monomial to the result.
                        result += coefficients[k] * monomial;

                        // Update the state of the monomial
                        nx0 = nx;
                        ny0 = ny;
                    }
                }

                // Export the result to global memory.
                out[g] = result;
            }
        }
        ''',
        'polynomial_sum'
    )
except:
    _polynomial_sum_kernel = None

def polynomial_sum(grid, weights, terms=None, pathing=None, out=None):
    """
    Returns a single real `Zernike polynomial <https://en.wikipedia.org/wiki/Zernike_polynomials>`_.

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

    terms = np.squeeze(terms)

    if terms.ndim == 1: # TODO check corner case!
        terms = _inverse_cantor_pairing(terms)

    assert terms.shape[1] == 2

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
            raise RuntimeError("TODO")
        if out.dtype != x_grid.dtype:
            raise RuntimeError("TODO")
        if cp != np and cp.get_array_module(x_grid) != cp.get_array_module(out):
            raise RuntimeError("TODO")

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

# Structured light

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

