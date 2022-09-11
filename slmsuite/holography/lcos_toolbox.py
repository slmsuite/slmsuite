"""
Helper functions for manipulating phase patterns.
"""

import numpy as np
from scipy import special

# Phase pattern collation and manipulation
def imprint(matrix, window, grid, function, imprint_operation="replace",
            centered=False, clip=True, **kwargs):
    """
    Imprints a region (defined by ``window``) of a ``matrix`` with a ``function``.

    Parameters
    ----------
    matrix : numpy.ndarray
        The data to imprint a ``function`` onto.
    window : (int, int, int, int) OR (array_like, array_like) OR array_like
        A number of formats are accepted:
        
        - List in ``(v.x, w, v.y, h)`` format, where ``w`` and ``h`` are the width and height of
          the region and  ``v`` is the lower left coordinate. If ``centered``, then ``v`` is
          instead the center of the region to imprint.
        - Tuple containing arrays of identical length corresponding to y and x indices.
        - Boolean array of same ``shape`` as ``matrix``; the window is defined where ``True`` pixels are.
    
    grid : 2-tuple of numpy.ndarray of floats OR :class:`~slmsuite.hardware.slms.slm.SLM`
        Meshgrids of normalized (x/lambda) coordinates for SLM pixels, in (x_grid, y_grid) form.
        These are precalculated and stored in any :class:`~slmsuite.hardware.slms.slm.SLM`, so
        such a class can be passed instead of the grids directly.
    function : lambda
        A function in the style of :mod:`~slmsuite.holography.lcos_toolbox` helper functions,
        which accept ``grid`` as the first argument.
    imprint_operation : str {"replace" OR "add"}
        Decides how the ``function`` is imparted to the ``matrix``.
        - If ``"replace"``, then the values of ``matrix`` inside ``window`` are replaced with ``function``.
        - If ``"add"``, then these are instead added together (useful, for instance, for global blazes).
    centered : bool
        See ``window``.
    clip : bool
        Whether to clip the imprint region if it exceeds the size of ``matrix``.
        If ``False``, then an error is raised when the size is exceeded.
    **kwargs :
        For passing additional arguments accepted by ``function``.

    Returns
    ----------
    matrix : numpy.ndarray
        The modified image. Note that the matrix is modified in place, and this return
        is merely a copy of the user's pointer to the data.

    Raises
    ----------
    ValueError
        If invalid ``window`` or ``imprint_operation`` are provided.
    """
    (x_grid, y_grid) = _process_grid(grid)

    if len(window) == 4:    # (v.x, w, v.y, h) format
        # Prepare helper vars
        xi = int(     window[0] - (window[1]/2 if centered else 0))
        xf = int(xi + window[1])
        yi = int(     window[2] - (window[3]/2 if centered else 0))
        yf = int(yi + window[3])

        if xi < 0:
            if clip:    xi = 0
            else:       raise ValueError()
        if xf >= matrix.shape[1]:
            if clip:    xf = matrix.shape[1] - 1
            else:       raise ValueError()
        if yi < 0:
            if clip:    yi = 0
            else:       raise ValueError()
        if yf >= matrix.shape[0]:
            if clip:    yf = matrix.shape[0] - 1
            else:       raise ValueError()

        # Modify the matrix
        if imprint_operation == "replace":
            matrix[yi:yf,xi:xf] = function((x_grid[yi:yf,xi:xf], y_grid[yi:yf,xi:xf]), **kwargs)
        elif imprint_operation == "add":
            matrix[yi:yf,xi:xf] = matrix[yi:yf,xi:xf] + function((x_grid[yi:yf,xi:xf], y_grid[yi:yf,xi:xf]), **kwargs)
        else:
            raise ValueError()
    elif len(window) == 2:  # (y_ind, x_ind) format
        # Prepare the lists
        y_ind = np.ravel(window[0])
        x_ind = np.ravel(window[1])

        if clip:
            if any(x_ind < 0):
                x_ind[x_ind < 0] = 0
            if any(x_ind >= matrix.shape[1]):
                x_ind[x_ind >= matrix.shape[1]] = matrix.shape[1] - 1
            if any(y_ind < 0):
                y_ind[y_ind < 0] = 0
            if any(y_ind >= matrix.shape[0]):
                x_ind[y_ind >= matrix.shape[0]] = matrix.shape[0] - 1
        else:
            pass    # Allow the indexing to fail, if it clips...

        # Modify the matrix
        if imprint_operation == "replace":
            matrix[y_ind, x_ind] = function((x_grid[y_ind, x_ind], y_grid[y_ind, x_ind]), **kwargs)
        elif imprint_operation == "add":
            matrix[y_ind, x_ind] = matrix[y_ind, x_ind] + function((x_grid[y_ind, x_ind], y_grid[y_ind, x_ind]), **kwargs)
        else:
            raise ValueError()
    elif np.shape(window) == np.shape(matrix):      # Boolean numpy array. Future: extra checks?
        
        # Modify the matrix
        if imprint_operation == "replace":
            matrix[window] = function((x_grid[window], y_grid[window]), **kwargs)
        elif imprint_operation == "add":
            matrix[window] = matrix[window] + function((x_grid[window], y_grid[window]), **kwargs)
        else:
            raise ValueError()
    else:
        raise ValueError()

    return matrix

# Unit and vector helper functions
blaze_units = ["norm", "kxy", "rad", "knm", "freq", "lpmm", "mrad", "deg"]
def convert_blaze_vector(vector, from_units="norm", to_units="norm", slm=None, shape=None):
        r"""
        Helper function for unit conversions.

        Currently supported units:

          ``"norm"``, ``"kxy"``
            Blaze :math:`k_x` normalized to wavenumber :math:`k`, i.e. :math:`\frac{k_x}{k}`.
            Equivalent to radians in the small angle approximation.
            This is the default unit for :mod:`slmsuite`.
          ``"knm"``
            Computational blaze units for a given Fourier domain ``shape``, zero-centered.
            This corresponds to integer points on the grid of this padded SLM's Fourier transform.
            See :class:`~slmsuite.holography.Hologram`.
          ``"freq"``
            Pixel frequency of a grating producing the blaze.
            e.g. 1/16 is a grating with a period of 16 pixels.
          ``"lpmm"``
            Line pairs per mm or lines per mm of a grating producing the blaze.
          ``"rad"``, ``"mrad"``, ``"deg"``
            Angle at which light is blazed in various units. Small angle approximation is assumed.

        Warning
        ~~~~~~~~
        The units ``"freq"``, ``"knm"``, and ``"lpmm"`` depend on SLM pixel size,
        so a ``slm`` should be passed. ``"knm"`` additionally requires the ``shape`` of
        the computational space. If these arguments are not included, the function returns
        an array ``nan`` values of the same shape as a valid result.

        Parameters
        ----------
        vector : array_like
            2-vectors for which we want to convert units, from ``from_units`` to ``to_units``.
            Processed according to :meth:`clean_2vectors()`.
        from_units, to_units : str
            Units which we are converting between. See the listed units above for options.
            Defaults to ``"norm"``.
        slm : :class:`~slmsuite.hardware.slms.slm.SLM` OR None
            Relevant SLM to pull data from in the case of
            ``"freq"``, ``"knm"``, or ``"lpmm"``.
        shape : (int, int) OR None
            Shape of the computational SLM space.

        Returns
        --------
        numpy.ndarray
            Result of the unit conversion, in the cleaned format of :meth:`clean_2vectors()`.
        """
        assert from_units in blaze_units and to_units in blaze_units

        vector = clean_2vectors(vector).astype(np.float)

        if from_units == "freq" or to_units == "freq":
            if slm is None:
                pitch_um = np.nan
            else:
                pitch_um = clean_2vectors([slm.dx_um, slm.dy_um])

        if from_units in ["freq", "lpmm"] or to_units in ["freq", "lpmm"]:
            if slm is None:
                wav_um = np.nan
            else:
                wav_um = slm.wav_um

        if from_units == "knm" or to_units == "knm":
            if slm is None:
                pitch = np.nan
            else:
                pitch = clean_2vectors([slm.dx, slm.dy])

            if shape is None:
                shape = np.nan
            else:
                shape = clean_2vectors(np.flip(np.squeeze(shape)))

            knm_conv = pitch * shape

        if      from_units == "norm" or \
                from_units == "kxy" or \
                from_units == "rad":  rad = vector
        elif    from_units == "knm":  rad = vector / knm_conv
        elif    from_units == "freq": rad = vector * wav_um / pitch_um
        elif    from_units == "lpmm": rad = vector * wav_um / 1000
        elif    from_units == "mrad": rad = vector / 1000
        elif    from_units == "deg":  rad = vector * np.pi / 180

        if      to_units == "norm" or \
                to_units == "kxy" or \
                to_units == "rad":  return rad
        elif    to_units == "knm":  return rad * knm_conv
        elif    to_units == "freq": return rad * pitch_um / wav_um
        elif    to_units == "lpmm": return rad * 1000 / wav_um
        elif    to_units == "mrad": return rad * 1000
        elif    to_units == "deg":  return rad * 180 / np.pi
def print_blaze_conversions(vector, from_units="norm", **kwargs):
    """
    Helper function to understand unit conversions.
    Prints all the supported unit conversions for a given vector.
    See :meth:`convert_blaze_vector()`.

    Parameters
    ----------
    vector : array_like
        Vector to convert. See :meth:`clean_2vectors()` for format.
    from_units : str
        Units of ``vector``, i.e. units to convert from.
    **kwargs
        Passed to :meth:`convert_blaze_vector()`.
    """
    for unit in blaze_units:
        result = convert_blaze_vector(vector,
            from_units=from_units, to_units=unit, **kwargs)

        print("'{}' : {}".format(unit, tuple(result.T[0])))
def clean_2vectors(vectors):
    """
    Makes sure a 2-vector or array of 2-vectors is arranged appropriately.

    Parameters
    ----------
    vectors : array_like
        2-vector or array of 2-vectors to process. Shape of ``(2, N)``.

    Returns
    -------
    vectors : numpy.ndarray
        Cleaned column vector.

    Raises
    ------
    AssertionError
        If the vector input was inappropriate.
    """
    # Convert to np.array and squeeze
    vectors = np.squeeze(vectors)

    if vectors.shape == (2,):
        vectors = vectors[:, np.newaxis].T

    # Handle the transposed case.
    if vectors.shape == (1, 2):
        vectors = vectors.T

    # Make sure that we are an array of 2-vectors.
    assert len(vectors.shape) == 2
    assert vectors.shape[0] == 2

    return vectors
# def get_affine_vectors(v0, v1, v2, N, mode="delta"):
#     """
#     Returns vectors corresponding to the points in an affine grid.

#     Parameters
#     ----------
#     v0, v1, v2 : array_like
#         2-vectors defining the affine transformation. ``v0`` is always the base/origin.
#         The action of ``v1`` and ``v2`` depends on ``mode``.
#         Cleaned with :meth:`~slmsuite.holography.lcos_toolbox.clean_2vectors()`.
#     N : int or (int, int)
#         Size of the grid ``(N1, N2)``. If a scalar is passed, then the grid is assumed square.
#     mode : {"delta", "corner"}
#         If ``"delta"``, then ``v1`` and ``v2`` are **differences** between the origin at 
#         index ``(0,0)`` and two closest indices ``(1,0)`` and ``(0,1)``, respectively.
#         Otherwise, ``v1`` and ``v2`` are the **positions** of the corners of the grid.

#     Returns
#     -------
#     numpy.ndarray
#         2-vector or array of 2-vectors in slm coordinates.
#     """
#     v0 = clean_2vectors(v0)
#     v1 = clean_2vectors(v1)
#     v2 = clean_2vectors(v2)

#     if isinstance(N, int):
#         N = (N, N)

#     if mode == "corner":
#         v1 = (v1 - v0)/N[0]
#         v2 = (v2 - v0)/N[1]

#     M = np.squeeze(np.array([[v1[0], v2[0]], [v1[1], v2[1]]]))
#     b = v0

#     x_list = np.arange(N[0])
#     y_list = np.arange(N[1])

#     x_grid, y_grid = np.meshgrid(x_list, y_list)
#     indices = np.vstack((x_grid.ravel(), y_grid.ravel()))

#     return np.matmul(M, indices) + b
def get_affine_vectors(v0, v1, v2, N, i0=(0,0), i1=(1,0), i2=(0,1)):
    """
    Returns vectors corresponding to the points in an affine grid.

    Parameters
    ----------
    v0, v1, v2 : array_like
        2-vectors defining the affine transformation. ``v0`` is always the base/origin.
        The action of ``v1`` and ``v2`` depends on ``mode``.
        Cleaned with :meth:`~slmsuite.holography.lcos_toolbox.clean_2vectors()`.
    N : int or (int, int)
        Size of the grid ``(N1, N2)``. If a scalar is passed, then the grid is assumed square.
    i0, i1, i2 : array_like OR None
        Should not be colinear.
        If ``i0`` is ``None``, defaults to the origin ``(0,0)``.
        If ``i1`` or ``i2`` are ``None``, ``v1`` or ``v2`` are interpreted as
        **differences** between ``(0,0)`` and ``(1,0)`` or ``(0,0)`` and ``(0,1)``,
        respectively, instead of as positions.

    Returns
    -------
    numpy.ndarray
        2-vector or array of 2-vectors in slm coordinates.
    """
    v0 = clean_2vectors(v0)
    v1 = clean_2vectors(v1)
    v2 = clean_2vectors(v2)

    if i0 is None:
        i0 = (0,0)
    i0 = clean_2vectors(i0)

    if i1 is None:
        i1 = i0 + clean_2vectors((1,0))
        v1 = v0 + v1
    else:
        i1 = clean_2vectors(i1)

    if i2 is None:
        i2 = i0 + clean_2vectors((0,1))
        v2 = v0 + v2
    else:
        i2 = clean_2vectors(i2)

    di1 = i1 - i0
    di2 = i2 - i0

    colinear = np.abs(np.sum(di1 * di2)) == np.sqrt(np.sum(di1 * di1)*np.sum(di2 * di2))
    assert not colinear, "Indices must not be colinear."

    # Deal with N and make indices.
    indices = None

    if isinstance(N, int):
        N = (N, N)
    elif len(N) == 2 and isinstance(N[0], int) and isinstance(N[1], int):
        pass
    elif isinstance(N, np.ndarray):
        indices = N
    else:
        raise ValueError("N={} not recognized.".format(N))

    if indices is None:
        x_list = np.arange(N[0])
        y_list = np.arange(N[1])

        x_grid, y_grid = np.meshgrid(x_list, y_list)
        indices = np.vstack((x_grid.ravel(), y_grid.ravel()))

    # Construct the matrix and return the vectors.
    M = np.squeeze(np.array([[v1[0], v2[0]], [v1[1], v2[1]]]))
    b = v0

    return np.matmul(M, indices - i0) + b

# Basic functions
def _process_grid(grid):
    """
    Functions in :mod:`.lcos_toolbox` make use of normalized meshgrids containing the normalized
    coordinate of each corresponding pixel. This helper function interprets what the user passes.

    Parameters
    ----------
    grid : 2-tuple of numpy.ndarray of floats OR :class:`~slmsuite.hardware.slms.slm.SLM`
        Meshgrids of normalized (x/lambda) coordinates for SLM pixels, in (x_grid, y_grid) form.
        These are precalculated and stored in any :class:`~slmsuite.hardware.slms.slm.SLM`, so
        such a class can be passed instead of the grids directly.

    Returns
    --------
    2-tuple of numpy.ndarray of floats
        The grids in `(x_grid, y_grid)` form.
    """

    # See if grid has x_grid or y_grid (==> SLM class)
    try:
        return (grid.x_grid, grid.y_grid)
    except:
        pass

    # Otherwise, assume it's a tuple
    assert len(grid) == 2, "Expected a 2-tuple with x and y meshgrids."

    return grid

def blaze(grid, vector=[0,0], offset=0):
    r"""
    Returns a simple blaze (phase ramp).
    
    .. math:: \phi(\vec{x}) = 2\pi \times \vec{k}_g\cdot\vec{x} + o

    Parameters
    ----------
    grid : 2-tuple of numpy.ndarray of floats OR :class:`~slmsuite.hardware.slms.slm.SLM`
        Meshgrids of normalized (x/lambda) coordinates for SLM pixels, in (x_grid, y_grid) form.
        These are precalculated and stored in any :class:`~slmsuite.hardware.slms.slm.SLM`, so
        such a class can be passed instead of the grids directly.
    vector : (float, float)
        Blaze vector in normalized (kx/k) units.
        See :meth:`~slmsuite.holography.lcos_toolbox.convert_blaze_vector()`
    offset :
        Phase offset for this blaze.
    """
    (x_grid, y_grid) = _process_grid(grid)

    return 2 * np.pi * (vector[0]*x_grid + vector[1]*y_grid) + offset
def lens(grid, f, center=[0, 0]):
    r"""
    Returns a simple thin lens (parabolic).
    
    .. math:: \phi(\vec{x}) = \frac{\pi}{f}(|\vec{x}|^2)

    Parameters
    ----------
    grid : 2-tuple of numpy.ndarray of floats OR :class:`~slmsuite.hardware.slms.slm.SLM`
        Meshgrids of normalized (x/lambda) coordinates for SLM pixels, in (x_grid, y_grid) form.
        These are precalculated and stored in any :class:`~slmsuite.hardware.slms.slm.SLM`, so
        such a class can be passed instead of the grids directly.
    f : float or 2-float
        Focus in normalized (kx/k) units.
        Future: add convert_focal_length method to parallel :meth:`.convert_blaze_vector()`
    center : 2-float
        Center of the lens in normalized (x/lambda) coordinates.
    """
    (x_grid, y_grid) = _process_grid(grid)

    if isinstance(f, (int, float)):
        f = [f, f]
    elif isinstance(f, list):
        assert len(f) == 2
    else:
        raise ValueError("Expected f to be a tuple of length 1 (f = fx = fy) or 2 ([fx, fy])")

    return np.pi * (np.square(x_grid - center[0]) / f[0] + np.square(y_grid - center[1]) / f[1])

# Structured light
def determine_source_radius(grid, w=None):
    """
    Helper function to determine the assumed Gaussian source radius for various
    structured light conversion functions.  For instance, see the ``w`` parameter in 
    :meth:`~slmsuite.holography.lcos_toolbox.laguerre_gaussian()`.

    Note
    ~~~~
    Future work: when ``grid`` is a :class:`~slmsuite.hardware.slms.slm.SLM` which has completed
    :meth:`~slmsuite.hardware.cameraslm.FourierSLM.fourier_calibration()`, this function should fit
    (and cache?) :attr:`~slmsuite.hardware.slms.slm.amplitude_measured` to a Gaussian
    and use the resulting width (and center?).

    Parameters
    ----------
    grid : 2-tuple of numpy.ndarray of floats OR :class:`~slmsuite.hardware.slms.slm.SLM`
        Meshgrids of normalized (x/lambda) coordinates for SLM pixels, in (x_grid, y_grid) form.
        These are precalculated and stored in any :class:`~slmsuite.hardware.slms.slm.SLM`, so
        such a class can be passed instead of the grids directly.
    w : float OR None
        The radius of the phase pattern in normalized (x/lambda) units.
        To produce perfect structured beams, this radius is equal to the radius of
        the gaussian profile of the source (ideally not clipped by the SLM).
        If ``w`` is left as ``None``, ``w`` is set to a quarter of the smallest normalized screen dimension.
    """
    (x_grid, y_grid) = _process_grid(grid)

    if w is None:
        return np.min([np.amax(x_grid), np.amax(y_grid)])/4
    else:
        return w

def laguerre_gaussian(grid, l, p, w=None):
    """
    Returns the phase farfield for a Laguerre-Gaussian beam.
    This function is especially useful to hone and validate SLM alignment. Perfect alignment will
    result in concentric and uniform fringes for higher order beams. Focusing issues, aberration,
    or pointing misalignment will mitigate this.

    Parameters
    ----------
    grid : 2-tuple of numpy.ndarray of floats OR :class:`~slmsuite.hardware.slms.slm.SLM`
        Meshgrids of normalized (x/lambda) coordinates for SLM pixels, in (x_grid, y_grid) form.
        These are precalculated and stored in any :class:`~slmsuite.hardware.slms.slm.SLM`, so
        such a class can be passed instead of the grids directly.
    l : int
        The azimuthal wavenumber, or orbital angular momentum. Can be negative.
    p : int
        The radial wavenumber. Should be non-negative.
    w : float OR None
        See :meth:`~slmsuite.holography.lcos_toolbox.determine_source_radius()`.
    """
    (x_grid, y_grid) = _process_grid(grid)

    w = determine_source_radius(grid, w)

    theta_grid = np.arctan2(x_grid, y_grid)
    radius_grid = y_grid*y_grid + x_grid*x_grid

    return np.mod(l*theta_grid + np.pi * np.heaviside(-special.genlaguerre(p, np.abs(l))(2*radius_grid/w/w), 0) + np.pi, 2*np.pi)
def hermite_gaussian(grid, nx, ny, w=None):
    """
    **(NotImplemented)** Returns the phase farfield for a Hermite-Gaussian beam.

    Parameters
    ----------
    grid : 2-tuple of numpy.ndarray of floats OR :class:`~slmsuite.hardware.slms.slm.SLM`
        Meshgrids of normalized (x/lambda) coordinates for SLM pixels, in (x_grid, y_grid) form.
        These are precalculated and stored in any :class:`~slmsuite.hardware.slms.slm.SLM`, so
        such a class can be passed instead of the grids directly.
    nx, ny : int
        The horizontal (`nx`) and vertical (`ny`) wavenumbers. `nx = ny = 0` yields a flat
        phase or a standard Gaussian beam.
    w : float
        See :meth:`~slmsuite.holography.lcos_toolbox.determine_source_radius()`.
    """
    (x_grid, y_grid) = _process_grid(grid)

    w = determine_source_radius(grid, w)
    raise NotImplementedError()
def ince_gaussian(grid, p, m, w=None):
    """
    **(NotImplemented)** Returns the phase farfield for a Ince-Gaussian beam.
    Ref: https://doi.org/10.1364/OL.29.000144
    """
    (x_grid, y_grid) = _process_grid(grid)

    w = determine_source_radius(grid, w)
    raise NotImplementedError()
def matheui_gaussian(grid, r, q, w=None):
    """
    **(NotImplemented)** Returns the phase farfield for a Matheui-Gaussian beam.
    Ref: https://doi.org/10.1364/AO.49.006903
    """
    (x_grid, y_grid) = _process_grid(grid)

    w = determine_source_radius(grid, w)
    raise NotImplementedError()

# Padding
def pad(matrix, shape):
    """
    Helper function to pad data with zeros. The padding is centered.
    This is used to get higher resolution upon Fourier transform.

    Parameters
    ----------
    matrix : numpy.ndarray
        Data to pad.
    shape : (int, int)
        Shape to pad into.
    """
    deltashape = ((shape[0] - matrix.shape[0])/2., (shape[1] - matrix.shape[1])/2.)

    assert deltashape[0] >= 0 and deltashape[1] >= 0, \
        "Shape {} is too large to pad to shape {}".format(tuple(matrix.shape), shape)

    padB = int(np.floor(deltashape[0]))
    padT = int(np.ceil( deltashape[0]))
    padL = int(np.floor(deltashape[1]))
    padR = int(np.ceil( deltashape[1]))

    toReturn = np.pad(matrix, [(padB, padT), (padL, padR)], mode='constant', constant_values=0)

    assert np.all(toReturn.shape == shape)

    return toReturn
def unpad(matrix, shape):
    """
    Helper function to unpad data. The padding is assumed to be centered.
    This is used to get higher resolution upon Fourier transform.

    Parameters
    ----------
    matrix : numpy.ndarray OR (int, int)
        Data to unpad. If this is a shape, return the slicing integers used to unpad that shape
        ``[padB:padT, padL:padR]``.
    shape : (int, int)
        Shape to unpad into.

    Returns
    ----------
    numpy.ndarray OR (int, int, int, int)
        Either the unpadded matrix or the integers used to unpad such a matrix.
    """
    mshape = np.shape(matrix)
    return_args = False
    if len(mshape) == 1 or np.prod(mshape) == 2:
        # Assume as tuple was provided.
        mshape = np.squeeze(matrix)
        return_args = True

    deltashape = ((shape[0] - mshape[0])/2., (shape[1] - mshape[1])/2.)

    assert deltashape[0] <= 0 and deltashape[1] <= 0, \
        "Shape {} is too small to unpad to shape {}".format(tuple(mshape), shape)

    padB = int(np.floor(-deltashape[0]))
    padT = int(mshape[0] - np.ceil( -deltashape[0]))
    padL = int(np.floor(-deltashape[1]))
    padR = int(mshape[1] - np.ceil( -deltashape[1]))

    if return_args:
        return (padB, padT, padL, padR)

    toReturn = matrix[padB:padT, padL:padR]

    assert np.all(toReturn.shape == shape)

    return toReturn
