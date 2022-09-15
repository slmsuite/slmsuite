"""
Helper functions for manipulating phase patterns.
"""

import numpy as np
from scipy import special
from scipy.spatial.distance import cityblock
from scipy.spatial import Voronoi, voronoi_plot_2d
import cv2
import matplotlib.pyplot as plt

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
          ``centered`` is ignored.
        - Boolean array of same ``shape`` as ``matrix``; the window is defined where ``True`` pixels are.
          ``centered`` is ignored.

    grid : (array_like, array_like) OR :class:`~slmsuite.hardware.slms.slm.SLM`
        Meshgrids of normalized (x/lambda) coordinates for SLM pixels, in (x_grid, y_grid) form.
        These are precalculated and stored in any :class:`~slmsuite.hardware.slms.slm.SLM`, so
        such a class can be passed instead of the grids directly.
    function : lambda
        A function in the style of :mod:`~slmsuite.holography.toolbox` helper functions,
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
            matrix[yi:yf,xi:xf] =   function(  (x_grid[yi:yf,xi:xf],
                                                y_grid[yi:yf,xi:xf]), **kwargs)
        elif imprint_operation == "add":
            matrix[yi:yf,xi:xf] +=  function(  (x_grid[yi:yf,xi:xf],
                                                y_grid[yi:yf,xi:xf]), **kwargs)
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
            matrix[y_ind, x_ind] =  function(  (x_grid[y_ind, x_ind],
                                                y_grid[y_ind, x_ind]), **kwargs)
        elif imprint_operation == "add":
            matrix[y_ind, x_ind] += function(  (x_grid[y_ind, x_ind],
                                                y_grid[y_ind, x_ind]), **kwargs)
        else:
            raise ValueError()
    elif np.shape(window) == np.shape(matrix):      # Boolean numpy array. Future: extra checks?

        # Modify the matrix
        if imprint_operation == "replace":
            matrix[window] =  function((x_grid[window], y_grid[window]), **kwargs)
        elif imprint_operation == "add":
            matrix[window] += function((x_grid[window], y_grid[window]), **kwargs)
        else:
            raise ValueError()
    else:
        raise ValueError()

    return matrix

# Unit helper functions
blaze_units = ["norm", "kxy", "rad", "knm", "freq", "lpmm", "mrad", "deg"]
def convert_blaze_vector(   vector, from_units="norm", to_units="norm",
                            slm=None, shape=None):
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

        if     (from_units == "norm" or
                from_units == "kxy" or
                from_units == "rad"): rad = vector
        elif    from_units == "knm":  rad = vector / knm_conv
        elif    from_units == "freq": rad = vector * wav_um / pitch_um
        elif    from_units == "lpmm": rad = vector * wav_um / 1000
        elif    from_units == "mrad": rad = vector / 1000
        elif    from_units == "deg":  rad = vector * np.pi / 180

        if     (to_units == "norm" or
                to_units == "kxy" or
                to_units == "rad"): return rad
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

# Vector and window helper functions
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
def get_affine_vectors(y0, y1, y2, N, x0=(0,0), x1=(1,0), x2=(0,1)):
    r"""
    Fits three points to an affine transformation. This transformation is given by:

    .. math:: \vec{y} = M \cdot \vec{x} + \vec{b}

    Parameters
    ----------
    y0, y1, y2 : array_like
        2-vectors defining the affine transformation. These vectors correspond to
        positions which we will fit our transformation to. These vectors have
        corresponding indices ``x0``, ``x1``, ``x2``; see these variables for more
        information. With the default values for the indices, ``y0`` is base/origin
        and ``y1`` and ``y2`` are the positions of the first point in
        the ``x`` and ``y`` directions of index-space, respectively.
        Cleaned with :meth:`~slmsuite.holography.toolbox.clean_2vectors()`.
    N : int OR (int, int) OR numpy.ndarray OR None
        Size of the grid of vectors to return ``(N1, N2)``.
        If a scalar is passed, then the grid is assumed square.
        If ``None`` or any non-positive integer is passed, then a dictionary
        with the affine transformation is instead returned.
    x0, x1, x2 : array_like OR None
        Should not be colinear.
        If ``x0`` is ``None``, defaults to the origin ``(0,0)``.
        If ``x1`` or ``x2`` are ``None``, ``y1`` or ``y2`` are interpreted as
        **differences** between ``(0,0)`` and ``(1,0)`` or ``(0,0)`` and ``(0,1)``,
        respectively, instead of as positions.
        Cleaned with :meth:`~slmsuite.holography.toolbox.clean_2vectors()`.

    Returns
    -------
    numpy.ndarray OR dict
        2-vector or array of 2-vectors in slm coordinates.
        If ``N`` is ``None`` or non-positive, then returns a dictionary with keys
        ``"M"`` and ``"b"`` (transformation matrix and shift, respectively).
    """
    # Parse vectors
    y0 = clean_2vectors(y0)
    y1 = clean_2vectors(y1)
    y2 = clean_2vectors(y2)

    # Parse index vectors
    if x0 is None:
        x0 = (0,0)
    x0 = clean_2vectors(x0)

    if x1 is None:
        x1 = x0 + clean_2vectors((1,0))
    else:
        x1 = clean_2vectors(x1)
        y1 = y1 - y0

    if x2 is None:
        x2 = x0 + clean_2vectors((0,1))
    else:
        x2 = clean_2vectors(x2)
        y2 = y2 - y0

    dx1 = x1 - x0
    dx2 = x2 - x0

    # Invert the index matrix.
    colinear = np.abs(np.sum(dx1 * dx2)) == np.sqrt(np.sum(dx1 * dx1)*np.sum(dx2 * dx2))
    assert not colinear, "Indices must not be colinear."

    J = np.linalg.inv(np.squeeze(np.array([[dx1[0], dx2[0]], [dx1[1], dx2[1]]])))

    # Construct the matrix.
    M = np.matmul(np.squeeze(np.array([[y1[0], y2[0]], [y1[1], y2[1]]])), J)
    b = y0 - np.matmul(M, x0)

    # Deal with N and make indices.
    indices = None
    affine_return = False

    if N is None:
        affine_return = True
    elif isinstance(N, int):
        if N <= 0:
            affine_return = True
        else:
            N = (N, N)
    elif len(N) == 2 and isinstance(N[0], int) and isinstance(N[1], int):
        if N[0] <= 0 or N[1] <= 0:
            affine_return = True
        else:
            pass
    elif isinstance(N, np.ndarray):
        indices = clean_2vectors(N)
    else:
        raise ValueError("N={} not recognized.".format(N))

    if affine_return:
        return {"M":M, "b":b}
    else:
        if indices is None:
            x_list = np.arange(N[0])
            y_list = np.arange(N[1])

            x_grid, y_grid = np.meshgrid(x_list, y_list)
            indices = np.vstack((x_grid.ravel(), y_grid.ravel()))

        return np.matmul(M, indices) + b
def get_smallest_distance(vectors, metric=cityblock):
    """
    Returns the smallest distance between pairs of points under a given ``metric``.

    Note
    ~~~~
    An :math:`\mathcal{O}(N^2)` brute force approach is currently implemented.
    Future work will involve an :math:`\mathcal{O}(N\log(N))`
    divide and conquer algorithm.

    Parameters
    ----------
    vectors : array_like
        Points to compare.
        Cleaned with :meth:`~slmsuite.holography.toolbox.clean_2vectors()`.
    metric : lambda
        Function to use to compare.
        Defaults to :meth:`scipy.spatial.distance.cityblock`.
    """
    vectors = clean_2vectors(vectors)
    N = vectors.shape[1]

    minimum = np.inf

    for x in range(N-1):
        for y in range(x+1, N):
            distance = metric(vectors[:,x], vectors[:,y])
            if distance < minimum:
                minimum = distance

    return minimum
def get_voronoi_windows(grid, vectors, radius=None, plot=True):
    """
    Gets boolean array windows for an array of vectors in the style of
    :meth:`~slmsuite.holography.toolbox.imprint()`,
    such that the ith window corresponds to the Voronoi cell centered around the ith vector.

    Parameters
    ----------
    grid : (int, int) OR (array_like, array_like) OR :class:`~slmsuite.hardware.slms.slm.SLM`
        Meshgrids of normalized (x/lambda) coordinates for SLM pixels, in (x_grid, y_grid) form.
        These are precalculated and stored in any :class:`~slmsuite.hardware.slms.slm.SLM`, so
        such a class can be passed instead of the grids directly.
        If an ``(int, int)`` is passed, this is assumed to be the shape of the device, and
        ``vectors`` are **assumed to be in pixel units instead of normalized units**.
    vectors : array_like
        Points to Voronoi-ify.
        Cleaned with :meth:`~slmsuite.holography.toolbox.clean_2vectors()`.
    radius : float
        Cells on the edge of the set of cells might be very large. This parameter bounds
        the cells with a boolean and to the aperture of the given ``radius``.
    plot : bool
        Whether to plot the resulting Voronoi diagram with :meth:`scipy.spatial.voronoi_plot_2d()`.

    Returns
    -------
    list of numpy.ndarray
        The resulting windows.
    """
    vectors = clean_2vectors(vectors)

    if (isinstance(grid, (list, tuple)) and
        isinstance(grid[0], (int)) and
        isinstance(grid[1], (int))):
        shape = grid
    else:
        (x_grid, y_grid) = _process_grid(grid)

        shape = x_grid.shape

        x_list = x_grid[0,:]
        y_list = y_grid[:,0]

        vectors = np.vstack(np.interp(vectors[0,:], x_list, np.arange(shape[1])),
                            np.interp(vectors[1,:], y_list, np.arange(shape[0])))

    # Half shape data.
    hsx = shape[1]/2
    hsy = shape[0]/2

    vectors_voronoi = np.concatenate((vectors.T,
        np.array([ [hsx, -3*hsy], [hsx, 5*hsy], [-3*hsx, hsy], [5*hsx, hsy] ])))

    vor = Voronoi(vectors_voronoi, furthest_site=False)

    if plot:
        sx = shape[1]
        sy = shape[0]

        fig = voronoi_plot_2d(vor)

        plt.plot(   np.array([0, sx, sx, 0, 0]),
                    np.array([0, 0, sy, sy, 0]), 'r')

        plt.xlim(-.05*sx, 1.05*sx)
        plt.ylim(1.05*sy, -.05*sy)

        plt.title("Voronoi Cells")

        plt.show()

    N = np.shape(vectors)[1]
    filled_regions = []

    for x in range(N):
        point = vor.points[x]
        region = vor.regions[vor.point_region[x]]
        pts = vor.vertices[region].astype(np.int32)

        canvas1 = np.zeros(shape, dtype=np.uint8)
        cv2.fillConvexPoly(canvas1, pts, 255, cv2.LINE_4)

        if radius is not None and radius > 0:
            canvas2 = np.zeros(shape, dtype=np.uint8)
            cv2.circle(canvas2, tuple(point.astype(np.int32)),
                                int(np.ceil(radius)), 255, -1)

            filled_regions.append((canvas1 > 0) & (canvas2 > 0))
        else:
            filled_regions.append(canvas1 > 0)

    return filled_regions

# Basic functions
def _process_grid(grid):
    """
    Functions in :mod:`.toolbox` make use of normalized meshgrids containing the normalized
    coordinate of each corresponding pixel. This helper function interprets what the user passes.

    Parameters
    ----------
    grid : (array_like, array_like) OR :class:`~slmsuite.hardware.slms.slm.SLM`
        Meshgrids of normalized (x/lambda) coordinates for SLM pixels, in (x_grid, y_grid) form.
        These are precalculated and stored in any :class:`~slmsuite.hardware.slms.slm.SLM`, so
        such a class can be passed instead of the grids directly.

    Returns
    --------
    (array_like, array_like)
        The grids in ``(x_grid, y_grid)`` form.
    """

    # See if grid has x_grid or y_grid (==> SLM class)
    try:
        return (grid.x_grid, grid.y_grid)
    except:
        pass

    # Otherwise, assume it's a tuple
    assert len(grid) == 2, "Expected a 2-tuple with x and y meshgrids."

    return grid

def blaze(grid, vector=(0,0), offset=0):
    r"""
    Returns a simple blaze (phase ramp).

    .. math:: \phi(\vec{x}) = 2\pi \times \vec{k}_g\cdot\vec{x} + o

    Parameters
    ----------
    grid : (array_like, array_like) OR :class:`~slmsuite.hardware.slms.slm.SLM`
        Meshgrids of normalized (x/lambda) coordinates for SLM pixels, in (x_grid, y_grid) form.
        These are precalculated and stored in any :class:`~slmsuite.hardware.slms.slm.SLM`, so
        such a class can be passed instead of the grids directly.
    vector : (float, float)
        Blaze vector in normalized (kx/k) units.
        See :meth:`~slmsuite.holography.toolbox.convert_blaze_vector()`
    offset :
        Phase offset for this blaze.
    """
    (x_grid, y_grid) = _process_grid(grid)

    return 2 * np.pi * (vector[0]*x_grid + vector[1]*y_grid) + offset
def lens(grid, f=(np.inf, np.inf), center=(0, 0)):
    r"""
    Returns a simple thin lens (parabolic).

    .. math:: \phi(\vec{x}) = \frac{\pi}{f}(|\vec{x}|^2)

    Parameters
    ----------
    grid : (array_like, array_like) OR :class:`~slmsuite.hardware.slms.slm.SLM`
        Meshgrids of normalized (x/lambda) coordinates for SLM pixels, in (x_grid, y_grid) form.
        These are precalculated and stored in any :class:`~slmsuite.hardware.slms.slm.SLM`, so
        such a class can be passed instead of the grids directly.
    f : float OR (float, float) or ((float, float), float)
        Focus in normalized (x/lambda) units. If `((float, float), float)` is given, it is
        interpreted as `((f_x, f_y), ang)` where `ang` is the angle in counter-clockwise
        radians to rotate the lens by.
        Future: add a ``convert_focal_length`` method to parallel :meth:`.convert_blaze_vector()`
    center : (float, float)
        Center of the lens in normalized (x/lambda) coordinates.
    """
    (x_grid, y_grid) = _process_grid(grid)

    # Parse center
    center = np.squeeze(center)

    assert center.shape == (2,)

    # Parse focal length.
    if isinstance(f, (int, float)):
        f = [f, f]
    if isinstance(f, (list, tuple, np.ndarray)):
        f = np.squeeze(f)

        assert f.shape == (2,)

        if isinstance(f[0], (list, tuple, np.ndarray)):
            ang = f[1]

            f = np.squeeze(f[0]).astype(np.float)
            assert f.shape == (2,)
        else:
            ang = 0

        assert not np.any(f == 0), "Cannot interpret a focal length of zero."

        # Optical power of lens
        g = [[1/f[0], 0], [0, 1/f[1]]]

        # Rotate if necessary
        if ang != 0:
            s = np.sin(ang)
            c = np.cos(ang)
            rot = np.array([[c, -s], [s, c]])

            g = np.matmul(np.linalg.inv(rot), np.matmul(g, rot))
    else:
        raise ValueError("Expected f to be a scalar, a vector of length 2, or a 2x2 matrix.")


    # Only add a component if necessary (for speed)
    out = None

    if g[0][0] != 0:
        if out is None:
            out  = np.square(x_grid - center[0]) * (g[0][0] * np.pi)
        else:
            out += np.square(x_grid - center[0]) * (g[0][0] * np.pi)

    if g[1][1] != 0:
        if out is None:
            out  = np.square(y_grid - center[1]) * (g[1][1] * np.pi)
        else:
            out += np.square(y_grid - center[1]) * (g[1][1] * np.pi)

    shear = (g[1][0] + g[0][1]) * np.pi

    if shear != 0:
        if out is None:
            out  = (x_grid - center[0]) * (y_grid - center[1]) * shear
        else:
            out += (x_grid - center[0]) * (y_grid - center[1]) * shear

    return out

# Structured light
def determine_source_radius(grid, w=None):
    """
    Helper function to determine the assumed Gaussian source radius for various
    structured light conversion functions.  For instance, see the ``w`` parameter in
    :meth:`~slmsuite.holography.toolbox.laguerre_gaussian()`.

    Note
    ~~~~
    Future work: when ``grid`` is a :class:`~slmsuite.hardware.slms.slm.SLM` which has completed
    :meth:`~slmsuite.hardware.cameraslm.FourierSLM.fourier_calibration()`, this function should fit
    (and cache?) :attr:`~slmsuite.hardware.slms.slm.amplitude_measured` to a Gaussian
    and use the resulting width (and center?).

    Parameters
    ----------
    grid : (array_like, array_like) OR :class:`~slmsuite.hardware.slms.slm.SLM`
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
    grid : (array_like, array_like) OR :class:`~slmsuite.hardware.slms.slm.SLM`
        Meshgrids of normalized (x/lambda) coordinates for SLM pixels, in (x_grid, y_grid) form.
        These are precalculated and stored in any :class:`~slmsuite.hardware.slms.slm.SLM`, so
        such a class can be passed instead of the grids directly.
    l : int
        The azimuthal wavenumber, or orbital angular momentum. Can be negative.
    p : int
        The radial wavenumber. Should be non-negative.
    w : float OR None
        See :meth:`~slmsuite.holography.toolbox.determine_source_radius()`.
    """
    (x_grid, y_grid) = _process_grid(grid)

    w = determine_source_radius(grid, w)

    theta_grid = np.arctan2(x_grid, y_grid)
    radius_grid = y_grid*y_grid + x_grid*x_grid

    return np.mod(  l*theta_grid +
        np.pi * np.heaviside(-special.genlaguerre(p, np.abs(l))(2*radius_grid/w/w), 0)
                    + np.pi,
                    2*np.pi)
def hermite_gaussian(grid, nx, ny, w=None):
    """
    **(NotImplemented)** Returns the phase farfield for a Hermite-Gaussian beam.

    Parameters
    ----------
    grid : (array_like, array_like) OR :class:`~slmsuite.hardware.slms.slm.SLM`
        Meshgrids of normalized (x/lambda) coordinates for SLM pixels, in (x_grid, y_grid) form.
        These are precalculated and stored in any :class:`~slmsuite.hardware.slms.slm.SLM`, so
        such a class can be passed instead of the grids directly.
    nx, ny : int
        The horizontal (``nx``) and vertical (``ny``) wavenumbers. ``nx = ny = 0`` yields a flat
        phase or a standard Gaussian beam.
    w : float
        See :meth:`~slmsuite.holography.toolbox.determine_source_radius()`.
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

    toReturn = np.pad(  matrix, [(padB, padT), (padL, padR)],
                        mode='constant', constant_values=0)

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
