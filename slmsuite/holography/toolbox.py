"""
Helper functions for manipulating phase patterns.
"""

import numpy as np
from scipy import special
from scipy.spatial.distance import euclidean, chebyshev
from scipy.spatial import Voronoi, voronoi_plot_2d
import cv2
import matplotlib.pyplot as plt
from math import factorial

# Phase pattern collation and manipulation
def imprint(
    matrix,
    window,
    function,
    grid,
    imprint_operation="replace",
    centered=False,
    clip=True,
    **kwargs
):
    r"""
    Imprints a region (defined by ``window``) of a ``matrix`` with a ``function``.
    This ``function`` must be in the style of :mod:`~slmsuite.holography.toolbox`
    phase helper functions, which attempts a ``grid`` parameter
    (see :mod:`~slmsuite.holography.toolbox.blaze()` or
    :mod:`~slmsuite.holography.toolbox.lens()`).
    For instance, we can imprint a blaze on a 200 by 200 pixel region
    of the SLM with:

    .. highlight:: python
    .. code-block:: python

        canvas = np.zeros(shape=slm.shape)  # Matrix to imprint onto.
        window = [200, 200, 200, 200]       # Region of the matrix to imprint.
        toolbox.imprint(canvas, window=window, function=toolbox.blaze, grid=slm, vector=(.001, .001))

    See also :ref:`examples`.

    Parameters
    ----------
    matrix : numpy.ndarray
        The data to imprint a ``function`` onto.
    window : (int, int, int, int) OR (array_like, array_like) OR array_like
        A number of formats are accepted:

        - List in ``(x, w, y, h)`` format, where ``w`` and ``h`` are the width and height of
          the region and  ``(x,y)`` is the lower left coordinate. If ``centered``, then ``(x,y)`` is
          instead the center of the region to imprint.
        - Tuple containing arrays of identical length corresponding to y and x indices.
          ``centered`` is ignored.
        - Boolean array of same ``shape`` as ``matrix``; the window is defined where ``True`` pixels are.
          ``centered`` is ignored.

    function : function
        A function in the style of :mod:`~slmsuite.holography.toolbox` helper functions,
        which accept ``grid`` as the first argument.
    grid : (array_like, array_like) OR :class:`~slmsuite.hardware.slms.slm.SLM`
        Meshgrids of normalized :math:`\frac{x}{\lambda}` coordinates
        corresponding to SLM pixels, in ``(x_grid, y_grid)`` form.
        These are precalculated and stored in any :class:`~slmsuite.hardware.slms.slm.SLM`, so
        such a class can be passed instead of the grids directly.
    imprint_operation : {"replace" OR "add"}
        Decides how the ``function`` is imparted to the ``matrix``.

        - If ``"replace"``, then the values of ``matrix`` inside ``window`` are replaced with ``function``.
        - If ``"add"``, then these are instead added together (useful, for instance, for global blazes).

    centered : bool
        See ``window``. Defaults to ``True``.
    clip : bool
        Whether to clip the imprint region if it exceeds the size of ``matrix``.
        If ``False``, then an error is raised when the size is exceeded.
        If ``True``, then the out-of-range pixels are instead filled with ``numpy.nan``.
        Defaults to ``True``.
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

    if len(window) == 4:  # (v.x, w, v.y, h) format
        # Prepare helper vars
        xi = int(window[0] - (window[1] / 2 if centered else 0))
        xf = int(xi + window[1])
        yi = int(window[2] - (window[3] / 2 if centered else 0))
        yf = int(yi + window[3])

        if xi < 0:
            if clip:
                xi = 0
            else:
                raise ValueError()
        if xf >= matrix.shape[1]:
            if clip:
                xf = matrix.shape[1] - 1
            else:
                raise ValueError()
        if yi < 0:
            if clip:
                yi = 0
            else:
                raise ValueError()
        if yf >= matrix.shape[0]:
            if clip:
                yf = matrix.shape[0] - 1
            else:
                raise ValueError()

        if xf < 0:
            if clip:
                xf = 0
            else:
                raise ValueError()
        if xi >= matrix.shape[1]:
            if clip:
                xi = matrix.shape[1] - 1
            else:
                raise ValueError()
        if yf < 0:
            if clip:
                yf = 0
            else:
                raise ValueError()
        if yi >= matrix.shape[0]:
            if clip:
                yi = matrix.shape[0] - 1
            else:
                raise ValueError()

        # Modify the matrix
        if imprint_operation == "replace":
            matrix[yi:yf, xi:xf] = function(
                (x_grid[yi:yf, xi:xf], y_grid[yi:yf, xi:xf]), **kwargs
            )
        elif imprint_operation == "add":
            matrix[yi:yf, xi:xf] += function(
                (x_grid[yi:yf, xi:xf], y_grid[yi:yf, xi:xf]), **kwargs
            )
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
            pass  # Allow the indexing to fail, if it clips...

        # Modify the matrix
        if imprint_operation == "replace":
            matrix[y_ind, x_ind] = function(
                (x_grid[y_ind, x_ind], y_grid[y_ind, x_ind]), **kwargs
            )
        elif imprint_operation == "add":
            matrix[y_ind, x_ind] += function(
                (x_grid[y_ind, x_ind], y_grid[y_ind, x_ind]), **kwargs
            )
        else:
            raise ValueError()
    elif np.shape(window) == np.shape(
        matrix
    ):  # Boolean numpy array. Future: extra checks?

        # Modify the matrix
        if imprint_operation == "replace":
            matrix[window] = function((x_grid[window], y_grid[window]), **kwargs)
        elif imprint_operation == "add":
            matrix[window] += function((x_grid[window], y_grid[window]), **kwargs)
        else:
            raise ValueError()
    else:
        raise ValueError()

    return matrix


# Unit helper functions
blaze_units = ["norm", "kxy", "rad", "knm", "freq", "lpmm", "mrad", "deg"]


def convert_blaze_vector(
    vector, from_units="norm", to_units="norm", slm=None, shape=None
):
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
            Processed according to :meth:`format_2vectors()`.
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
            Result of the unit conversion, in the cleaned format of :meth:`format_2vectors()`.
        """
    assert from_units in blaze_units and to_units in blaze_units

    vector = format_2vectors(vector).astype(np.float)

    if from_units == "freq" or to_units == "freq":
        if slm is None:
            pitch_um = np.nan
        else:
            pitch_um = format_2vectors([slm.dx_um, slm.dy_um])

    if from_units in ["freq", "lpmm"] or to_units in ["freq", "lpmm"]:
        if slm is None:
            wav_um = np.nan
        else:
            wav_um = slm.wav_um

    if from_units == "knm" or to_units == "knm":
        if slm is None:
            pitch = np.nan
        else:
            pitch = format_2vectors([slm.dx, slm.dy])

        if shape is None:
            shape = np.nan
        else:
            shape = format_2vectors(np.flip(np.squeeze(shape)))

        knm_conv = pitch * shape

    if from_units == "norm" or from_units == "kxy" or from_units == "rad":
        rad = vector
    elif from_units == "knm":
        rad = vector / knm_conv
    elif from_units == "freq":
        rad = vector * wav_um / pitch_um
    elif from_units == "lpmm":
        rad = vector * wav_um / 1000
    elif from_units == "mrad":
        rad = vector / 1000
    elif from_units == "deg":
        rad = vector * np.pi / 180

    if to_units == "norm" or to_units == "kxy" or to_units == "rad":
        return rad
    elif to_units == "knm":
        return rad * knm_conv
    elif to_units == "freq":
        return rad * pitch_um / wav_um
    elif to_units == "lpmm":
        return rad * 1000 / wav_um
    elif to_units == "mrad":
        return rad * 1000
    elif to_units == "deg":
        return rad * 180 / np.pi


def print_blaze_conversions(vector, from_units="norm", **kwargs):
    """
    Helper function to understand unit conversions.
    Prints all the supported unit conversions for a given vector.
    See :meth:`convert_blaze_vector()`.

    Parameters
    ----------
    vector : array_like
        Vector to convert. See :meth:`format_2vectors()` for format.
    from_units : str
        Units of ``vector``, i.e. units to convert from.
    **kwargs
        Passed to :meth:`convert_blaze_vector()`.
    """
    for unit in blaze_units:
        result = convert_blaze_vector(
            vector, from_units=from_units, to_units=unit, **kwargs
        )

        print("'{}' : {}".format(unit, tuple(result.T[0])))


# Vector and window helper functions
def format_2vectors(vectors):
    """
    Validates that an array of 2D vectors is a ``numpy.ndarray`` of shape ``(2, N)``.
    Handles shaping and transposing if, for instance, tuples or row vectors are passed.

    Parameters
    ----------
    vectors : array_like
        2-vector or array of 2-vectors to process. Shape of ``(2, N)``.

    Returns
    -------
    vectors : numpy.ndarray
        Cleaned column vector(s).

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


def fit_affine(y0, y1, y2, N=None, x0=(0, 0), x1=(1, 0), x2=(0, 1)):
    r"""
    Fits three points to an affine transformation. This transformation is given by:

    .. math:: \vec{y} = M \cdot \vec{x} + \vec{b}

    At base, this function finds and optionally uses affine transformations:

    .. highlight:: python
    .. code-block:: python

        y0 = (1.,1.)    # Origin
        y1 = (2.,2.)    # First point in x direction
        y2 = (1.,2.)    # first point in y direction

        # Dict with keys "M", and "b":
        affine_dict =   fit_affine(y0, y1, y2, N=None)

        # Array with shape (2,25) corresponding to a 5x5 evaluation of the above:
        vector_array =  fit_affine(y0, y1, y2, N=(5,5))

    However, ``fit_affine`` is more powerful that this, and can fit an affine
    transformation to semi-arbitrary sets of points with known indices
    in the coordinate  system of the dependent variable :math:`\vec{x}`,
    as long as the passed indices ``x0``, ``x1``, ``x2`` are not colinear.

    .. highlight:: python
    .. code-block:: python

        # y11 is at x index (1,1), etc
        fit_affine(y11, y34, y78, (5,5), (1,1), (3,4), (7,8))

        # These indices don't have to be integers
        fit_affine(a, b, c, (5,5), (np.pi,1.5), (20.5,np.sqrt(2)), (7.7,42.0))

    Optionally, basis vectors can be passed directly instead of adding these
    vectors to the origin, by making use of passing ``None`` for ``x1`` or ``x2``:

    .. highlight:: python
    .. code-block:: python

        origin =    (1.,1.)     # Origin
        dv1 =       (1.,1.)     # Basis vector in x direction
        dv2 =       (1.,0.)     # Basis vector in y direction

        # The following are equivalent:
        option1 = fit_affine(origin, origin+dv1, origin+dv2, (5,5))
        option2 = fit_affine(origin, dv1, dv2, (5,5), x1=None, x2=None)

        assert option1 == option2

    Parameters
    ----------
    y0, y1 : array_like
        See ``y2``.
    y2 : array_like
        2-vectors defining the affine transformation. These vectors correspond to
        positions which we will fit our transformation to. These vectors have
        corresponding indices ``x0``, ``x1``, ``x2``; see these variables for more
        information. With the default values for the indices, ``y0`` is base/origin
        and ``y1`` and ``y2`` are the positions of the first point in
        the ``x`` and ``y`` directions of index-space, respectively.
        Cleaned with :meth:`~slmsuite.holography.toolbox.format_2vectors()`.
    N : int OR (int, int) OR numpy.ndarray OR None
        Size of the grid of vectors to return ``(N1, N2)``.
        If a scalar is passed, then the grid is assumed square.
        If ``None`` or any non-positive integer is passed, then a dictionary
        with the affine transformation is instead returned.
        Defaults to ``None``.
    x0, x1 : array_like OR None
        See ``x2``.
    x2 : array_like OR None
        Should not be colinear.
        If ``x0`` is ``None``, defaults to the origin ``(0,0)``.
        If ``x1`` or ``x2`` are ``None``, ``y1`` or ``y2`` are interpreted as
        **differences** between ``(0,0)`` and ``(1,0)`` or ``(0,0)`` and ``(0,1)``,
        respectively, instead of as positions.
        Cleaned with :meth:`~slmsuite.holography.toolbox.format_2vectors()`.

    Returns
    -------
    numpy.ndarray OR dict
        2-vector or array of 2-vectors ``(2, N)`` in slm coordinates.
        If ``N`` is ``None`` or non-positive, then returns a dictionary with keys
        ``"M"`` and ``"b"`` (transformation matrix and shift, respectively).
    """
    # Parse vectors
    y0 = format_2vectors(y0)
    y1 = format_2vectors(y1)
    y2 = format_2vectors(y2)

    # Parse index vectors
    if x0 is None:
        x0 = (0, 0)
    x0 = format_2vectors(x0)

    if x1 is None:
        x1 = x0 + format_2vectors((1, 0))
    else:
        x1 = format_2vectors(x1)
        y1 = y1 - y0

    if x2 is None:
        x2 = x0 + format_2vectors((0, 1))
    else:
        x2 = format_2vectors(x2)
        y2 = y2 - y0

    dx1 = x1 - x0
    dx2 = x2 - x0

    # Invert the index matrix.
    colinear = np.abs(np.sum(dx1 * dx2)) == np.sqrt(
        np.sum(dx1 * dx1) * np.sum(dx2 * dx2)
    )
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
        indices = format_2vectors(N)
    else:
        raise ValueError("N={} not recognized.".format(N))

    if affine_return:
        return {"M": M, "b": b}
    else:
        if indices is None:
            x_list = np.arange(N[0])
            y_list = np.arange(N[1])

            x_grid, y_grid = np.meshgrid(x_list, y_list)
            indices = np.vstack((x_grid.ravel(), y_grid.ravel()))

        return np.matmul(M, indices) + b


def smallest_distance(vectors, metric=chebyshev):
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
        Cleaned with :meth:`~slmsuite.holography.toolbox.format_2vectors()`.
    metric : lambda
        Function to use to compare.
        Defaults to :meth:`scipy.spatial.distance.chebyshev()`.
        :meth:`scipy.spatial.distance.euclidean()` is also common.
    """
    vectors = format_2vectors(vectors)
    N = vectors.shape[1]

    minimum = np.inf

    for x in range(N - 1):
        for y in range(x + 1, N):
            distance = metric(vectors[:, x], vectors[:, y])
            if distance < minimum:
                minimum = distance

    return minimum


def voronoi_windows(grid, vectors, radius=None, plot=False):
    r"""
    Gets boolean array windows for an array of vectors in the style of
    :meth:`~slmsuite.holography.toolbox.imprint()`,
    such that the ith window corresponds to the Voronoi cell centered around the ith vector.

    Parameters
    ----------
    grid : (array_like, array_like) OR :class:`~slmsuite.hardware.slms.slm.SLM` OR (int, int)
        Meshgrids of normalized :math:`\frac{x}{\lambda}` coordinates
        corresponding to SLM pixels, in ``(x_grid, y_grid)`` form.
        These are precalculated and stored in any :class:`~slmsuite.hardware.slms.slm.SLM`, so
        such a class can be passed instead of the grids directly.
        If an ``(int, int)`` is passed, this is assumed to be the shape of the device, and
        ``vectors`` are **assumed to be in pixel units instead of normalized units**.
    vectors : array_like
        Points to Voronoi-ify.
        Cleaned with :meth:`~slmsuite.holography.toolbox.format_2vectors()`.
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
    vectors = format_2vectors(vectors)

    if (
        isinstance(grid, (list, tuple))
        and isinstance(grid[0], (int))
        and isinstance(grid[1], (int))
    ):
        shape = grid
    else:
        (x_grid, y_grid) = _process_grid(grid)

        shape = x_grid.shape

        x_list = x_grid[0, :]
        y_list = y_grid[:, 0]

        vectors = np.vstack((
            np.interp(vectors[0, :], x_list, np.arange(shape[1])),
            np.interp(vectors[1, :], y_list, np.arange(shape[0])),
        ))

    # Half shape data.
    hsx = shape[1] / 2
    hsy = shape[0] / 2

    vectors_voronoi = np.concatenate(
        (
            vectors.T,
            np.array(
                [[hsx, -3 * hsy], [hsx, 5 * hsy], [-3 * hsx, hsy], [5 * hsx, hsy]]
            ),
        )
    )

    vor = Voronoi(vectors_voronoi, furthest_site=False)

    if plot:
        sx = shape[1]
        sy = shape[0]

        fig = voronoi_plot_2d(vor)

        plt.plot(np.array([0, sx, sx, 0, 0]), np.array([0, 0, sy, sy, 0]), "r")

        plt.xlim(-0.05 * sx, 1.05 * sx)
        plt.ylim(1.05 * sy, -0.05 * sy)

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
            cv2.circle(
                canvas2, tuple(point.astype(np.int32)), int(np.ceil(radius)), 255, -1
            )

            filled_regions.append((canvas1 > 0) & (canvas2 > 0))
        else:
            filled_regions.append(canvas1 > 0)

    return filled_regions


def lloyds_algorithm(grid, vectors, iterations=10, plot=False):
    r"""
    Implements `Lloyd's Algorithm <https://en.wikipedia.org/wiki/Lloyd's_algorithm>`
    on a set of ``vectors`` using the helper function
    :meth:`~slmsuite.holography.toolbox.voronoi_windows()`.
    This iteratively forces a set of ``vectors` away from each other until
    they become more evenly distributed over a space.

    Parameters
    ----------
    grid : (array_like, array_like) OR :class:`~slmsuite.hardware.slms.slm.SLM` OR (int, int)
        See :meth:`~slmsuite.holography.toolbox.voronoi_windows()`.
    vectors : array_like
        See :meth:`~slmsuite.holography.toolbox.voronoi_windows()`.
    iterations : int
        Number of iterations to apply Lloyd's Algorithm.
    plot : bool
        Whether to plot each iteration of the algorithm.

    Returns
    -------
    numpy.ndarray
        The result of Lloyd's Algorithm.
    """
    result = np.copy(format_2vectors(vectors))
    (x_grid, y_grid) = _process_grid(grid)

    for _ in range(iterations):
        windows = voronoi_windows(grid, result, plot=plot)

        no_change = True

        # For each point, move towards the centroid of the window.
        for index, window in enumerate(windows):
            centroid_x = np.mean(x_grid[window])
            centroid_y = np.mean(y_grid[window])

            # Iterate
            if abs(centroid_x - result[0, index]) < 1 and abs(centroid_y - result[1, index]) < 1:
                pass
            else:
                no_change = False
                result[0, index] = np.mean(x_grid[window])
                result[1, index] = np.mean(y_grid[window])

        # If this iteration did nothing, then finish.
        if no_change:
            break

    return result


def lloyds_points(grid, n_points, iterations=10, plot=False):
    r"""
    Implements `Lloyd's Algorithm <https://en.wikipedia.org/wiki/Lloyd's_algorithm>`
    without seed ``vectors``. Instead, autogenerates the seed ``vectors`` randomly.
    See :meth:`~slmsuite.holography.toolbox.lloyds_algorithm()`.

    Parameters
    ----------
    grid : (array_like, array_like) OR :class:`~slmsuite.hardware.slms.slm.SLM` OR (int, int)
        See :meth:`~slmsuite.holography.toolbox.voronoi_windows()`.
    n_points : int
        Number of points to generate inside a space.
    iterations : int
        Number of iterations to apply Lloyd's Algorithm.
    plot : bool
        Whether to plot each iteration of the algorithm.

    Returns
    -------
    numpy.ndarray
        The result of Lloyd's Algorithm.
    """
    if (
        isinstance(grid, (list, tuple))
        and isinstance(grid[0], (int))
        and isinstance(grid[1], (int))
    ):
        shape = grid
    else:
        (x_grid, y_grid) = _process_grid(grid)
        shape = x_grid.shape

    vectors = np.vstack((
        np.random.randint(0, shape[1], n_points),
        np.random.randint(0, shape[0], n_points)
    ))

    # Regenerate until no overlaps (improve for performance?)
    while smallest_distance(vectors) < 1:
        vectors = np.vstack((
            np.random.randint(0, shape[1], n_points),
            np.random.randint(0, shape[0], n_points)
        ))

    grid2 = np.meshgrid(range(shape[1]), range(shape[0]))

    result = lloyds_algorithm(grid2, vectors, iterations, plot)

    if isinstance(grid, (list, tuple)):
        return result
    else:
        return np.vstack((x_grid[result], y_grid[result]))


# Basic functions
def _process_grid(grid):
    r"""
    Functions in :mod:`.toolbox` make use of normalized meshgrids containing the normalized
    coordinate of each corresponding pixel. This helper function interprets what the user passes.

    Parameters
    ----------
    grid : (array_like, array_like) OR :class:`~slmsuite.hardware.slms.slm.SLM`
        Meshgrids of normalized :math:`\frac{x}{\lambda}` coordinates
        corresponding to SLM pixels, in ``(x_grid, y_grid)`` form.
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


def blaze(grid, vector=(0, 0), offset=0):
    r"""
    Returns a simple blaze (phase ramp).

    .. math:: \phi(\vec{x}) = 2\pi \cdot \vec{k}_g \cdot \vec{x} + o

    Parameters
    ----------
    grid : (array_like, array_like) OR :class:`~slmsuite.hardware.slms.slm.SLM`
        Meshgrids of normalized :math:`\frac{x}{\lambda}` coordinates
        corresponding to SLM pixels, in ``(x_grid, y_grid)`` form.
        These are precalculated and stored in any :class:`~slmsuite.hardware.slms.slm.SLM`, so
        such a class can be passed instead of the grids directly.
    vector : (float, float)
        Blaze vector in normalized :math:`\frac{k_x}{k}` units.
        See :meth:`~slmsuite.holography.toolbox.convert_blaze_vector()`
    offset :
        Phase offset for this blaze.

    Returns
    -------
    numpy.ndarray
        The phase for this function.
    """
    (x_grid, y_grid) = _process_grid(grid)

    return 2 * np.pi * (vector[0] * x_grid + vector[1] * y_grid) + offset


def lens(grid, f=(np.inf, np.inf), center=(0, 0), angle=0):
    r"""
    Returns a simple thin lens (parabolic). When ``f`` is isotropic and ``angle`` :math:`\theta` is zero,

    .. math:: \phi(\vec{x}) = \frac{\pi}{f}(|\vec{x} - \vec{c}|^2)

    Otherwise,

    .. math:: \phi(x, y) = \pi \left[ G_{00}(x - c_x)^2 + 2G_{10}(x - c_x)(y - c_y) + G_{11}(y - c_y)^2 \right]

    Using the rotation of the lens power:

    .. math::   \begin{bmatrix}
                    G_{00} & G_{10} \\
                    G_{10} & G_{11} \\
                \end{bmatrix}
                =
                R(-\theta)
                \begin{bmatrix}
                    1/f_x & 0 \\
                    0 & 1/f_y \\
                \end{bmatrix}
                R(\theta).

    Note
    ~~~~
    In the future, we should add shear variance in the style of
    :meth:`~slmsuite.hardware.analysis.image_variance()`.
    Perhaps if the user passes in a 3-tuple for ``f``?

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
    center : (float, float)
        Center of the lens in normalized :math:`\frac{x}{\lambda}` coordinates.
        Defaults to no shift.
    angle : float
        Angle to rotate the basis of the lens by. Defaults to zero.

    Returns
    -------
    numpy.ndarray
        The phase for this function.
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
        assert not np.any(f == 0), "Cannot interpret a focal length of zero."

        # Optical power of lens
        g = [[1 / f[0], 0], [0, 1 / f[1]]]

        # Rotate if necessary
        if angle != 0:
            s = np.sin(angle)
            c = np.cos(angle)
            rot = np.array([[c, -s], [s, c]])

            g = np.matmul(np.linalg.inv(rot), np.matmul(g, rot))
    else:
        raise ValueError(
            "Expected f to be a scalar, a vector of length 2, or a 2x2 matrix."
        )

    # Only add a component if necessary (for speed)
    out = None

    if g[0][0] != 0:
        if out is None:
            out = np.square(x_grid - center[0]) * (g[0][0] * np.pi)
        else:
            out += np.square(x_grid - center[0]) * (g[0][0] * np.pi)

    if g[1][1] != 0:
        if out is None:
            out = np.square(y_grid - center[1]) * (g[1][1] * np.pi)
        else:
            out += np.square(y_grid - center[1]) * (g[1][1] * np.pi)

    shear = (g[1][0] + g[0][1]) * np.pi

    if shear != 0:
        if out is None:
            out = (x_grid - center[0]) * (y_grid - center[1]) * shear
        else:
            out += (x_grid - center[0]) * (y_grid - center[1]) * shear

    return out


def zernike(grid, n, m, aperture=None):
    r"""
    Returns a single Zernike polynomial.

    Parameters
    ----------
    grid : (array_like, array_like) OR :class:`~slmsuite.hardware.slms.slm.SLM`
        Meshgrids of normalized :math:`\frac{x}{\lambda}` coordinates
        corresponding to SLM pixels, in ``(x_grid, y_grid)`` form.
        These are precalculated and stored in any :class:`~slmsuite.hardware.slms.slm.SLM`, so
        such a class can be passed instead of the grids directly.
    n, m : int
        Cartesian Zernike index defining the polynomial.
    aperture : {"circular", "elliptical", "cropped"} OR (float, float) OR None
        See :meth:`.zernike_sum()`.

    Returns
    -------
    numpy.ndarray
        The phase for this function.
    """
    return zernike_sum(grid, (((n, m), 1), ), aperture=aperture)


def zernike_sum(grid, weights, aperture=None):
    r"""
    Returns a summation of Zernike polynomials.

    Important
    ~~~~~~~~~
    Zernike polynomials are canonically defined on a circular aperture. However, we may
    want to use these polynomials on other apertures (e.g. a rectangular SLM).
    Cropping this aperture breaks the orthogonality and normalization of the set, but
    this is fine for many applications. While it is possible to orthonormalize the
    cropped set, we do not do so in :mod:`slmsuite`, as this is not critical for target
    applications such as abberation correction.

    Parameters
    ----------
    grid : (array_like, array_like) OR :class:`~slmsuite.hardware.slms.slm.SLM`
        Meshgrids of normalized :math:`\frac{x}{\lambda}` coordinates
        corresponding to SLM pixels, in ``(x_grid, y_grid)`` form.
        These are precalculated and stored in any :class:`~slmsuite.hardware.slms.slm.SLM`, so
        such a class can be passed instead of the grids directly.
    weights : list of ((int, int), float)
        Which Zernike polynomials to sum. The ``(int, int)`` is the cartesian index
        ``(n, m)``. The float is the weight for the given index.
    aperture : {"circular", "elliptical", "cropped"} OR (float, float) OR None
        How to scale the polynomials relative to the grid shape. This is relative
        to the :math:`R = 1` edge of a standard Zernike pupil.

        ``"circular"``, ``None``
          The circle is scaled isotropically until the pupil edge touches the grid edge.
        ``"elliptical"``
          The circle is scaled anisotropically until each cartesian pupil edge touches a grid
          edge. Generally produces and ellipse.
        ``"cropped"``
          The circle is scaled isotropically until the rectangle of the grid is
          circumscribed by the circle.
        ``(float, float)``
          Custom scaling. These values are multiplied to the ``x_grid`` and ``y_grid``
          directly, respectively. The edge of the pupil corresponds to where
          ``x_grid**2 + y_grid**2 = 1``.

    Returns
    -------
    numpy.ndarray
        The phase for this function.
    """
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
        raise ValueError("Type {} not recognized.".format(type(aperture)))

    mask = np.square(x_grid * x_scale) + np.square(y_grid * y_scale) <= 1
    use_mask = np.any(mask == 0)

    if use_mask:
        x_grid_scaled = x_grid[mask] * x_scale
        y_grid_scaled = y_grid[mask] * y_scale
    else:
        x_grid_scaled = x_grid * x_scale
        y_grid_scaled = y_grid * y_scale

    summed_coefficients = {}

    for (key, weight) in weights:
        coefficients = _zernike_coefficients(key[0], key[1])

        for power_key, factor in coefficients.items():
            power_factor = factor * weight
            if power_key in summed_coefficients:
                summed_coefficients[power_key] += power_factor
            else:
                summed_coefficients[power_key] = power_factor

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

    return canvas

_zernike_cache = {}

def _zernike_coefficients(n, m):
    """
    Returns the coefficients for the :math:`x^ay^b` terms of the cartesian Zernike polynomial
    of index `(`n, m)``. This is returned as a dictionary of form ``{(a,b) : coefficient}``.
    Uses the algorithm given in [0]_.

    .. [0] Efficient Cartesian representation of Zernike polynomials in computer memory.
    """
    n = int(n)
    m = int(m)

    assert 0 <= m <= n, "Invalid cartesian Zernike index."

    key = (n, m)

    if not key in _zernike_cache:
        zernike_this = {}

        l = n - 2 * m

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

        def comb(n, k):
            return factorial(n) / (factorial(k) * factorial(n-k))

        for i in range(q+1):
            for j in range(m+1):
                for k in range(m-j+1):
                    factor = -1 if (i + j) % 2 else 1
                    factor *= comb(l, 2 * i + p)
                    factor *= comb(m - j, k)
                    factor *= (float(factorial(n - j))
                        / (factorial(j) * factorial(m - j) * factorial(n - m - j)))

                    power_key = (n - 2*(i + j + k) - p, 2 * (i + k) + p)

                    if power_key in zernike_this:
                        zernike_this[power_key] += factor
                    else:
                        zernike_this[power_key] = factor

        _zernike_cache[key] = {power_key: factor for power_key, factor in zernike_this.items() if factor != 0}

    return _zernike_cache[key]

# Structured light
def _determine_source_radius(grid, w=None):
    r"""
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
        Determined radius.
    """
    (x_grid, y_grid) = _process_grid(grid)

    if w is None:
        return np.min([np.amax(x_grid), np.amax(y_grid)]) / 4
    else:
        return w


def laguerre_gaussian(grid, l, p, w=None):
    r"""
    Returns the phase farfield for a Laguerre-Gaussian beam.
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
    Returns the phase farfield for a Hermite-Gaussian beam.

    Ref: https://doi.org/10.1364/AO.54.008444

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
    **(Untested)** Returns the phase farfield for an Ince-Gaussian beam.

    Ref: https://doi.org/10.1364/OL.29.000144

    Ref: https://doi.org/10.1364/AO.54.008444

    Ref: https://doi.org/10.3390/jimaging8050144

    Ref: https://en.wikipedia.org/wiki/Elliptic_coordinate_system

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
    **(NotImplemented)** Returns the phase farfield for a Matheui-Gaussian beam.

    Ref: https://doi.org/10.1364/AO.49.006903

    Returns
    -------
    numpy.ndarray
        The phase for this function.
    """
    (x_grid, y_grid) = _process_grid(grid)

    w = _determine_source_radius(grid, w)
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
        The desired shape of the ``matrix``.

    Returns
    -------
    numpy.ndarray
        Padded ``matrix``.
    """
    deltashape = (
        (shape[0] - matrix.shape[0]) / 2.0,
        (shape[1] - matrix.shape[1]) / 2.0,
    )

    assert (
        deltashape[0] >= 0 and deltashape[1] >= 0
    ), "Shape {} is too large to pad to shape {}".format(tuple(matrix.shape), shape)

    padB = int(np.floor(deltashape[0]))
    padT = int(np.ceil(deltashape[0]))
    padL = int(np.floor(deltashape[1]))
    padR = int(np.ceil(deltashape[1]))

    toReturn = np.pad(
        matrix, [(padB, padT), (padL, padR)], mode="constant", constant_values=0
    )

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
        The desired shape of the ``matrix``.

    Returns
    ----------
    numpy.ndarray OR (int, int, int, int)
        Either the unpadded ``matrix`` or the integers used to unpad such a matrix,
        depending what is passed as ``matrix``.
    """
    mshape = np.shape(matrix)
    return_args = False
    if len(mshape) == 1 or np.prod(mshape) == 2:
        # Assume as tuple was provided.
        mshape = np.squeeze(matrix)
        return_args = True

    deltashape = ((shape[0] - mshape[0]) / 2.0, (shape[1] - mshape[1]) / 2.0)

    assert (
        deltashape[0] <= 0 and deltashape[1] <= 0
    ), "Shape {} is too small to unpad to shape {}".format(tuple(mshape), shape)

    padB = int(np.floor(-deltashape[0]))
    padT = int(mshape[0] - np.ceil(-deltashape[0]))
    padL = int(np.floor(-deltashape[1]))
    padR = int(mshape[1] - np.ceil(-deltashape[1]))

    if return_args:
        return (padB, padT, padL, padR)

    toReturn = matrix[padB:padT, padL:padR]

    assert np.all(toReturn.shape == shape)

    return toReturn
