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
import warnings

# Phase pattern collation and manipulation
def imprint(
    matrix,
    window,
    function,
    grid,
    imprint_operation="replace",
    centered=False,
    clip=True,
    transform=0,
    shift=(0,0),
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
    transform : float or ((float, float), (float, float))
       Passed to :meth:`shift_grid`, operating on the cropped imprint grid.
       This is left as an option such that the user does not have to transform the
       entire ``grid`` to satisfy a tiny imprinted patch.
       See :meth:`shift_grid`.
    shift : (float, float)
       Passed to :meth:`shift_grid`, operating on the cropped imprint grid.
       This is left as an option such that the user does not have to transform the
       entire ``grid`` to satisfy a tiny imprinted patch.
       See :meth:`shift_grid`.
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
                shift_grid((x_grid[yi:yf, xi:xf], y_grid[yi:yf, xi:xf]), transform, shift),
                **kwargs
            )
        elif imprint_operation == "add":
            matrix[yi:yf, xi:xf] += function(
                shift_grid((x_grid[yi:yf, xi:xf], y_grid[yi:yf, xi:xf]), transform, shift),
                **kwargs
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
                shift_grid((x_grid[y_ind, x_ind], y_grid[y_ind, x_ind]), transform, shift),
                **kwargs
            )
        elif imprint_operation == "add":
            matrix[y_ind, x_ind] += function(
                shift_grid((x_grid[y_ind, x_ind], y_grid[y_ind, x_ind]), transform, shift),
                **kwargs
            )
        else:
            raise ValueError()
    elif np.shape(window) == np.shape(
        matrix
    ):  # Boolean numpy array. Future: extra checks?

        # Modify the matrix
        if imprint_operation == "replace":
            matrix[window] = function(
                shift_grid((x_grid[window], y_grid[window]), transform, shift),
                **kwargs
            )
        elif imprint_operation == "add":
            matrix[window] += function(
                shift_grid((x_grid[window], y_grid[window]), transform, shift),
                **kwargs
            )
        else:
            raise ValueError()
    else:
        raise ValueError()

    return matrix


# Unit helper functions
BLAZE_LABELS = {
    "norm" : (r"$k_x/k$", r"$k_y/k$"),
    "kxy" : (r"$k_x/k$", r"$k_y/k$"),
    "rad" : (r"$\theta_x$ [rad]", r"$\theta_y$ [rad]"),
    "knm" : (r"$n$ [pix]", r"$m$ [pix]"),
    "freq" : (r"$f_x$ [1/pix]", r"$f_y$ [1/pix]"),
    "lpmm" : (r"$k_x/2\pi$ [1/mm]", r"$k_y/2\pi$ [1/mm]"),
    "mrad" : (r"$\theta_x$ [mrad]", r"$\theta_y$ [mrad]"),
    "deg" : (r"$\theta_x$ [$^\circ$]", r"$\theta_y$ [$^\circ$]")
}
BLAZE_UNITS = BLAZE_LABELS.keys()

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
            Computational blaze units for a given Fourier domain ``shape``.
            This corresponds to integer points on the grid of this
            (potentially padded) SLM's Fourier transform.
            See :class:`~slmsuite.holography.Hologram`.

            Important
            ~~~~~~~~~
            The ``"knm"`` basis is centered at ``shape/2``, unlike all of the other units.

          ``"freq"``
            Pixel frequency of a grating producing the blaze.
            e.g. 1/16 is a grating with a period of 16 pixels.
          ``"lpmm"``
            Line pairs per mm or lines per mm of a grating producing the blaze.
          ``"rad"``, ``"mrad"``, ``"deg"``
            Angle at which light is blazed in various units. Small angle approximation is assumed.

        Warning
        ~~~~~~~
        The units ``"freq"``, ``"knm"``, and ``"lpmm"`` depend on SLM pixel size,
        so a ``slm`` should be passed (otherwise returns an array of ``nan`` values).
        ``"knm"`` additionally requires the ``shape`` of the computational space.
        If not included when an slm is passed, ``shape=slm.shape`` is assumed.

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
            Shape of the computational SLM space in :mod:`numpy` ``(h, w)`` form.
            Defaults to ``slm.shape`` if ``slm`` is not ``None``.

        Returns
        --------
        numpy.ndarray
            Result of the unit conversion, in the cleaned format of :meth:`format_2vectors()`.
        """
    assert from_units in BLAZE_UNITS and to_units in BLAZE_UNITS

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
            if slm is None:
                shape = np.nan
            else:
                shape = slm.shape
        shape = format_2vectors(np.flip(np.squeeze(shape)))

        knm_conv = pitch * shape

    if from_units == "norm" or from_units == "kxy" or from_units == "rad":
        rad = vector
    elif from_units == "knm":
        rad = (vector - shape / 2.0) / knm_conv
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
        return rad * knm_conv + shape / 2.0
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
    for unit in BLAZE_UNITS:
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

        plt.gca().set_aspect('equal')

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


import slmsuite.toolbox.phase as toolbox_phase


def shift_grid(grid, transform=None, shift=None):
    """
    Shifts and transforms an SLM grid.

    Parameters
    ----------
    grid : (array_like, array_like) OR :class:`~slmsuite.hardware.slms.slm.SLM`
        Meshgrids of normalized :math:`\frac{x}{\lambda}` coordinates
        corresponding to SLM pixels, in ``(x_grid, y_grid)`` form.
        These are precalculated and stored in any :class:`~slmsuite.hardware.slms.slm.SLM`, so
        such a class can be passed instead of the grids directly.
    transform : float OR ((float, float), (float, float)) OR None
        If a scalar is passed, angle to rotate the basis of the lens by.
        Defaults to zero if `None`.
        If a 2x2 matrix is passed, transforms the :math:`x` and :math:`y` grids
        according to :math:`x' = M_{00}x + M_{01}y`,  :math:`y' = M_{10}y + M_{11}y`.
    shift : (float, float) OR None
        Center of the grid in normalized :math:`\frac{x}{\lambda}` coordinates.
        Defaults to no shift if `None`.

    Returns
    -------
    grid : (array_like, array_like)
        The shifted grid
    """
    (x_grid, y_grid) = _process_grid(grid)

    if transform is None:
        transform = 0

    if shift is None:
        shift = (0, 0)

    if transform == 0:
        return (
            x_grid if shift[0] == 0 else (x_grid - shift[0]),
            y_grid if shift[1] == 0 else (y_grid - shift[1])
        )
    else:
        if np.isscalar(transform):
            s = np.sin(transform)
            c = np.cos(transform)
            transform = np.array([[c, -s], [s, c]])
        else:
            transform = np.squeeze(transform)

        assert np.shape(transform) == 2

        return (
            transform[0,0] * x_grid - transform[0,0] * y_grid if shift[0] == 0 else (c * x_grid - s * y_grid - shift[0]),
            transform[0,0] * x_grid + transform[1,1] * y_grid if shift[1] == 0 else (c * y_grid + s * x_grid - shift[1])
        )


def blaze(grid, vector=(0, 0), offset=0):
    r"""
    **(Deprecated; use :meth:`slmsuite.toolbox.phase.blaze` instead)**
    """
    return toolbox_phase.blaze(grid, vector, offset)


def lens(grid, f=(np.inf, np.inf), center=None, angle=None):
    r"""
    **(Deprecated; use :meth:`slmsuite.toolbox.phase.lens` instead)**
    """
    if center is not None or angle is not None:
        grid = shift_grid(grid, transform=angle, shift=center)

    return toolbox_phase.blaze(grid, f)


def axicon(grid, f=(np.inf, np.inf), w=None):
    r"""
    **(Deprecated; use :meth:`slmsuite.toolbox.phase.axicon` instead)**
    """
    return toolbox_phase.axicon(grid, f, w)


def zernike(grid, n, m, aperture=None):
    r"""
    **(Deprecated; use :meth:`slmsuite.toolbox.phase.zernike` instead)**
    """
    return toolbox_phase.zernike(grid, n, m, aperture)


def zernike_sum(grid, weights, aperture=None):
    r"""
    **(Deprecated; use :meth:`slmsuite.toolbox.phase.zernike_sum` instead)**
    """
    return toolbox_phase.zernike_sum(grid, weights, aperture)


# Structured light
def laguerre_gaussian(grid, l, p, w=None):
    r"""
    **(Deprecated; use :meth:`slmsuite.toolbox.phase.laguerre_gaussian` instead)**
    """
    return toolbox_phase.laguerre_gaussian(grid, l, p, w)


def hermite_gaussian(grid, n, m, w=None):
    r"""
    **(Deprecated; use :meth:`slmsuite.toolbox.phase.hermite_gaussian` instead)**
    """
    return toolbox_phase.hermite_gaussian(grid, n, m, w)


def ince_gaussian(grid, p, m, parity=1, ellipticity=1, w=None):
    r"""
    **(NotImplemented; Deprecated; use :meth:`slmsuite.toolbox.phase.ince_gaussian` instead)**
    """
    return ince_gaussian(grid, p, m, parity, ellipticity, w)


def matheui_gaussian(grid, r, q, w=None):
    """
    **(NotImplemented; Deprecated; use :meth:`slmsuite.toolbox.phase.matheui_gaussian` instead)**
    """
    return matheui_gaussian(grid, r, q, w)


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
        The desired shape of the ``matrix`` in :mod:`numpy` ``(h, w)`` form.

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
        Data to unpad. If this is a shape in :mod:`numpy` ``(h, w)`` form,
        returns the four slicing integers used to unpad that shape ``[padB:padT, padL:padR]``.
    shape : (int, int)
        The desired shape of the ``matrix`` in :mod:`numpy` ``(h, w)`` form.

    Returns
    ----------
    numpy.ndarray OR (int, int, int, int)
        Either the unpadded ``matrix`` or the four slicing integers used to unpad such a matrix,
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
