r"""
Helper functions for manipulating phase patterns.
"""

import numpy as np
from scipy.spatial.distance import chebyshev
from scipy.spatial import Voronoi, voronoi_plot_2d
import cv2
import matplotlib.pyplot as plt

from slmsuite.misc.math import (
    INTEGER_TYPES, REAL_TYPES
)

# Unit definitions.

BLAZE_LABELS = {
    "norm" : (r"$k_x/k$",               r"$k_y/k$"),
    "kxy" :  (r"$k_x/k$",               r"$k_y/k$"),
    "rad" :  (r"$\theta_x$ [rad]",      r"$\theta_y$ [rad]"),
    "knm" :  (r"$n$ [pix]",             r"$m$ [pix]"),
    "ij" :   (r"Camera $x$ [pix]",      r"Camera $y$ [pix]"),
    "freq" : (r"$f_x$ [1/pix]",         r"$f_y$ [1/pix]"),
    "lpmm" : (r"$k_x/2\pi$ [1/mm]",     r"$k_y/2\pi$ [1/mm]"),
    "mrad" : (r"$\theta_x$ [mrad]",     r"$\theta_y$ [mrad]"),
    "deg" :  (r"$\theta_x$ [$^\circ$]", r"$\theta_y$ [$^\circ$]")
}
BLAZE_UNITS = BLAZE_LABELS.keys()

# Unit helper functions.

def convert_blaze_vector(
    vector, from_units="norm", to_units="norm", slm=None, shape=None
):
    r"""
    Helper function for vector unit conversions.

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

        The ``"knm"`` basis is centered at ``shape/2``, unlike all of the other units.

        ``"ij"``
        Camera pixel units.
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
    The unit ``"ij"``, camera pixels, requires calibration data stored in a CameraSLM, so
    this must be passed in place of ``slm``.
    The unit ``"knm"`` additionally requires the ``shape`` of the computational space.
    If not included when an slm is passed, ``shape=slm.shape`` is assumed.

    Parameters
    ----------
    vector : array_like
        2-vectors for which we want to convert units, from ``from_units`` to ``to_units``.
        Processed according to :meth:`format_2vectors()`.
    from_units, to_units : str
        Units which we are converting between. See the listed units above for options.
        Defaults to ``"norm"``.
    slm : :class:`~slmsuite.hardware.slms.slm.SLM` OR :class:`~slmsuite.hardware.cameraslms.CameraSLM` OR None
        Relevant SLM to pull data from in the case of
        ``"freq"``, ``"knm"``, or ``"lpmm"``.
        If :class:`~slmsuite.hardware.cameraslms.CameraSLM`, the unit ``"ij"`` can be
        processed too.
    shape : (int, int) OR None
        Shape of the computational SLM space. Defaults to ``slm.shape`` if ``slm``
        is not ``None``.

    Returns
    --------
    numpy.ndarray
        Result of the unit conversion, in the cleaned format of :meth:`format_2vectors()`.
    """
    assert from_units in BLAZE_UNITS, \
        "toolbox.py: Unit '{}' not recognized as a valid unit for convert_blaze_vector().".format(from_units)
    assert to_units in BLAZE_UNITS, \
        "toolbox.py: Unit '{}' not recognized as a valid unit for convert_blaze_vector().".format(to_units)

    vector = format_2vectors(vector).astype(float)

    # Determine whether a CameraSLM was passed (to enable "ij" units)
    if hasattr(slm, "slm"):
        cameraslm = slm
        slm = cameraslm.slm
    else:
        cameraslm = None

    if from_units == "ij" or to_units == "ij":
        if cameraslm is None or cameraslm.fourier_calibration is None:
            return vector * np.nan

    # Generate conversion factors for various units
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

    # Convert the input to normalized "kxy" units.
    if from_units == "norm" or from_units == "kxy" or from_units == "rad":
        rad = vector
    elif from_units == "knm":
        rad = (vector - shape / 2.0) / knm_conv
    elif from_units == "ij":
        rad = cameraslm.ijcam_to_kxyslm(vector)
    elif from_units == "freq":
        rad = vector * wav_um / pitch_um
    elif from_units == "lpmm":
        rad = vector * wav_um / 1000
    elif from_units == "mrad":
        rad = vector / 1000
    elif from_units == "deg":
        rad = vector * np.pi / 180

    # Convert from normalized "kxy" units to the desired output units.
    if to_units == "norm" or to_units == "kxy" or to_units == "rad":
        return rad
    elif to_units == "knm":
        return rad * knm_conv + shape / 2.0
    elif to_units == "ij":
        return cameraslm.kxyslm_to_ijcam(vector)
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


def convert_blaze_radius(radius, from_units="norm", to_units="norm", slm=None, shape=None):
    """
    Helper function for scalar unit conversions.
    Uses :meth:`convert_blaze_vector` to deduce the (average, in the case of an
    anisotropic transformation) scalar radius when going between sets of units.

    Parameters
    ----------
    radius : float
        The scalar radius to convert.
    from_units, to_units : str
        Passed to :meth:`convert_blaze_vector`.
    slm : :class:`~slmsuite.hardware.slms.slm.SLM` OR :class:`~slmsuite.hardware.cameraslms.CameraSLM` OR None
        Passed to :meth:`convert_blaze_vector`.
    shape : (int, int) OR None
        Passed to :meth:`convert_blaze_vector`.
    """
    v0 = convert_blaze_vector(
        (0, 0), from_units=from_units, to_units=to_units, slm=slm, shape=shape
    )
    vx = convert_blaze_vector(
        (radius, 0), from_units=from_units, to_units=to_units, slm=slm, shape=shape
    )
    vy = convert_blaze_vector(
        (0, radius), from_units=from_units, to_units=to_units, slm=slm, shape=shape
    )
    return np.mean([np.linalg.norm(vx - v0), np.linalg.norm(vy - v0)])


# Windows creation functions. Windows are views into 2D arrays.

def window_slice(window, shape=None, centered=False, circular=False):
    """
    Get the slices that describe the window's view into the larger array.

    Parameters
    ----------
    window : (int, int, int, int) OR (array_like, array_like) OR array_like
        A number of formats are accepted:
        - List in ``(x, w, y, h)`` format, where ``w`` and ``h`` are the width and height of
          the region and  ``(x,y)`` is the upper-left coordinate.
          If ``centered``, then ``(x,y)`` is instead the center of the region to imprint.
          If ``circular``, then an elliptical region circumscribed by the rectangular region is returned.
        - Tuple containing arrays of identical length corresponding to y and x indices.
          ``centered`` and ``circular`` are ignored.
        - Boolean array of same ``shape`` as ``matrix``; the window is defined where ``True`` pixels are.
          ``centered`` and ``circular`` are ignored.
    shape : (int, int) OR None
        The (height, width) of the array that the window is a view into.
        If not ``None``, indices beyond those allowed by ``shape`` will be clipped.
    centered : bool
        See ``window``.
    circular : bool
        See ``window``.

    Returns
    -------
    slice_ : (slice, slice) OR (array_like, array_like) OR (array_like)
        The slice for the window.
    """
    # (v.x, w, v.y, h) format
    if len(window) == 4:
        # Prepare helper vars
        xi = int(window[0] - ((window[1] - 2) / 2 if centered else 0))
        xf = xi + int(window[1])
        yi = int(window[2] - ((window[3] - 2) / 2 if centered else 0))
        yf = yi + int(window[3])

        if shape is not None:
            [xi, xf] = np.clip([xi, xf], 0, shape[1] - 1)
            [yi, yf] = np.clip([yi, yf], 0, shape[0] - 1)

        if circular:    # If a circular window is desired, compute this.
            x_list = np.arange(xi, xf)
            y_list = np.arange(yi, yf)
            x_grid, y_grid = np.meshgrid(x_list, y_list)

            xc = xi + int((window[1] - 1) / 2)
            yc = yi + int((window[3] - 1) / 2)

            rr_grid = (
                (window[3] ** 2) * np.square(x_grid.astype(float) - xc) +
                (window[1] ** 2) * np.square(y_grid.astype(float) - yc)
            )

            mask_grid = rr_grid <= (window[1] ** 2) * (window[3] ** 2) / 4.

            return window_slice((y_grid[mask_grid], x_grid[mask_grid]), shape=shape)
        else:           # Otherwise, return square slices
            slice_ = (slice(yi, yf), slice(xi, xf))
    # (y_ind, x_ind) format
    elif len(window) == 2:
        # Prepare the lists
        y_ind = np.ravel(window[0])
        x_ind = np.ravel(window[1])
        if shape is not None:
            x_ind = np.clip(x_ind, 0, shape[1] - 1)
            y_ind = np.clip(y_ind, 0, shape[0] - 1)
        slice_ = (y_ind, x_ind)
    # Boolean numpy array.
    elif np.ndim(window) == 2:
        slice_ = window
    else:
        raise ValueError("Unrecognized format for `window`.")

    return slice_


def window_square(window, padding_frac=0, padding_pix=0):
    """
    Find a square that covers the active region of ``window``.

    Parameters
    ----------
    window : numpy.ndarray<bool> (height, width)
        Boolean mask.
    padding : float
        Fraction of the window width and height to pad these by on all sides.
        For instance,
        This result is clipped to be within ``shape`` of the window.

    Returns
    -------
    window_square : (int, int, int, int)
        A square that covers the active region of ``window``
        in the format (x, width2, y, height2) where
        (x, y) is the upper left coordinate, and (width2, height2) define
        the extent.
    """
    limits = []

    # For each axis...
    for a in [0, 1]:
        if len(window) == 2:        # Handle two list case
            limit = np.array([np.amin(window[a]), np.amax(window[a])+1])
        elif np.ndim(window) == 2:  # Handle the boolean array case
            collapsed = np.where(np.any(window, axis=a))  # Collapse the other axis
            limit = np.array([np.amin(collapsed), np.amax(collapsed)+1])
        else:
            raise ValueError("Unrecognized format for `window`.")

        # Add padding if desired.
        padding_ = int(np.floor(np.diff(limit) * padding_frac) + padding_pix)
        limit += np.array([-padding_, padding_])

        # Clip the padding to shape.
        if np.ndim(window) == 2:
            limit = np.clip(limit, 0, window.shape[1-a])

        limits.append(tuple(limit))

    # Return desired format.
    return (
        limits[0][0], limits[0][1] - limits[0][0],
        limits[1][0], limits[1][1] - limits[1][0]
    )


def voronoi_windows(grid, vectors, radius=None, plot=False):
    r"""
    Returns boolean array windows corresponding to the Voronoi cells for a set of vectors.
    These boolean array windows are in the style of :meth:`~slmsuite.holography.toolbox.imprint()`.
    The ith window corresponds to the Voronoi cell centered around the ith vector.

    Note
    ~~~~
    The :meth:`cv2.fillConvexPoly()` function used to fill each window dilates
    slightly outside the window bounds. To avoid pixels belonging to multiple windows
    simultaneously, we crop away previously-assigned pixels from new windows while these are
    being iteratively generated. As a result, windows earlier in the list will be slightly
    larger than windows later in the list.

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

    # Add additional points in a diamond outside the shape of interest to cause all
    # windows of interest to be finite.
    vectors_voronoi = np.concatenate((
        vectors.T,
        np.array(
            [[hsx, -3 * hsy], [hsx, 5 * hsy], [-3 * hsx, hsy], [5 * hsx, hsy]]
        ),
    ))

    vor = Voronoi(vectors_voronoi, furthest_site=False)

    if plot:
        sx = shape[1]
        sy = shape[0]

        # Use the built-in scipy function to plot a visualization of the windows.
        fig = voronoi_plot_2d(vor)

        # Plot a bounding box corresponding to the grid.
        plt.plot(np.array([0, sx, sx, 0, 0]), np.array([0, 0, sy, sy, 0]), "r")

        # Format and show the plot.
        plt.xlim(-0.05 * sx, 1.05 * sx)
        plt.ylim(1.05 * sy, -0.05 * sy)
        plt.gca().set_aspect('equal')
        plt.title("Voronoi Cells")
        plt.show()

    # Gather data from scipy Voronoi and return as a list of boolean windows.
    N = np.shape(vectors)[1]
    filled_regions = []
    already_filled = np.zeros(shape, dtype=np.uint8)

    for x in range(N):
        point = tuple(np.around(vor.points[x]).astype(np.int32))
        region = vor.regions[vor.point_region[x]]
        pts = np.around(vor.vertices[region]).astype(np.int32)

        canvas1 = np.zeros(shape, dtype=np.uint8)
        cv2.fillConvexPoly(canvas1, pts, 255, cv2.LINE_4)

        # Crop the window to with a given radius, if desired.
        if radius is not None and radius > 0:
            canvas2 = np.zeros(shape, dtype=np.uint8)
            cv2.circle(
                canvas2, point, int(np.ceil(radius)), 255, -1
            )

            filled_regions.append((canvas1 > 0) & (canvas2 > 0) & np.logical_not(already_filled))
        else:
            filled_regions.append((canvas1 > 0) & np.logical_not(already_filled))

        already_filled |= filled_regions[-1]

    return filled_regions


# Phase pattern collation and manipulation. Uses windows.

def imprint(
    matrix,
    window,
    function,
    grid=None,
    imprint_operation="replace",
    centered=False,
    circular=False,
    clip=True,
    transform=0,
    shift=(0,0),
    **kwargs
):
    r"""
    Imprints a region (defined by ``window``) of a ``matrix`` with a ``function``.
    This ``function`` must be in the style of :mod:`~slmsuite.holography.toolbox.phase`
    phase helper functions, which expect a ``grid`` parameter to define the coordinate basis
    (see :meth:`~slmsuite.holography.toolbox.phase.blaze()` or
    :meth:`~slmsuite.holography.toolbox.phase.lens()`).

    For instance, we can imprint a blaze on a 200 by 200 pixel region
    of the SLM with:

    .. highlight:: python
    .. code-block:: python

        canvas = np.zeros(shape=slm.shape)  # Matrix to imprint onto.
        window = [200, 200, 200, 200]       # Region of the matrix to imprint.
        toolbox.imprint(canvas, window=window, function=toolbox.phase.blaze, grid=slm, vector=(.001, .001))

    See also :ref:`examples`.

    Parameters
    ----------
    matrix : numpy.ndarray
        The data to imprint a ``function`` onto.
    window
        See :meth:`~slmsuite.holography.toolbox.window_slice()`.
    function : function OR float
        A function in the style of :mod:`~slmsuite.holography.toolbox` helper functions,
        which accept ``grid`` as the first argument.
        Also accepts floating point values, in which case this value is simply added.
    grid : (array_like, array_like) OR :class:`~slmsuite.hardware.slms.slm.SLM` OR None
        Meshgrids of normalized :math:`\frac{x}{\lambda}` coordinates
        corresponding to SLM pixels, in ``(x_grid, y_grid)`` form.
        These are precalculated and stored in any :class:`~slmsuite.hardware.slms.slm.SLM`, so
        such a class can be passed instead of the grids directly.
        ``None`` can only be passed if a float is passed as ``function``.
    imprint_operation : {"replace" OR "add"}
        Decides how the ``function`` is imparted to the ``matrix``.

        - If ``"replace"``, then the values of ``matrix`` inside ``window`` are replaced with ``function``.
        - If ``"add"``, then these are instead added together (useful, for instance, for global blazes).
    centered
        See :meth:`~slmsuite.holography.toolbox.window_slice()`.
    circular
        See :meth:`~slmsuite.holography.toolbox.window_slice()`.
    clip : bool
        Whether to clip the imprint region if it exceeds the size of ``matrix``.
        If ``False``, then an error is raised when the size is exceeded.
        If ``True``, then the out-of-range pixels are instead filled with ``numpy.nan``.
    transform : float or ((float, float), (float, float))
       Passed to :meth:`shift_grid`, operating on the cropped imprint grid.
       This is left as an option such that the user does not have to transform the
       entire ``grid`` to satisfy a tiny imprinted patch.
       See :meth:`shift_grid` for more details.
    shift : (float, float)
       Passed to :meth:`shift_grid`, operating on the cropped imprint grid.
       This is left as an option such that the user does not have to transform the
       entire ``grid`` to satisfy a tiny imprinted patch.
       See :meth:`shift_grid` for more details.
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
    # Format the grid.
    if grid is not None:
        (x_grid, y_grid) = _process_grid(grid)

    # Get slices for the window in the matrix.
    shape = matrix.shape if clip else None
    slice_ = window_slice(window, shape=shape, centered=centered, circular=circular)

    # Decide whether to treat function as a float.
    is_float = isinstance(function, REAL_TYPES)

    if not is_float:
        assert grid is not None, "toolbox.py: imprint grid cannot be None if a function is given."

    # Modify the matrix.
    if imprint_operation == "replace":
        if is_float:
            matrix[slice_] = function
        else:
            matrix[slice_] = function(
                shift_grid((x_grid[slice_], y_grid[slice_]), transform, shift),
                **kwargs
            )
    elif imprint_operation == "add":
        if is_float:
            matrix[slice_] += function
        else:
            matrix[slice_] += function(
                shift_grid((x_grid[slice_], y_grid[slice_]), transform, shift),
                **kwargs
            )
    else:
        raise ValueError("Unrecognized imprint operation {}.".format(imprint_operation))

    return matrix


# Vector helper functions.

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


def fit_3pt(y0, y1, y2, N=None, x0=(0, 0), x1=(1, 0), x2=(0, 1), orientation_check=False):
    r"""
    Fits three points to an affine transformation. This transformation is given by:

    .. math:: \vec{y} = M \cdot \vec{x} + \vec{b}

    At base, this function finds and optionally uses affine transformations:

    .. highlight:: python
    .. code-block:: python

        y0 = (1.,1.)    # Origin
        y1 = (2.,2.)    # First point in x direction
        y2 = (1.,2.)    # first point in y direction

        # If N is None, return a dict with keys "M", and "b"
        affine_dict =   fit_3pt(y0, y1, y2, N=None)

        # If N is provided, evaluates the transformation on indices with the given shape
        # In this case, the requested 5x5 indices results in an array with shape (2,25)
        vector_array =  fit_3pt(y0, y1, y2, N=(5,5))

    However, ``fit_3pt`` is more powerful that this, and can fit an affine
    transformation to semi-arbitrary sets of points with known indices
    in the coordinate  system of the dependent variable :math:`\vec{x}`,
    as long as the passed indices ``x0``, ``x1``, ``x2`` are not colinear.

    .. highlight:: python
    .. code-block:: python

        # y11 is at x index (1,1), etc
        fit_3pt(y11, y34, y78, N=(5,5), x0=(1,1), x1=(3,4), x2=(7,8))

        # These indices don't have to be integers
        fit_3pt(a, b, c, N=(5,5), x0=(np.pi,1.5), x1=(20.5,np.sqrt(2)), x2=(7.7,42.0))

    Optionally, basis vectors can be passed directly instead of adding these
    vectors to the origin, by making use of passing ``None`` for ``x1`` or ``x2``:

    .. highlight:: python
    .. code-block:: python

        origin =    (1.,1.)     # Origin
        dv1 =       (1.,1.)     # Basis vector in x direction
        dv2 =       (1.,0.)     # Basis vector in y direction

        # The following are equivalent:
        option1 = fit_3pt(origin, np.add(origin, dv1), np.add(origin, dv2), N=(5,5))
        option2 = fit_3pt(origin, dv1, dv2, N=(5,5), x1=None, x2=None)

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
    orientation_check : bool
        If ``True``, removes the last two points in the affine grid.
        If ``False``, does nothing.

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
    elif isinstance(N, INTEGER_TYPES):
        if N <= 0:
            affine_return = True
        else:
            N = (N, N)
    elif (
        not np.isscalar(N) and len(N) == 2 and
        isinstance(N[0], INTEGER_TYPES) and isinstance(N[1], INTEGER_TYPES)
    ):
        if N[0] <= 0 or N[1] <= 0:
            affine_return = True
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
        if orientation_check:
            indices = indices[:, 0:-2]

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


def lloyds_algorithm(grid, vectors, iterations=10, plot=False):
    r"""
    Implements `Lloyd's Algorithm <https://en.wikipedia.org/wiki/Lloyd's_algorithm>`_
    on a set of ``vectors`` using the helper function
    :meth:`~slmsuite.holography.toolbox.voronoi_windows()`.
    This iteratively forces a set of ``vectors`` away from each other until
    they become more evenly distributed over a space.
    This function could be made much more computationally efficient by using analytic
    methods to compute Voronoi cell area, rather than the current numerical approach.

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
    Implements `Lloyd's Algorithm <https://en.wikipedia.org/wiki/Lloyd's_algorithm>`_
    without seed ``vectors``; instead, autogenerates the seed ``vectors`` randomly.
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


# Grid functions.

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
    if hasattr(grid, "x_grid") and hasattr(grid, "y_grid"):
        return (grid.x_grid, grid.y_grid)

    # Otherwise, assume it's a tuple
    assert len(grid) == 2, "Expected a 2-tuple with x and y meshgrids."

    return grid


import slmsuite.holography.toolbox.phase as phase


def shift_grid(grid, transform=None, shift=None):
    r"""
    Returns a copy of a coordinate basis ``grid`` with a given ``shift`` and
    ``transformation``. These can be the :math:`\vec{b}` and :math:`M` of a standard
    affine transformation as used elsewhere in the package.
    Such grids are used as arguments for phase patterns, such as those in
    :mod:`slmsuite.holography.toolbox.phase`.

    Parameters
    ----------
    grid : (array_like, array_like) OR :class:`~slmsuite.hardware.slms.slm.SLM`
        Meshgrids of normalized :math:`\frac{x}{\lambda}` coordinates
        corresponding to SLM pixels, in ``(x_grid, y_grid)`` form.
        These are precalculated and stored in any :class:`~slmsuite.hardware.slms.slm.SLM`, so
        such a class can be passed instead of the grids directly.
    transform : float OR ((float, float), (float, float)) OR None
        If a scalar is passed, this is the angle to rotate the basis of the lens by.
        Defaults to zero if ``None``.
        If a 2x2 matrix is passed, transforms the :math:`x` and :math:`y` grids
        according to :math:`x' = M_{00}x + M_{01}y`,  :math:`y' = M_{10}y + M_{11}y`.
    shift : (float, float) OR None
        Translational shift of the grid in normalized :math:`\frac{x}{\lambda}` coordinates.
        Defaults to no shift if ``None``.

    Returns
    -------
    grid : (array_like, array_like)
        The shifted grid
    """
    # Parse arguments.
    (x_grid, y_grid) = _process_grid(grid)

    if transform is None:
        transform = 0

    if shift is None:
        shift = (0, 0)

    if np.isscalar(transform) and transform == 0:   # The trivial case
        return (
            x_grid if shift[0] == 0 else (x_grid - shift[0]),
            y_grid if shift[1] == 0 else (y_grid - shift[1])
        )
    else:                                           # transform is not trivial.
        # Interpret angular transform as a matrix.
        if np.isscalar(transform):
            s = np.sin(transform)
            c = np.cos(transform)
            transform = np.array([[c, -s], [s, c]])
        else:
            transform = np.squeeze(transform)

        assert transform.ndim == 2

        # Use the matrix to transform the grid.
        return (
            transform[0,0] * (x_grid - shift[0]) + transform[0,1] * (y_grid - shift[1]),
            transform[1,0] * (x_grid - shift[0]) + transform[1,1] * (y_grid - shift[1])
        )


# Padding functions.

def pad(matrix, shape):
    """
    Helper function to pad data with zeros. The padding is centered.
    This is used to get higher resolution upon Fourier transform.

    Parameters
    ----------
    matrix : numpy.ndarray
        Data to pad.
    shape : (int, int) OR None
        The desired shape of the ``matrix`` in :mod:`numpy` ``(h, w)`` form.
        If ``None``, the ``matrix`` is returned unpadded.

    Returns
    -------
    numpy.ndarray
        Padded ``matrix``.
    """
    if shape is None:
        return matrix

    deltashape = (
        (shape[0] - matrix.shape[0]) / 2.0,
        (shape[1] - matrix.shape[1]) / 2.0,
    )

    assert (
        deltashape[0] >= 0 and deltashape[1] >= 0
    ), "Shape {} is too large to pad to shape {}".format(tuple(matrix.shape), shape)

    pad_b = int(np.floor(deltashape[0]))
    pad_t = int(np.ceil(deltashape[0]))
    pad_l = int(np.floor(deltashape[1]))
    pad_r = int(np.ceil(deltashape[1]))

    padded = np.pad(
        matrix, [(pad_b, pad_t), (pad_l, pad_r)], mode="constant", constant_values=0
    )

    assert np.all(padded.shape == shape)

    return padded


def unpad(matrix, shape):
    """
    Helper function to unpad data. The padding is assumed to be centered.
    This is used to get higher resolution upon Fourier transform.

    Parameters
    ----------
    matrix : numpy.ndarray OR (int, int)
        Data to unpad. If this is a shape in :mod:`numpy` ``(h, w)`` form,
        returns the four slicing integers used to unpad that shape ``[pad_b:pad_t, pad_l:pad_r]``.
    shape : (int, int) OR None
        The desired shape of the ``matrix`` in :mod:`numpy` ``(h, w)`` form.
        If ``None``, the ``matrix`` is returned unchanged.

    Returns
    ----------
    numpy.ndarray OR (int, int, int, int)
        Either the unpadded ``matrix`` or the four slicing integers used to unpad such a matrix,
        depending what is passed as ``matrix``.
    """
    mshape = np.shape(matrix)
    return_args = False
    if len(mshape) == 1 or np.prod(mshape) == 2:
        # Assume a shape was provided.
        mshape = np.squeeze(matrix)
        return_args = True

    if shape is None:
        if return_args:
            return (0, mshape[0], 0, mshape[1])
        else:
            return matrix

    deltashape = ((shape[0] - mshape[0]) / 2.0, (shape[1] - mshape[1]) / 2.0)

    assert (
        deltashape[0] <= 0 and deltashape[1] <= 0
    ), "Shape {} is too small to unpad to shape {}".format(tuple(mshape), shape)

    pad_b = int(np.floor(-deltashape[0]))
    pad_t = int(mshape[0] - np.ceil(-deltashape[0]))
    pad_l = int(np.floor(-deltashape[1]))
    pad_r = int(mshape[1] - np.ceil(-deltashape[1]))

    if return_args:
        return (pad_b, pad_t, pad_l, pad_r)

    unpadded = matrix[pad_b:pad_t, pad_l:pad_r]

    assert np.all(unpadded.shape == shape)

    return unpadded