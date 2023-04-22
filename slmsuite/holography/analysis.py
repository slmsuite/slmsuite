"""Helper functions for processing images."""

import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from functools import reduce
from scipy.optimize import curve_fit, minimize

from slmsuite.holography.toolbox import format_2vectors
from slmsuite.misc.math import REAL_TYPES
from slmsuite.misc.fitfunctions import gaussian2d


def take(
        images, vectors, size,
        centered=True, integrate=False, clip=False,
        return_mask=False, plot=False, mp=np
    ):
    """
    Crop integration regions around an array of ``vectors``, yielding an array of images.

    Each integration region is a rectangle of the same ``size``. Similar to but more
    general than :meth:`numpy.take`. Useful for gathering data from spots in spot
    arrays. Operates with some speed due to the vectorized nature of implemented slicing.

    Parameters
    ----------
    images : array_like
        2D image or array of 2D images.
    vectors : array_like of floats
        2-vector (or 2-vector array). Location(s) of integration region anchor(s) in pixels,
        see ``centered``.
        See :meth:`~slmsuite.holography.toolbox.format_2vectors`.
    size : int or (int, int)
        Size of the rectangular integration region in ``(w, h)`` format in pixels.
        If a scalar is given, assume square ``(w, w)``.
    centered : bool
        Whether to center the integration region on the ``vectors``.
        If ``False``, ``vectors`` indicates the upper-left corner of the integration region.
        Defaults to ``True``.
    integrate : bool
        If ``True``, the spatial dimension are integrated (summed), yielding a result of the
        same length as the number of vectors. Defaults to ``False``.
    clip : bool
        Whether to allow out-of-range integration regions. ``True`` allows regions outside
        the valid area, setting the invalid region to ``np.nan``
        (or zero if the array datatype does not support ``np.nan``).
        ``False`` throws an error upon out of range. Defaults to ``False``.
    return_mask : bool
        If ``True``, returns a boolean mask corresponding to the regions which are taken
        from. Defaults to ``False``. The average user will ignore this.
    plot : bool
        Calls :meth:`take_plot()` to visualize the images regions.
    mp : module
        If ``images`` are :mod:`cupy` objects, then :mod:`cupy` must be passed as
        ``mp``. Very useful to minimize the cost of moving data between the GPU and CPU.
        Defaults to :mod:`numpy`.
        Indexing variables inside `:meth:`take` still use :mod:`numpy` for speed, no
        matter what module is used.

    Returns
    -------
    numpy.ndarray OR cupy.ndarray
        If ``integrate`` is ``False``, returns an array containing the images cropped
        from the regions of size ``(image_count, h, w)``.
        If ``integrate`` is ``True``, instead returns an array of floats of size ``(image_count,)``
        where each float corresponds to the :meth:`numpy.sum` of a cropped image.
        If ``mp`` is :mod:`cupy`, then a ``cupy.ndarray`` is returned.
    """
    # Clean variables.
    if isinstance(size, REAL_TYPES):
        size = (size, size)

    vectors = format_2vectors(vectors)

    # Prepare helper variables. Future: consider caching for speed, if not negligible.
    edge_x = np.arange(size[0]) - ((int(size[0] - 1) / 2) if centered else 0)
    edge_y = np.arange(size[1]) - ((int(size[1] - 1) / 2) if centered else 0)

    region_x, region_y = np.meshgrid(edge_x, edge_y)

    # Get the lists for the integration regions.
    integration_x = np.around(np.add(
        region_x.ravel()[:, np.newaxis].T, vectors[:][0][:, np.newaxis]
    )).astype(int)
    integration_y = np.around(np.add(
        region_y.ravel()[:, np.newaxis].T, vectors[:][1][:, np.newaxis]
    )).astype(int)

    images = mp.array(images, copy=False)
    shape = mp.shape(images)

    if clip:  # Prevent out-of-range errors by clipping.
        mask = (
            (integration_x < 0) | (integration_x >= shape[-1]) |
            (integration_y < 0) | (integration_y >= shape[-2])
        )

        if np.any(mask):
            # Clip these indices to prevent errors.
            np.clip(integration_x, 0, shape[-1] - 1, out=integration_x)
            np.clip(integration_y, 0, shape[-2] - 1, out=integration_y)
        else:
            # No clipping needed.
            clip = False
    else:
        pass  # Don't prevent out-of-range errors.

    if return_mask:
        canvas = np.zeros(shape[:2], dtype=bool)
        canvas[integration_y, integration_x] = True

        if plot:
            plt.imshow(canvas)
            plt.show()

        return canvas
    else:
        # Take the data, depending on the shape of the images.
        if len(shape) == 2:
            result = images[np.newaxis, integration_y, integration_x]
        elif len(shape) == 3:
            result = images[:, integration_y, integration_x]
        else:
            raise RuntimeError("Unexpected shape for images: {}".format(shape))

        if clip:  # Set values that were out of range to nan instead of erroring.
            try:  # If the datatype of result is incompatible with nan, set to zero instead.
                result[:, mask] = np.nan
            except:
                result[:, mask] = 0
        else:
            pass

        if integrate:  # Sum over the integration axis.
            return mp.squeeze(mp.sum(result, axis=-1))
        else:  # Reshape the integration axis.
            return mp.reshape(result, (vectors.shape[1], size[1], size[0]))


def take_plot(images):
    """
    Plots non-integrated results of :meth:`.take()` in a square array of subplots.

    Parameters
    ----------
    images : numpy.ndarray
        Stack of 2D images, usually a :meth:`take()` output.
    """
    # Gather helper variables and set the min and max of all the subplots.
    (img_count, sy, sx) = np.shape(images)
    M = int(np.ceil(np.sqrt(img_count)))

    sx = sx / 2.0 - 0.5
    sy = sy / 2.0 - 0.5
    extent = (-sx, sx, -sy, sy)

    vmin = np.nanmin(images)
    vmax = np.nanmax(images)

    # Make the figure and subplots.
    plt.figure(figsize=(12, 12))

    for x in range(img_count):
        ax = plt.subplot(M, M, x + 1)

        ax.imshow(
            images[x, :, :],
            vmin=vmin,
            vmax=vmax,
            extent=extent,
            interpolation='none'
        )
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)

    plt.show()


def image_remove_field(images, deviations=1, out=None):
    r"""
    Zeros the field of a stack of images such that moment calculations will succeed.
    Consider, for example, a small spot on a field with strong background.
    Moment calculations in this situation will dominantly measure the moments
    of the background (i.e. the field). This function zeros the image below some threshold.
    This threshold is set to either the mean plus ``deviations`` standard deviations,
    computed uniquely for each image, or the median of each image if ``deviations``
    is ``None``. This is equivalent to background subtraction.

    Parameters
    ----------
    images : numpy.ndarray
        A matrix in the style of the output of :meth:`take()`, with shape ``(image_count, h, w)``, where
        ``(h, w)`` is the width and height of the 2D images and ``image_count`` is the number of
        images. A single image is interpreted correctly as ``(1, h, w)`` even if
        ``(h, w)`` is passed, though the returned image remains shape ``(h, w)`` in that case.
    deviations : int OR None
        Number of standard deviations above the mean to set the threshold.
        If ``None``, uses the median as the threshold instead.
        Defaults to ``None``.
    out : numpy.ndarray or None
        The array to place the output data into. Should be the same shape as ``images``.
        This function operates in-place if ``out`` equals ``images``.

    Returns
    -------
    out : numpy.ndarray
        ``images`` or a copy of ``images``, with each image background-subtracted.
    """
    # Parse images. Convert to float.
    images = np.array(images, copy=False)
    if not isinstance(images.dtype, np.floating):
        images = np.array(images, copy=False, dtype=float)  # Hack to prevent integer underflow.

    # Parse out.
    if out is None:
        out = np.copy(images)
    elif out is not images:
        out = np.copyto(out, images)

    # Make sure that we're testing 3D images.
    single_image = len(images.shape) == 2
    if single_image:
        images_ = np.reshape(images, (1, images.shape[0], images.shape[1]))
    else:
        images_ = images
    img_count = images_.shape[0]

    # Generate the threshold.
    if deviations is None:  # Median case
        threshold = np.nanmedian(images_, axis=(1, 2))
    else:   # Mean + deviations * std case
        threshold = (
            np.nanmean(images_, axis=(1, 2))
            + deviations*np.nanstd(images_, axis=(1, 2))
        )
    if not single_image:
        threshold = np.reshape(threshold, (img_count, 1, 1))

    # Remove the field. This needs the float from before. Unsigned integer could underflow.
    out -= threshold
    out[out < 0] = 0

    return out


def image_moment(images, moment=(1, 0), centers=(0, 0), normalize=True, nansum=False):
    r"""
    Computes the given `moment <https://en.wikipedia.org/wiki/Moment_(mathematics)>`_
    :math:`M_{m_xm_y}` for a stack of images.
    This involves integrating each image against polynomial trial functions:

    .. math:: M_{m_xm_y} = \frac{   \int_{-w_x/2}^{+w_x/2} dx \, (x-c_x)^{m_x}
                                    \int_{-w_y/2}^{+w_y/2} dy \, (y-c_y)^{m_y}
                                    P(x+x_0, y+y_0)
                                }{  \int_{-w_x/2}^{+w_x/2} dx \,
                                    \int_{-w_y/2}^{+w_y/2} dy \,
                                    P(x, y)},

    where :math:`P(x, y)` is a given 2D image, :math:`(x_0, y_0)` is the center of a
    window of size :math:`w_x \times w_y`, and :math:`(c_x, c_y)` is a shift in the
    center of the trial functions.

    Warning
    ~~~~~~~
    This function does not check for or correct for negative values in ``images``.
    Negative values may produce unusual results.

    Warning
    ~~~~~~~
    Higher order even moments (e.g. 2) will potentially yield unexpected results if
    the images are not background-subtracted. For instance, a calculation on an image
    with large background will yield the moment of the window, rather than say anything
    about the image. Consider using :meth:`image_remove_field()` to background-subtract.

    Parameters
    ----------
    images : numpy.ndarray
        A matrix in the style of the output of :meth:`take()`, with shape ``(image_count, h, w)``, where
        ``(h, w)`` is the width and height of the 2D images and ``image_count`` is the number of
        images. A single image is interpreted correctly as ``(1, h, w)`` even if
        ``(h, w)`` is passed.
    moment : (int, int)
        The moments in the :math:`x` and :math:`y` directions: :math:`(m_x, m_y)`. For instance,

        - :math:`M_{m_xm_y} = M_{10}` corresponds to the :math:`x` moment or
          the position in the :math:`x` dimension.
        - :math:`M_{m_xm_y} = M_{11}` corresponds to :math:`xy` shear.
        - :math:`M_{m_xm_y} = M_{02}` corresponds to the :math:`y^2` moment, or the variance
          (squared width for a Gaussian) in the :math:`y` direction,
          given a zero or zeroed (via ``centers``) :math:`M_{01}` moment.

    centers : tuple or numpy.ndarray
        Perturbations to the center of the trial function, :math:`(c_x, c_y)`.
    normalize : bool
        Whether to normalize ``images``.
        If ``False``, normalization is assumed to have been precomputed.
    nansum : bool
        Whether to use :meth:`numpy.nansum()` in place of :meth:`numpy.sum()`.
        :meth:`numpy.nansum()` treats ``nan`` values as zeros.
        This is useful in the case where ``clip=True`` is passed to :meth:`take()`
        (out of range is set to ``nan``).

    Returns
    -------
    numpy.ndarray
        The moment :math:`M_{m_xm_y}` evaluated for every image. This is of size ``(image_count,)``
        for provided ``images`` data of shape ``(image_count, h, w)``.
    """
    images = np.array(images, copy=False)
    if len(images.shape) == 2:
        images = np.reshape(images, (1, images.shape[0], images.shape[1]))
    (img_count, w_y, w_x) = images.shape

    if nansum:
        np_sum = np.nansum
    else:
        np_sum = np.sum

    if normalize:
        normalization = np_sum(images, axis=(1, 2), keepdims=False)
        reciprical = np.reciprocal(
            normalization, where=normalization != 0, out=np.zeros(img_count,)
        )
    else:
        reciprical = 1

    if moment[0] == 0 and moment[1] == 0:  # 0,0 (norm) case
        if normalize:
            return np.ones((img_count,))
        else:
            return np_sum(images, axis=(1, 2), keepdims=False)
    else:
        if len(np.shape(centers)) == 2:
            c_x = np.reshape(centers[0], (img_count, 1, 1))
            c_y = np.reshape(centers[1], (img_count, 1, 1))
        elif len(np.shape(centers)) == 1:
            c_x = centers[0]
            c_y = centers[1]

        edge_x = np.reshape(np.arange(w_x) - (w_x - 1) / 2.0, (1, 1, w_x)) - c_x
        edge_y = np.reshape(np.arange(w_y) - (w_y - 1) / 2.0, (1, w_y, 1)) - c_y

        edge_x = np.power(edge_x, moment[0], out=edge_x)
        edge_y = np.power(edge_y, moment[1], out=edge_y)

        if moment[1] == 0:  # only x case
            return np_sum(images * edge_x, axis=(1, 2), keepdims=False) * reciprical
        elif moment[0] == 0:  # only y case
            return np_sum(images * edge_y, axis=(1, 2), keepdims=False) * reciprical
        else:  # shear case
            return np_sum(images * edge_x * edge_y, axis=(1, 2), keepdims=False) * reciprical


def image_normalization(images, nansum=False):
    """
    Computes the zeroth order moments, equivalent to spot mass or normalization,
    for a stack of images.

    Parameters
    ----------
    images : numpy.ndarray
        A matrix in the style of the output of :meth:`take()`, with shape ``(image_count, h, w)``, where
        ``(h, w)`` is the width and height of the 2D images and ``image_count`` is the number of
        images. A single image is interpreted correctly as ``(1, h, w)`` even if
        ``(h, w)`` is passed.
    nansum : bool
        Whether to use :meth:`numpy.nansum()` in place of :meth:`numpy.sum()`.

    Returns
    -------
    numpy.ndarray
        The normalization factor :math:`M_{11}` in an array of shape ``(image_count,)``.
    """
    return image_moment(images, (0, 0), normalize=False, nansum=nansum)


def image_normalize(images, nansum=False, remove_field=False):
    """
    Normalizes of a stack of images via the the zeroth order moments.

    Parameters
    ----------
    images : numpy.ndarray
        A matrix in the style of the output of :meth:`take()`, with shape ``(image_count, h, w)``, where
        ``(h, w)`` is the width and height of the 2D images and ``image_count`` is the number of
        images. A single image is interpreted correctly as ``(1, h, w)`` even if
        ``(h, w)`` is passed, though the returned image remains shape ``(h, w)`` in that case.
    nansum : bool
        Whether to use :meth:`numpy.nansum()` in place of :meth:`numpy.sum()`.
    remove_field : bool
        Whether to apply :meth:`.image_remove_field()` to avoid background-dominated moments.

    Returns
    -------
    images_normalized : numpy.ndarray
        A copy of ``images``, with each image normalized.
    """
    if remove_field:
        images = image_remove_field(images)
    else:
        images = np.array(images, copy=False, dtype=float)

    single_image = len(images.shape) == 2

    normalization = image_normalization(images, nansum=nansum)

    if single_image:
        normalization = np.asscalar(normalization)
        if normalization == 0:
            return np.zeros_like(images)
        else:
            return images / normalization
    else:
        reciprical = np.reciprocal(
            normalization, where=normalization != 0, out=np.zeros(len(normalization))
        )
        return images * np.reshape(reciprical, (len(normalization), 1, 1))


def image_positions(images, normalize=True, nansum=False):
    """
    Computes the two first order moments, equivalent to spot position relative to image center,
    for a stack of images.
    Specifically, returns :math:`M_{10}` and :math:`M_{01}`.

    Parameters
    ----------
    images : numpy.ndarray
        A matrix in the style of the output of :meth:`take()`, with shape ``(image_count, h, w)``, where
        ``(h, w)`` is the width and height of the 2D images and ``image_count`` is the number of
        images. A single image is interpreted correctly as ``(1, h, w)`` even if
        ``(h, w)`` is passed.
    normalize : bool
        Whether to normalize ``images``.
        If ``False``, normalization is assumed to have been precomputed.
    nansum : bool
        Whether to use :meth:`numpy.nansum()` in place of :meth:`numpy.sum()`.

    Returns
    -------
    numpy.ndarray
        Stack of :math:`M_{10}`, :math:`M_{01}`.
    """
    if normalize:
        images = image_normalize(images, nansum=nansum)

    return np.vstack(
        (
            image_moment(images, (1, 0), normalize=False, nansum=nansum),
            image_moment(images, (0, 1), normalize=False, nansum=nansum),
        )
    )


def image_variances(images, centers=None, normalize=True, nansum=False):
    r"""
    Computes the three second order central moments, equivalent to variance, for a stack
    of images.
    Specifically, this function returns a stack of the moments :math:`M_{20}` and
    :math:`M_{02}`, along with :math:`M_{11}`, which are the variance in the :math:`x`
    and :math:`y` directions, along with the so-called shear variance.
    Recall that variance defined as

    .. math:: (\Delta x)^2 = \left<(x - \left<x\right>)^2\right>.

    This equation is made central by subtraction of :math:`\left<x\right>`.
    The user can of course use :meth:`take_moment` directly to access the
    non-central moments; this function is a helper to access useful quantities
    for analysis of spot size and skewness.

    Parameters
    ----------
    images : numpy.ndarray
        A matrix in the style of the output of :meth:`take()`, with shape ``(image_count, h, w)``, where
        ``(h, w)`` is the width and height of the 2D images and ``image_count`` is the number of
        images. A single image is interpreted correctly as ``(1, h, w)`` even if
        ``(h, w)`` is passed.
    centers : numpy.ndarray OR None
        If the user has already computed :math:`\left<x\right>`, for example via
        :meth:`image_positions()`, then this can be passed though ``centers``. The default
        None computes ``centers`` internally.
    normalize : bool
        Whether to normalize ``images``.
        If ``False``, normalization is assumed to have been precomputed.
    nansum : bool
        Whether to use :meth:`numpy.nansum()` in place of :meth:`numpy.sum()`.

    Returns
    -------
    numpy.ndarray
        Stack of :math:`M_{20}`, :math:`M_{02}`, and :math:`M_{11}`. Shape ``(3, image_count)``.
    """
    if normalize:
        images = image_normalize(images, nansum=nansum)

    if centers is None:
        centers = image_positions(images, normalize=False, nansum=nansum)

    m20 = image_moment(images, (2, 0), centers=centers, normalize=False, nansum=nansum)
    m11 = image_moment(images, (1, 1), centers=centers, normalize=False, nansum=nansum)
    m02 = image_moment(images, (0, 2), centers=centers, normalize=False, nansum=nansum)

    return np.vstack((m20, m02, m11))


def image_ellipticity(variances):
    r"""
    Given the output of :meth:`image_variances()`,
    return a measure of spot ellipticity for each moment triplet.
    The output of :meth:`image_variances()` contains the moments :math:`M_{20}`,
    :math:`M_{02}`, and :math:`M_{11}`. These terms make up a :math:`2 \times 2` matrix,
    which is equivalent to a rotated elliptical scaling according to the eigenvalues
    :math:`\lambda_+` and :math:`\lambda_-` and some rotation matrix :math:`R(\phi)`.

    .. math::   \begin{bmatrix}
                    M_{20} & M_{11} \\
                    M_{11} & M_{02} \\
                \end{bmatrix}
                =
                R(-\phi)
                \begin{bmatrix}
                    \lambda_+ & 0 \\
                    0 & \lambda_- \\
                \end{bmatrix}
                R(\phi).

    We use this knowledge, along with tricks for eigenvalue calculations on
    :math:`2 \times 2` matrices, to build up a metric for ellipticity:

    .. math:: \mathcal{E} = 1 - \frac{\lambda_-}{\lambda_+}.

    Notice that

    - when :math:`\lambda_+ = \lambda_-` (isotropic scaling), the metric is zero and
    - when :math:`\lambda_- = 0` (flattened to a line), the metric is unity.

    Parameters
    ----------
    variances : numpy.ndarray
        The output of :meth:`image_variances()`. Shape ``(3, image_count)``.

    Returns
    -------
    numpy.ndarray
        Array of ellipticities for the given moments. Shape ``(image_count,)``.
    """
    m20 = variances[0, :]
    m02 = variances[1, :]
    m11 = variances[2, :]

    # We can use a trick for eigenvalue calculations of 2x2 matrices to avoid
    # more complicated calculations.
    half_trace = (m20 + m02) / 2
    determinant = m20 * m02 - m11 * m11

    eig_half_difference = np.sqrt(np.square(half_trace) - determinant)

    eig_plus = half_trace + eig_half_difference
    eig_minus = half_trace - eig_half_difference

    return 1 - (eig_minus / eig_plus)


def image_ellipticity_angle(variances):
    r"""
    Given the output of :meth:`image_variances()`,
    return the rotation angle of the scaled basis for each moment triplet.
    This is the angle between the :math:`x` axis and the
    major axis (large eigenvalue axis).

    Parameters
    ----------
    moment2 : numpy.ndarray
        The output of :meth:`image_variances()`. Shape ``(3, image_count)``.

    Returns
    -------
    numpy.ndarray
        Array of angles for the given moments.
        For highly circular spots, this angle is not meaningful, and dominated by
        experimental noise.
        For perfectly circular spots, zero is returned.
        Shape ``(image_count,)``.
    """
    m20 = variances[0, :]
    m02 = variances[1, :]
    m11 = variances[2, :]

    # Some quick math (see image_ellipticity).
    half_trace = (m20 + m02) / 2
    determinant = m20 * m02 - m11 * m11

    eig_plus = half_trace + np.sqrt(np.square(half_trace) - determinant)

    # We know that M * v = eig_plus * v. This yields a system of equations:
    #   m20 * x + m11 * y = eig_plus * x
    #   m11 * x + m02 * y = eig_plus * y
    # We're trying to solve for angle, which is just atan(x/y). We can solve for x/y:
    #   m11 * x = (eig_plus - m02) * y        ==>         x/y = (eig_plus - m02) / m11
    return np.arctan2(eig_plus - m02, m11, where=m11 != 0, out=np.zeros_like(m11))


def image_fit(images, grid_ravel=None, function=gaussian2d, guess=None, plot=False):
    """
    Fit each image in a stack of images to a 2D ``function``.

    Parameters
    ----------
    images : numpy.ndarray (image_count, height, width)
        An image or array of images to fit. A single image is interpreted correctly as
        ``(1, h, w)`` even if ``(h, w)`` is passed.
    grid_ravel : 2-tuple of array_like of reals (height * width)
        Raveled components of the meshgrid describing coordinates over the images.
    function : lambda ((float, float), ... ) -> float
        Some fitfunction which accepts ``(x,y)`` coordinates as first argument.
        Defaults to :meth:`~slmsuite.misc.fitfunctions.gaussian2d()`.
    guess : None OR numpy.ndarray (parameter_count, image_count)
        - If ``guess`` is ``None``, will construct a guess based on the ``function`` passed.
          Functions for which guesses are implemented include:

          - :meth:`~~slmsuite.misc.fitfunctions.gaussian2d()`

        - If ``guess`` is ``None`` and ``function`` does not have a guess
          implemented, no guess will be provided to the optimizer.
        - If ``guess`` is a ``numpy.ndarray``, a slice of the array will be provided
          to the optimizer as a guess for the fit parameters for each image.
    plot : bool
        Whether to create a plot for each fit.
    show : bool
        Whether or not to call :meth:`matplotlib.pyplot.show` after generating
        the plot.

    Returns
    -------
    numpy.ndarray (``result_count``, ``image_count``)
        A matrix with the fit results. The first row
        contains the rsquared quality of each fit.
        The values in the remaining rows correspond to the parameters
        for the supplied fit function.
        Failed fits have an rsquared of ``numpy.nan`` and parameters
        are set to the provided or constructed guess or ``numpy.nan``
        if no guess was provided or constructed.

    Raises
    ------
    NotImplementedError
        If the provided ``function`` does not have a guess implemented.
    """
    # Setup.
    if images.ndim == 2:
        images = images.reshape((1, *images.shape))
    (image_count, w_y, w_x) = images.shape
    img_shape = (w_y, w_x)

    if grid_ravel is None:
        edge_x = np.reshape(np.arange(w_x) - (w_x - 1) / 2.0, (1, 1, w_x))
        edge_y = np.reshape(np.arange(w_y) - (w_y - 1) / 2.0, (1, w_y, 1))
        grid = np.meshgrid(edge_x, edge_y)
        grid_ravel = (grid[0].ravel(), grid[1].ravel())

    # Number of fit parameters the function accepts.
    param_count =  function.__code__.co_argcount - 1

    # Number of parameters to return.
    result_count = param_count + 1
    result = np.full((result_count, image_count), np.nan)

    # Construct guesses.
    if guess is None:
        if function is gaussian2d:
            images_normalized = image_normalize(images, remove_field=True)
            centers = image_positions(images_normalized, normalize=False)
            variances = image_variances(images_normalized, centers=centers, normalize=False)
            maxs = np.amax(images, axis=(1, 2))
            mins = np.amin(images, axis=(1, 2))

            guess = np.vstack((
                centers,
                maxs - mins,
                mins,
                np.sqrt(variances[0:2, :]),
                variances[2, :]
            ))

            guess_raw = np.vstack((
                centers,
                maxs - mins,
                mins,
                variances[0:2, :],
                variances[2, :]
            ))

    # Fit and plot each image.
    for img_idx in range(image_count):
        img = images[img_idx, :, :].ravel()

        # Get guess.
        p0 = None if guess is None else guess[:, img_idx]

        # Attempt fit.
        fit_succeeded = True
        popt = None

        try:
            popt, _ = curve_fit(function, grid_ravel, img, ftol=1e-5, p0=p0,)
        except RuntimeError:    # The fit failed if scipy says so.
            fit_succeeded = False
        else:                   # The fit failed if any of the parameters aren't finite.
            if np.any(np.logical_not(np.isfinite(popt))):
                fit_succeeded = False

        if fit_succeeded:   # Calculate r2.
            ss_res = np.sum(np.square(img - function(grid_ravel, *popt)))
            ss_tot = np.sum(np.square(img - np.mean(img)))
            r2 = 1 - (ss_res / ss_tot)
        else:               # r2 is nan and the fit parameters are the guess or nan.
            popt = p0 if p0 is not None else np.full(param_count, np.nan)
            r2 = np.nan

        result[0, img_idx] = r2
        result[1:, img_idx] = popt

        # Plot.
        if plot:
            # Data.
            data = np.reshape(img, img_shape)
            if p0 is not None:
                guess_ = np.reshape(function(grid_ravel, *p0), img_shape)
            else:
                guess_ = np.zeros(img_shape)
            result_ = np.reshape(function(grid_ravel, *popt), img_shape)
            vmin = np.min((
                np.min(data),
                np.min(guess_) if p0 is not None else np.inf,
                np.min(result_)
            ))
            vmax = np.max((
                np.max(data),
                np.max(guess_) if p0 is not None else -np.inf,
                np.max(result_)
            ))

            # Plot.
            fig, axs = plt.subplots(1, 3, figsize=(3 * 6.4, 4.8))
            fig.suptitle("Image {}".format(img_idx))
            ax0, ax1, ax2 = axs
            ax0.imshow(data, vmin=vmin, vmax=vmax)
            ax0.set_title("Data")
            ax1.imshow(guess_, vmin=vmin, vmax=vmax)
            ax1.set_title("Guess")
            ax2.imshow(result_, vmin=vmin, vmax=vmax)
            ax2.set_title("Result")

            plt.show()

    return result


def fit_affine(x, y, guess_affine=None, plot=False):
    r"""
    For two sets of points with equal length, find the best-fit affine
    transformation that transforms from the first basis to the second.
    Best fit is defined as minimization on the least squares euclidean norm.

    .. math:: \vec{y} = M \cdot \vec{x} + \vec{b}

    Parameters
    ----------
    x, y : array_like
        Array of vectors of shape ``(2, N)`` in the style of :meth:`format_2vectors()`.
    plot : bool
        Whether to produce a debug plot.

    Returns
    -------
    dict
        A dictionary with fields ``"M"`` and ``"b"``.
    """
    x = format_2vectors(x)
    y = format_2vectors(y)
    assert x.shape == y.shape

    if guess_affine is None:
        # Calculate the centroids and the centered coordinates.
        xc = np.nanmean(x, axis=1)[:, np.newaxis]
        yc = np.nanmean(y, axis=1)[:, np.newaxis]

        if np.all(np.isnan(xc)) or np.all(np.isnan(yc)):
            raise ValueError("analysis.py: vectors cannot contain a row of all-nan values")

        x_ = x - xc
        y_ = y - yc

        # Points very close to the centroid have disproportionate influence on the guess.
        # Ignore the points which are closer than a median-dependent threshold.
        threshold = np.median(np.sqrt(np.sum(np.square(x_), axis=0))) / 2

        # Generate a guess transformation.
        nan_list = np.full_like(y_[0,:], np.nan)

        # This could probably be vectorized more. Also not sure if all corner cases work.
        M_guess = np.array([
            [
                np.nanmean(np.divide(y_[0,:], x_[0,:], where=x_[0,:] > threshold, out=nan_list.copy())),
                np.nanmean(np.divide(y_[0,:], x_[1,:], where=x_[1,:] > threshold, out=nan_list.copy()))
            ],
            [
                np.nanmean(np.divide(y_[1,:], x_[0,:], where=x_[0,:] > threshold, out=nan_list.copy())),
                np.nanmean(np.divide(y_[1,:], x_[1,:], where=x_[1,:] > threshold, out=nan_list.copy()))
            ]
        ])

        # Fix nan instances. This means the matrix is no longer unique, so we choose the
        # case where the nans are mapped to zero.
        M_guess[np.isnan(M_guess)] = 0

        # Back compute the offset.
        b_guess = yc - np.matmul(M_guess, xc)
    else:
        if isinstance(guess_affine, dict) and "M" in guess_affine and "b" in guess_affine:
            M_guess = guess_affine["M"]
            b_guess = guess_affine["b"]
        else:
            raise ValueError("analysis.py: guess_affine must be a dictionary with 'M' and 'b' fields.")

    # Least squares fit.
    def err(p):
        M = np.array([[p[0], p[1]], [p[2], p[3]]])
        b = format_2vectors([p[4], p[5]])

        y_ = np.matmul(M, x) + b

        return np.nansum(np.square(y_ - y))

    guess = (M_guess[0,0], M_guess[0,1], M_guess[1,0], M_guess[1,1], b_guess[0,0], b_guess[1,0])

    try:        # Try with default scipy minimization. (Future: better opt than minimize?).
        m = minimize(err, x0=guess)
        p = [float(pp) for pp in m.x]

        M = np.array([[p[0], p[1]], [p[2], p[3]]])
        b = format_2vectors([p[4], p[5]])
    except:     # Fail elegantly (warn the user?).
        M = M_guess
        b = b_guess

    # Debug plot if desired.
    if plot and x.shape[0] == 2:
        plt.scatter(y[0,:], y[1,:], s=20, fc="b", ec="b")

        result_guess = np.matmul(M_guess, x) + b_guess
        plt.scatter(result_guess[0,:], result_guess[1,:], s=40, fc="none", ec="r")

        result = np.matmul(M, x) + b
        plt.scatter(result[0,:], result[1,:], s=60, fc="none", ec="g")

        plt.gca().set_aspect("equal")
        plt.show()

    # Return as a dictionary
    return {"M":M, "b":b}


def blob_detect(
    img,
    filter=None,
    plot=False,
    title="",
    fig=None,
    axs=None,
    zoom=False,
    show=False,
    **kwargs
):
    """
    Detect blobs in an image.

    Wraps :class:`cv2.SimpleBlobDetector` (see
    `these <https://docs.opencv.org/3.4/d8/da7/structcv_1_1SimpleBlobDetector_1_1Params.html>`_
    `links <https://learnopencv.com/blob-detection-using-opencv-python-c/>`_)
    Default parameters are optimized for bright spot detection on dark background,
    but can be changed with ``**kwargs``.

    Parameters
    ----------
    img : numpy.ndarray
        The image to perform blob detection on.
    filter : {"dist_to_center", "max_amp"} OR None
        One of ``dist_to_center`` or ``max_amp``.
    plot : bool
        Whether to show a debug plot.
    title : str
        Plot title.
    fig : matplotlib.figure.Figure
        Figure for plotting.
    axs : list of matplotlib.axes.Axis or matplotlib.axes.Axis
        Axes for plotting.
    show : bool
        Whether or not to show the plot.
    **kwargs
       Extra arguments for :class:`cv2.SimpleBlobDetector`.

    Returns
    -------
    blobs : ndarray
        List of blobs found by  ``detector``.
    detector : :class:`cv2.SimpleBlobDetector`
        A blob detector with customized parameters.
    """
    img_8it = _make_8bit(np.copy(img))
    params = cv2.SimpleBlobDetector_Params()

    # Configure default parameters
    params.blobColor = 255
    params.minThreshold = 10
    params.maxThreshold = 255
    params.thresholdStep = 10
    # params.minArea = 0
    params.filterByArea = False  # Can be changed to compute rel to diffraction limit
    params.filterByCircularity = False
    params.filterByConvexity = False
    params.filterByInertia = False

    # Load in custom configuration
    for key, val in kwargs.items():
        setattr(params, key, val)

    # Create the detector and detect blobs
    detector = cv2.SimpleBlobDetector_create(params)
    blobs = detector.detect(img_8it)

    if len(blobs) == 0:
        raise Exception("No blobs found! Try blurring image.")

    # Sort blobs according to `filter`.
    if filter == "dist_to_center":
        dist_to_center = [
            np.linalg.norm(np.array(blob.pt) - np.array(img.shape[::-1]) / 2)
            for blob in blobs
        ]
        blobs = [blobs[np.argmin(dist_to_center)]]
    elif filter == "max_amp":
        bin_size = int(np.mean([blob.size for blob in blobs]))
        for i, blob in enumerate(blobs):
            # Try fails when blob is on edge of camera.
            try:
                blobs[i].response = float(
                    img_8it[
                        np.ix_(
                            int(blob.pt[1]) + np.arange(-bin_size, bin_size),
                            int(blob.pt[0]) + np.arange(-bin_size, bin_size),
                        )
                    ].sum()
                )
            except Exception:
                blobs[i].response = float(0)
        blobs = [blobs[np.argmax([blob.response for blob in blobs])]]

    if plot:
        # Get blob statistics.
        blob_count = len(blobs)
        blob_centers = np.zeros((2, blob_count))
        blob_diameters = np.zeros(blob_count)
        for (blob_idx, blob) in enumerate(blobs):
            blob_centers[:, blob_idx] = blob.pt
            blob_diameters[blob_idx] = blob.size
        blob_xs = blob_centers[0, :]
        blob_ys = blob_centers[1, :]
        blob_xmin = np.min(blob_xs)
        blob_xmax = np.max(blob_xs)
        blob_ymin = np.min(blob_ys)
        blob_ymax = np.max(blob_ys)
        zoom_padx = 2 * (blob_xmax - blob_xmin) / blob_count
        zoom_pady = 2 * (blob_ymax - blob_ymin) / blob_count

        # Plot setup.
        if fig is None and axs is None:
            if zoom:
                fig, axs = plt.subplots(1, 2)
            else:
                fig, axs = plt.subplots(1, 1)
                axs = (axs,)
        if zoom:
            ax0, ax1 = axs
        else:
            ax0, = axs
        fig.suptitle(title)

        # Full image.
        vmin = np.min(img_8it)
        vmax = np.max(img_8it)
        im = ax0.imshow(img_8it, vmin=vmin, vmax=vmax)

        # Zoomed Image.
        if zoom:
            ax1.imshow(img_8it, vmin=vmin, vmax=vmax)
            xmax = img.shape[1] + 0.5
            xmin = 0.5
            xlims = (
                np.clip((blob_xmin - zoom_padx), xmin, xmax),
                np.clip((blob_xmax + zoom_padx), xmin, xmax)
            )
            ymax = img.shape[0] + 0.5
            ymin = 0.5
            ylims = (
                np.clip((blob_ymin - zoom_pady), ymin, ymax),
                np.clip((blob_ymax + zoom_pady), ymin, ymax)
            )
            ax1.set_xlim(xlims)
            ax1.set_ylim(np.flip(ylims))
            ax1.set_title("Zoom")
            fig.colorbar(im, ax=ax1)
            ax0.set_title("Full")
        else:
            fig.colorbar(im, ax=ax0)

        # Blob patches
        for blob_idx in range(blob_count):
                patch = matplotlib.patches.Circle(
                    (float(blob_centers[0, blob_idx]), float(blob_centers[1, blob_idx])),
                    radius=float(blob_diameters[blob_idx] / 2),
                    color="red",
                    linewidth=1,
                    fill=None
                )
                ax0.add_patch(patch)
                if zoom:
                    ax1.add_patch(patch)
        if show:
            plt.show()

    return blobs, detector


def blob_array_detect(
    img,
    size,
    orientation=None,
    orientation_check=True,
    dft_threshold=50,
    dft_padding=0,
    plot=False,
):
    r"""
    Detect an array of spots and return the orientation as an affine transformation.
    Primarily used for calibration.

    For a rectangular array of spots imaged in ``img``,
    find the variables :math:`\vec{M}` and :math:`\vec{b}` for the  affine transformation

    .. math:: \vec{y} = M \cdot \vec{x} + \vec{b}

    which converts spot indices :math:`\vec{x}` into camera pixel indices :math:`\vec{y}`.

    Parameters
    ----------
    img : numpy.ndarray
        The image in question.
    size : (int, int) OR int
        The size of the rectangular array in number of spots ``(Nx, Ny)``.
        If a single ``int`` size is given, then assume ``(N, N)``.
    orientation : dict or None
        Guess array orientation (same format as the returned) from previous known results.
        If None (the default), orientation is estimated from looking for peaks in the
        Fourier transform of the image.
    orientation_check : bool
        If enabled, looks for two missing spots at one corner as a parity check on rotation.
        Used by :meth:`~slmsuite.hardware.cameraslms.FourierSLM.fourier_calibrate()`.
        See :meth:`~slmsuite.hardware.cameraslms.FourierSLM.make_rectangular_array()`.
    dft_threshold : float in [0, 255]
        Minimum value of peak in blob detect of the DFT of `img` when `orientation` is `None`.
        Passed as kwarg to :meth:`blob_detect` with name `minThreshold`.
    dft_padding : int
        Increases the dimensions of the padded `img` before the DFT is taken when `orientation`
        is `None`. Dimensions are increased by a factor of `2 ** dft_padding`.
        Increasing this value increases the k-space resolution of the DFT,
        and can improve orientation detection.
    plot : bool
        Whether or not to plot debug plots. Default is ``False``.

    Returns
    --------
    dict
        Orientation dictionary with the following keys, corresponding to
        the affine transformation:

         - ``"M"`` : ``numpy.ndarray`` (2, 2).
         - ``"b"`` : ``numpy.ndarray`` (2, 1).
    """
    img_8it = _make_8bit(img)

    if orientation is not None:     # If an orientation was provided, use this as a guess.
        M = orientation["M"]
    else:                           # Otherwise, find a guess orientation.
        # FFT to find array pitch and orientation.
        # Take the largest dimension rounded up to nearest power of 2.
        # Future: clean this up to behave like other parts of the package.
        fftsize = int(2 ** np.ceil(np.log2(np.max(np.shape(img))))) * 2 ** dft_padding
        dft = np.fft.fftshift(np.fft.fft2(img_8it, s=[fftsize, fftsize]))
        fft_blur_size = (
            int(2 * np.ceil(fftsize / 1000)) + 1
        )  # Future: Make not arbitrary.
        dft_amp = cv2.GaussianBlur(np.abs(dft), (fft_blur_size, fft_blur_size), 0)

        # Need copy for some reason:
        # https://github.com/opencv/opencv/issues/18120
        thresholdStep = 10
        blobs, _ = blob_detect(
            dft_amp.copy(),
            plot=False,
            minThreshold=dft_threshold,
            thresholdStep=thresholdStep,
        )
        blobs = np.array(blobs)
        dft_fit_failed = len(blobs) < 5

        if plot:
            fig, axs = plt.subplots(1, 2, figsize=(12, 6), facecolor='white')

            plt_img = _make_8bit(dft_amp.copy())

            # Determine the bounds of the zoom region, padded by zoom_pad
            zoom_pad = 50

            x = np.array([blob.pt[0] for blob in blobs])
            xl = [
                np.clip(np.amin(x) - zoom_pad, 0, dft_amp.shape[1]),
                np.clip(np.amax(x) + zoom_pad, 0, dft_amp.shape[1]),
            ]

            y = np.array([blob.pt[1] for blob in blobs])
            yl = [
                np.clip(np.amin(y) - zoom_pad, 0, dft_amp.shape[0]),
                np.clip(np.amax(y) + zoom_pad, 0, dft_amp.shape[0]),
            ]

            # Plot the unzoomed figure
            axs[0].imshow(plt_img)

            # Plot a red rectangle to show the extents of the zoom region
            rect = plt.Rectangle(
                (float(xl[0]), float(yl[0])),
                float(np.diff(xl)), float(np.diff(yl)),
                ec="r", fc="none"
            )
            axs[0].add_patch(rect)
            axs[0].set_title("DFT Result - Full")
            axs[0].set_xticks([])
            axs[0].set_yticks([])

            # Plot the zoomed figure
            axs[1].imshow(plt_img)
            axs[1].scatter(
                x,
                y,
                facecolors="none",
                edgecolors="r",
                marker="o",
                s=1000,
                linewidths=1,
            )
            for spine in ["top", "bottom", "right", "left"]:
                axs[1].spines[spine].set_color("r")
                axs[1].spines[spine].set_linewidth(1.5)
            axs[1].set_title("DFT Result - Zoom")
            axs[1].set_xticks([])
            axs[1].set_yticks([])
            axs[1].set_xlim(xl)
            axs[1].set_ylim(np.flip(yl))

            for ax in axs:
                ax.set_xlabel("Image Reciprocal $x$ [1/pix]")
                ax.set_ylabel("Image Reciprocal $y$ [1/pix]")
            fig.tight_layout(pad=4.0)

            plt.show()

        if dft_fit_failed:
            blobs, _ = blob_detect(
                dft_amp.copy(),
                plot=True,
                minThreshold=dft_threshold,
                thresholdStep=thresholdStep,
            )

            raise RuntimeError(
                "Not enough spots found in DFT, expected 5. Try:\n"
                "- increasing exposure time\n"
                "- increasing `dft_padding`\n"
                "- decreasing `dft_threshold`"
            )

        # Future: Revamp to improve this part of the algorithm.
        #
        # Failure paths include:
        # - Accidentally finding 5 points in a line on one axis instead of 5 points
        #   making a plus sign.
        # - Adjacent points being drowned out by the strength of the 0th order.
        # - etc.
        #
        # Potential paths for revamp:
        # - fit a Bravais lattice more directly
        # - e.g. using np.fft.fftfreq and np.fft.fftshift

        # 2.1) Get the max point (DTF 0th order) and its next four neighbors.
        blob_dist = np.zeros(len(blobs))
        k = np.zeros((len(blobs), 2))
        for i, blob in enumerate(blobs):
            k[i, 0] = -1 / 2 + blob.pt[0] / dft_amp.shape[1]
            k[i, 1] = -1 / 2 + blob.pt[1] / dft_amp.shape[0]

            # Assumes max at 0th order.
            blob_dist[i] = np.linalg.norm(
                np.array([k[i, 0], k[i, 1]])
            )

        sort_ind = np.argsort(blob_dist)[:5]
        blobs = blobs[sort_ind]
        blob_dist = blob_dist[sort_ind]
        k = k[sort_ind]

        # 2.2) Calculate array metrics.
        left =   np.argmin([k[:, 0]])  # Smallest x
        right =  np.argmax([k[:, 0]])  # Largest x
        bottom = np.argmin([k[:, 1]])  # Smallest y
        top =    np.argmax([k[:, 1]])  # Largest y

        # 2.3) Calculate the vectors in the imaging domain.
        x = 2 * (k[right, :] - k[left, :]) / (blob_dist[right] + blob_dist[left]) ** 2
        y = 2 * (k[top, :] - k[bottom, :]) / (blob_dist[top] + blob_dist[bottom]) ** 2

        M = np.array([[x[0], y[0]], [x[1], y[1]]])

    # 3) Make the array kernel for convolutional detection of the array center.
    # Make lists that we will use to make the kernel: the array...
    x_list = np.arange(-(size[0] - 1) / 2.0, (size[0] + 1) / 2.0)
    y_list = np.arange(-(size[1] - 1) / 2.0, (size[1] + 1) / 2.0)

    x_centergrid, y_centergrid = np.meshgrid(x_list, y_list)
    centers = np.vstack((x_centergrid.ravel(), y_centergrid.ravel()))

    # ...and the array padded by one.
    pad = 1
    p = int(pad * 2)

    x_list_larger = np.arange(-(size[0] + p - 1) / 2.0, (size[0] + p + 1) / 2.0)
    y_list_larger = np.arange(-(size[1] + p - 1) / 2.0, (size[1] + p + 1) / 2.0)

    x_centergrid_larger, y_centergrid_larger = np.meshgrid(x_list_larger, y_list_larger)
    centers_larger = np.vstack(
        (x_centergrid_larger.ravel(), y_centergrid_larger.ravel())
    )

    # If we're not sure about how things are flipped, consider alternatives...
    if size[0] != size[1] and orientation is None:
        M_alternative = np.array([[M[0, 1], M[0, 0]], [M[1, 1], M[1, 0]]])
        M_options = [M, M_alternative]
    else:
        M_options = [M]

    results = []

    # Iterate through these alternatives.
    for M_trial in M_options:
        # Find the position of the centers for this trial matrix.
        rotated_centers = np.matmul(M_trial, centers)
        rotated_centers_larger = np.matmul(M_trial, centers_larger)

        # Make the kernel
        max_pitch = int(
            np.amax([np.linalg.norm(M_trial[:, 0]), np.linalg.norm(M_trial[:, 1])])
        )

        mask = np.zeros(
            (
                int(
                    np.amax(rotated_centers_larger[1, :])
                    - np.amin(rotated_centers_larger[1, :])
                    + max_pitch
                ),
                int(
                    np.amax(rotated_centers_larger[0, :])
                    - np.amin(rotated_centers_larger[0, :])
                    + max_pitch
                ),
            )
        )

        rotated_centers += np.flip(mask.shape)[:, np.newaxis] / 2
        rotated_centers_larger += np.flip(mask.shape)[:, np.newaxis] / 2

        # Pixels to use for the kernel.
        x_array = np.around(rotated_centers[0, :]).astype(int)
        y_array = np.around(rotated_centers[1, :]).astype(int)

        x_larger = np.around(rotated_centers_larger[0, :]).astype(int)
        y_larger = np.around(rotated_centers_larger[1, :]).astype(int)

        # Make a mask with negative power at the border, positive
        # at the array, with integrated intensity of 0.
        area = size[0] * size[1]
        perimeter = 2 * (size[0] + size[1]) + 4

        mask[y_larger, x_larger] = -area
        mask[y_array, x_array] = perimeter

        mask = _make_8bit(mask)

        # 4) Do the autocorrelation
        try:
            res = cv2.matchTemplate(img_8it, mask, cv2.TM_CCOEFF)
            _, max_val, _, max_loc = cv2.minMaxLoc(res)
        except:
            max_val = 0
            max_loc = [0, 0]

        b_fixed = (
            np.array(max_loc)[:, np.newaxis] + np.flip(mask.shape)[:, np.newaxis] / 2
        )

        # Parity check
        if orientation is None and orientation_check:
            try:
                cam_array_ind = np.ix_(
                    max_loc[1] + np.arange(mask.shape[0]),
                    max_loc[0] + np.arange(mask.shape[1]),
                )
                match = img_8it[cam_array_ind]

                wmask = 0.1
                w = np.max([1, int(wmask * max_pitch)])
                edge = np.arange(-w, w + 1)

                integration_x, integration_y = np.meshgrid(edge, edge)

                rotated_integration_x = np.around(np.add(
                    integration_x.ravel()[:, np.newaxis].T,
                    rotated_centers[:][0][:, np.newaxis],
                )).astype(int)
                rotated_integration_y = np.around(np.add(
                    integration_y.ravel()[:, np.newaxis].T,
                    rotated_centers[:][1][:, np.newaxis],
                )).astype(int)

                spotpowers = np.reshape(
                    np.sum(match[rotated_integration_y, rotated_integration_x], 1),
                    np.flip(size),
                )

                # Find the two dimmest pixels.
                spotbooleans = spotpowers <= np.sort(spotpowers.ravel())[1]

                assert np.sum(spotbooleans) == 2

                # Find whether the corners are dimmest.
                corners = spotbooleans[[-1, -1, 0, 0], [-1, 0, 0, -1]]

                assert np.sum(corners) == 1

                # We want a dim corner at -1, -1.
                rotation_parity = np.where(corners)[0][0]
                spotbooleans_rotated = np.rot90(spotbooleans, rotation_parity)

                theta = rotation_parity * np.pi / 2
                c = np.cos(theta)
                s = np.sin(theta)
                rotation = np.array([[c, -s], [s, c]])

                # Look for the second missing spot.
                flip_parity = int(spotbooleans_rotated[-1, -2]) - int(
                    spotbooleans_rotated[-2, -1]
                )

                assert abs(flip_parity) == 1

                if flip_parity == 1:
                    flip = np.array([[1, 0], [0, 1]])
                else:
                    flip = np.array([[0, 1], [1, 0]])

                M_fixed = np.matmul(M_trial, np.matmul(rotation, flip))
                parity_success = True
            except Exception as e:
                M_fixed = M_trial
                parity_success = False
        else:
            M_fixed = M_trial
            parity_success = True

        results.append((max_val, b_fixed, M_fixed, parity_success))

    if len(results) == 1:
        index = 0
    else:
        # Default to max_val if the parity check results in the same thing.
        if results[0][3] == results[1][3]:
            index = int(results[1][0] > results[0][0])
        else:
            # Index is the one that satisfied the parity check
            index = int(results[1][3])

    orientation = {"M": results[index][2], "b": results[index][1]}

    # Hone the center of our fit by averaging the positional deviations of spots.
    # We do this multiple times to allow outliers (1 std error above) to stabilize.
    # Future: Use a more physics-based psf, optimize for speed, maybe remove_field.
    hone_count = 3
    for _ in range(hone_count):
        guess_positions = np.matmul(orientation["M"], centers) + orientation["b"]

        # Calculate a point spread function (psf) integration window size.
        psf = 2 * int(np.floor(np.amin(np.amax(np.abs(orientation["M"]), axis=0))) / 2) + 1
        psf = np.max([3, psf])

        # Grab windows (sized by psf) about the guess_positions.
        regions = take(
            img, guess_positions, psf, centered=True, integrate=False, clip=True
        )

        # Get the first order moment around each of the guess windows.
        shift = image_positions(regions) - (guess_positions - np.around(guess_positions))

        # Remove outliers.
        shift_error = np.sqrt(np.square(shift[0, :]) + np.square(shift[1, :]))
        thresh = np.mean(shift_error) + np.std(shift_error)
        shift[:, shift_error > thresh] = np.nan

        # Locally fit an affine based on the measured positions.
        true_positions = guess_positions + shift
        orientation = fit_affine(centers, true_positions, orientation)

    if plot:
        array_center = orientation["b"]
        true_centers = np.matmul(orientation["M"], centers) + orientation["b"]

        showmatch = False

        fig, axs = plt.subplots(
            1, 2 + showmatch, constrained_layout=True, figsize=(12, 12)
        )

        # Determine the bounds of the zoom region, padded by 50
        x = true_centers[0, :]
        xl = [
            np.clip(np.amin(x) - max_pitch, 0, img.shape[1]),
            np.clip(np.amax(x) + max_pitch, 0, img.shape[1]),
        ]
        y = true_centers[1, :]
        yl = [
            np.clip(np.amin(y) - max_pitch, 0, img.shape[0]),
            np.clip(np.amax(y) + max_pitch, 0, img.shape[0]),
        ]

        # Plot the zoom window, with red axes.
        axs[1].imshow(img)
        axs[1].scatter(
            x, y, facecolors="none", edgecolors="r", marker="o", s=80, linewidths=0.5
        )

        for i, ctr in enumerate(true_centers.T):
            if i < 2 or i > len(true_centers.T) - 3:
                axs[1].annotate(i, (ctr[0] + 4, ctr[1] - 4), c="r", size="x-small")

        axs[1].scatter(array_center[0], array_center[1], c="r", marker="x", s=10)
        axs[1].set_title("Result - Zoom")
        axs[1].set_xlim(xl)
        axs[1].set_ylim(np.flip(yl))

        for spine in ["top", "bottom", "right", "left"]:
            axs[1].spines[spine].set_color("r")
            axs[1].spines[spine].set_linewidth(1.5)

        # Plot the non-zoom axes.
        axs[0].imshow(img_8it)
        axs[0].scatter(array_center[0], array_center[1], c="r", marker="x", s=10)

        # Plot a red rectangle to show the extents of the zoom region
        rect = plt.Rectangle(
            (float(xl[0]), float(yl[0])),
            float(np.diff(xl)), float(np.diff(yl)),
            ec="r", fc="none"
        )
        axs[0].add_patch(rect)
        axs[0].set_title("Result - Full")

        if showmatch:
            axs[2].imshow(match)
            axs[2].set_title("Result - Match")

        # Handle xy labels.
        for ax in axs[:2]:
            ax.set_xlabel("Image $x$ [pix]")
            ax.set_ylabel("Image $y$ [pix]")
        fig.tight_layout(pad=4.0)

        plt.show()

    return orientation


def _make_8bit(img):
    """
    Convert an image to ``numpy.uint8``, scaling to the limits.

    This function is useful to convert float or larger bitdepth images to
    8-bit, which :mod:`cv2` accepts and can speedily process.

    Parameters
    ----------
    img : numpy.ndarray
        The image in question.

    Returns
    -------
    ndarray
        img as an 8-bit image.
    """
    img = img.astype(float)

    img -= np.amin(img)
    img = img / np.amax(img) * (2 ** 8 - 1)

    return img.astype(np.uint8)


def get_orientation_transformation(rot="0", fliplr=False, flipud=False):
    """
    Compile a transformation lambda from simple rotates and flips.

    Useful to turn an image to an orientation which is user-friendly.
    Used by :class:`~slmsuite.hardware.cameras.camera.Camera` and subclasses.

    Parameters
    ----------
    rot : str OR int
        Rotates returned image by the corresponding degrees in ``["90", "180", "270"]``
        or :meth:`numpy.rot90` code in ``[1, 2, 3]``. Defaults to no rotation otherwise.
    fliplr : bool
        Flips returned image left right.
    flipud : bool
        Flips returned image up down.

    Returns
    -------
    function (array_like) -> numpy.ndarray
        Compiled image transformation.
    """
    transforms = list()

    if fliplr == True:
        transforms.append(np.fliplr)
    if flipud == True:
        transforms.append(np.flipud)

    if rot == "90" or rot == 1:
        transforms.append(lambda img: np.rot90(img, 1))
    elif rot == "180" or rot == 2:
        transforms.append(lambda img: np.rot90(img, 2))
    elif rot == "270" or rot == 3:
        transforms.append(lambda img: np.rot90(img, 3))

    return reduce(lambda f, g: lambda x: f(g(x)), transforms, lambda x: x)
