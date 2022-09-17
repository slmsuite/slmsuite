"""Helper functions for processing images."""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from functools import reduce
from scipy.ndimage import gaussian_filter1d as sp_gaussian_filter1d
from scipy.optimize import curve_fit

from slmsuite.holography.toolbox import clean_2vectors
from slmsuite.misc.fitfunctions import gaussian2d_fitfun

def take(imgs, points, size, centered=False, integrate=False, clip=False, plot=False):
    """
    Crop integration regions around an array of ``points``, yielding an array of images.

    Each integration region is a rectangle of the same ``size``. Similar to but more
    general than :meth:`numpy.take`; useful for gathering data from spots in spot arrays.

    Parameters
    ----------
    imgs : array_like
        2D image or array of 2D images.
    points : array_like of floats
        2-vector (or 2-vector array) listing location(s) of integration region(s).
        See :meth:`~slmsuite.holography.toolbox.clean_2vectors`.
    size : int or (int, int)
        Size of the rectangular integration region in ``(w, h)`` format. If a scalar is given,
        assume square ``(w, w)``.
    centered : bool
        Whether to center the integration region on the ``points``.
        If False, the lower left corner is used.
    integrate : bool
        If true, the spatial dimension are integrated (summed), yielding a result of the
        same length as the number of points.
    clip : bool
        Whether to allow out-of-range integration regions. ``True`` allows regions outside
        the valid area, setting the invalid region to ``np.nan``
        (or zero if the array datatype does not support ``np.nan``).
        ``False`` throws an error upon out of range.

    Returns
    -------
    numpy.ndarray
        If ``integrate`` is ``False``, returns an array containing the images cropped
        from the regions of size `(N, h, w)`. 
        If ``integrate`` is ``True``, instead returns an array of floats of size `(N,)`
        where each float corresponds to the :meth:`numpy.sum` of a cropped image.
    """
    # Clean variables.
    if isinstance(size, int):
        size = (size, size)

    points = clean_2vectors(points)

    # Prepare helper variables. Future: consider caching for speed, if not negligible.
    edge_x = np.arange(size[0]) - (int(size[0]/2) if centered else 0)
    edge_y = np.arange(size[1]) - (int(size[1]/2) if centered else 0)

    region_x, region_y = np.meshgrid(edge_x, edge_y)

    integration_x = np.add( region_x.ravel()[:, np.newaxis].T,
                            points[:][0][:, np.newaxis]).astype(np.int)
    integration_y = np.add( region_y.ravel()[:, np.newaxis].T,
                            points[:][1][:, np.newaxis]).astype(np.int)

    shape = np.shape(imgs)

    if clip:    # Prevent out-of-range errors by clipping.
        mask = ((integration_x < 0) | (integration_x >= shape[-1]) |
                (integration_y < 0) | (integration_y >= shape[-2]))

        # Clip these indices to prevent errors.
        np.clip(integration_x, 0, shape[-1]-1, out=integration_x)
        np.clip(integration_y, 0, shape[-2]-1, out=integration_y)
    else:
        pass    # Don't prevent out-of-range errors.

    # Take the data, depending on the
    if len(shape) == 2:
        result = imgs[np.newaxis, integration_y, integration_x]
    elif len(shape) == 3:
        result = imgs[:, integration_y, integration_x]
    else:
        raise RuntimeError("Unexpected shape for imgs: {}".format(shape))

    if clip:    # Set values that were out of range to nan instead of erroring.
        try:    # If the datatype of result is incompatible with nan, set to zero instead.
            result[:, mask] = np.nan
        except:
            result[:, mask] = 0
    else:
        pass

    if plot:
        take_plot(np.reshape(result, (points.shape[1], size[1], size[0])))

    if integrate:   # Sum over the integration axis
        return np.sum(result, axis=-1)
    else:           # Reshape the integration axis
        return np.reshape(result, (points.shape[1], size[1], size[0]))

def take_plot(taken):
    """
    Plots non-integrated results of :meth:`.take()` in a square array of subplots.
    
    Parameters
    ----------
    taken : numpy.ndarray
        Array of 2D images, usually a :meth:`take()` output.
    """
    (N, sy, sx) = np.shape(taken)
    M = int(np.ceil(np.sqrt(N)))

    plt.figure(figsize=(12,12))

    sx = sx/2. - .5
    sy = sy/2. - .5
    extent = (-sx, sx, -sy, sy)

    for x in range(N):
        ax = plt.subplot(M, M, x+1)

        ax.imshow(taken[x, :, :], extent=extent)
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)

    plt.show()

def take_moment(taken, moment=(1,0), centers=(0,0), normalize=True, nansum=False):
    r"""
    Array-wise computes the given moment :math:`M_{m_xm_y}`.
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
    This function does not check if the images in ``taken`` are non-negative, or correct
    for this. Negative values may produce unusual results.

    Warning
    ~~~~~~~
    Higher order even moments (e.g. 2) will potentially yield unexpected results if
    the images are not background-subtracted. For instance, a calculation on an image
    with large background will yield the moment of the window, rather than say anything
    about the image.

    Parameters
    ----------
    taken : numpy.ndarray
        A matrix in the style of the output of :meth:`take()`, with shape `(N, wy, wx)`, where
        `(wx, wy)` is the width and height of the 2D images and :math:`N` is the number of
        images.
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
        Whether to normalize ``taken``.
        If ``False``, normalization is assumed to have been precomputed.
    nansum : bool
        Whether to use :meth:`numpy.nansum()` in place of :meth:`numpy.sum()`.
        :meth:`numpy.nansum()` treats ``nan`` values as zeros.
        This is useful in the case where ``clip=True`` is passed to :meth:`take()`
        (out of range is set to ``nan``).

    Returns
    -------
    numpy.ndarray
        The moment :math:`M_{m_xm_y}` evaluated for every image. This is of size `(N,)`
        for provided `taken` data of shape `(N, h, w)`.
    """
    (N, w_y, w_x) = taken.shape

    if len(np.shape(centers)) == 2:
        c_x = np.reshape(centers[0], (N, 1, 1))
        c_y = np.reshape(centers[1], (N, 1, 1))
    elif len(np.shape(centers)) == 1:
        c_x = centers[0]
        c_y = centers[1]

    edge_x = np.reshape(np.power(np.arange(w_x) - (w_x-1)/2., moment[0]), (1, 1, w_x)) - c_x
    edge_y = np.reshape(np.power(np.arange(w_y) - (w_y-1)/2., moment[1]), (1, w_y, 1)) - c_y

    if nansum:
        np_sum = np.nansum
    else:
        np_sum = np.sum

    if normalize:
        normalization = np_sum(taken, axis=(1,2), keepdims=False)
        reciprical = np.reciprocal(normalization, where=normalization != 0, out=np.zeros(N,))
    else:
        reciprical = 1

    if moment[1] == 0:                          # x case
        return np_sum(taken * edge_x, axis=(1,2), keepdims=False) * reciprical
    elif moment[0] == 0:                        # y case
        return np_sum(taken * edge_y, axis=(1,2), keepdims=False) * reciprical
    elif moment[1] != 0 and moment[1] != 0:     # Shear case
        return np_sum(taken * edge_x * edge_y, axis=(1,2), keepdims=False) * reciprical
    else:                                       # 0,0 (norm) case
        if normalize:
            return np.ones((N,))
        else:
            return np_sum(taken, axis=(1,2), keepdims=False)

def take_moment0(taken, nansum=False):
    """
    Array-wise 
    computes the zeroth order moments, equivalent to mass or normalization.

    Parameters
    ----------
    taken : numpy.ndarray
        Array of 2D images, usually a :meth:`take()` output.
    nansum : bool
        Whether to use :meth:`numpy.nansum()` in place of :meth:`numpy.sum()`.

    Returns
    -------
    numpy.ndarray
        The normalization factor :math:`M_{11}`.
    """
    return take_moment(taken, (0,0), normalize=False, nansum=nansum)

def take_moment1(taken, normalize=True, nansum=False):
    """
    Array-wise 
    computes the two first order moments, equivalent to position.

    Parameters
    ----------
    taken : numpy.ndarray
        Array of 2D images, usually a :meth:`take()` output.
    normalize : bool
        Whether to normalize ``taken``.
        If ``False``, normalization is assumed to have been precomputed.
    nansum : bool
        Whether to use :meth:`numpy.nansum()` in place of :meth:`numpy.sum()`.

    Returns
    -------
    numpy.ndarray
        Stack of :math:`M_{10}`, :math:`M_{01}`.
    """
    if normalize:
        taken = take_normalize(taken)

    return np.vstack(  (take_moment(taken, (1,0), normalize=False, nansum=nansum),
                        take_moment(taken, (0,1), normalize=False, nansum=nansum)) )

def take_moment2(taken, centers=None, normalize=True, nansum=False):
    r"""
    Array-wise 
    computes the three second order central moments, equivalent to variance.
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
    taken : numpy.ndarray
        Array of 2D images, usually a :meth:`take()` output.
    centers : numpy.ndarray OR None
        If the user has already computed :math:`\left<x\right>`, for example via
        :meth:`take_moment1()`, then this can be passed though ``centers``. The default
        None computes ``centers`` interally.
    normalize : bool
        Whether to normalize ``taken``.
        If ``False``, normalization is assumed to have been precomputed.
    nansum : bool
        Whether to use :meth:`numpy.nansum()` in place of :meth:`numpy.sum()`.

    Returns
    -------
    numpy.ndarray
        Stack of :math:`M_{20}`, :math:`M_{02}`, and :math:`M_{11}`.
    """
    if normalize:
        taken = take_normalize(taken)

    if centers is None:
        centers = take_moment1(taken, normalize=False, nansum=nansum)

    m20 = take_moment(taken, (2,0), centers=centers, normalize=False, nansum=nansum)
    m11 = take_moment(taken, (1,1), centers=centers, normalize=False, nansum=nansum)
    m02 = take_moment(taken, (0,2), centers=centers, normalize=False, nansum=nansum)

    return np.vstack((m20, m02, m11))

def take_moment2_circularity(moment2):
    r"""
    Given the output of :meth:`take_moment2()`, 
    return a measure of spot circularity for each moment triplet.
    The output of :meth:`take_moment2()` contains the moments :math:`M_{20}`,
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
                R(-\phi).

    We use this knowledge, along with tricks for eigenvalue calculations on 
    :math:`2 \times 2` matrices, to build up a metric for circularity:

    .. math:: \mathcal{C} = \frac{\lambda_-}{\lambda_+}.

    Notice that 

    - when :math:`\lambda_+ = \lambda_-` (isotropic scaling), the metric is unity and
    - when :math:`\lambda_- = 0` (flattened to a line), the metric is zero.
    
    Parameters
    ----------
    moment2 : numpy.ndarray
        The output of :meth:`take_moment2()`.

    Returns
    -------
    numpy.ndarray
        Array of circularities for the given moments.
    """
    m20 = moment2[0,:]
    m02 = moment2[1,:]
    m11 = moment2[2,:]

    # We can use a trick for eigenvalue calculations of 2x2 matrices to avoid
    # more complicated calculations.
    half_trace = (m20 + m02)/2
    determinant = m20 * m20 - m11 * m11

    eig_half_difference = np.sqrt(np.square(half_trace) - determinant)

    eig_plus =  half_trace + eig_half_difference
    eig_minus = half_trace - eig_half_difference

    return eig_minus / eig_plus

def take_moment2_ellipcicity_angle(moment2):
    r"""
    Given the output of :meth:`take_moment2()`, 
    return the rotation angle of the scaled basis for each moment triplet. 
    This is the angle between the :math:`x` axis and the 
    major axis (large eigenvalue axis).

    Parameters
    ----------
    moment2 : numpy.ndarray
        The output of :meth:`take_moment2()`.
    
    Returns
    -------
    numpy.ndarray
        Array of angles for the given moments.
        For highly circular spots, this angle is not meaningful, and dominated by
        experimental noise.
        For perfectly circular spots, zero is returned.
    """
    m20 = moment2[0,:]
    m02 = moment2[1,:]
    m11 = moment2[2,:]

    # Some quick math (see take_moment2_circularity).
    half_trace = (m20 + m02)/2
    determinant = m20 * m20 - m11 * m11

    eig_plus =  half_trace + np.sqrt(np.square(half_trace) - determinant)

    # We know that M * v = lambda * v. This yields a system of equations:
    #   m20 * x + m11 * y = lambda * x
    #   m11 * x + m02 * y = lambda * y
    # We're trying to solve for angle, which is just atan(x/y). We can solve for x/y:
    #   m11 * x = (lambda - m02) * y        ==>         x/y = (lambda - m02) / m11
    return np.arctan2(eig_plus - m02, m11, where=m11 != 0, out=np.zeros_like(m11))

def take_normalize(taken, nansum=False):
    """
    Array-wise 
    calculates the zeroth order moments and uses them to normalize the images.

    Parameters
    ----------
    taken : numpy.ndarray
        Array of 2D images, usually a :meth:`take()` output.

    Returns
    -------
    taken_normalized : numpy.ndarray
        A copy of ``taken``, with each image normalized.
    """
    N = taken.shape[0]
    normalization = take_moment0(taken, nansum=nansum)
    reciprical = np.reciprocal(normalization, where=normalization != 0, out=np.zeros(N,))
    return taken * np.reshape(reciprical, (N, 1, 1))

def take_fit(taken, function=gaussian2d_fitfun, guess=False):
    """
    **(Untested)** Array-wise 
    fit to a given function.

    Parameters
    ----------
    taken : numpy.ndarray
        Array of 2D images, usually a :meth:`take()` output.
    function : lambda
        Some fitfunction. Defaults to 
        :meth:`~slmsuite.misc.fitfunctions.gaussian2d_fitfun()`.
    guess : bool
        Whether to use a guess for the peak locations. Only works for the
        default ``function`` at the moment.

    Returns
    -------
    numpy.ndarray
        A matrix with the fit results. This is of shape ``(M, N)``, where ``M``
        is the number of arguments that ``function`` accepts. The slot for the
        ``xy`` points is replaced with measurements of the rsquared quality of
        the fit. Failed fits have a column filled with ``np.nan``.
    """
    (N, w_y, w_x) = taken.shape

    edge_x = np.reshape(np.arange(w_x) - (w_x-1)/2., (1, 1, w_x))
    edge_y = np.reshape(np.arange(w_y) - (w_y-1)/2., (1, w_y, 1))

    grid_x, grid_y = np.meshgrid(edge_x, edge_y)

    grid_xy = (grid_x.ravel(), grid_y.ravel())

    if guess:
        if function is gaussian2d_fitfun:
            centers = take_moment1(taken, normalize=False)
            widths =  take_moment2(taken, centers=centers, normalize=False)
        else:
            raise RuntimeError("Do not know how to parse guess for unknown function.")

    result = np.full((function.__code__.co_argcount, N), np.nan)

    for n in range(N):
        try:
            img = taken[n, :, :].ravel()

            if guess:
                if function is gaussian2d_fitfun:
                    # x0, y0, a, c, wx, wy, wxy
                    popt0 = [   
                        centers[n, 0], 
                        centers[n, 1],
                        np.amax(img) - np.amin(img), 
                        np.amin(img),
                        np.sqrt(widths[0, n]), 
                        np.sqrt(widths[1, n]), 
                        widths[2, n]
                    ]

            popt, _ = curve_fit(
                function,
                grid_xy,
                img,
                ftol=1e-5,
                p0=popt0,
            )

            ss_res = np.sum(np.square(img - function(grid_xy, *popt)))
            ss_tot = np.sum(np.square(img - np.mean(img)))
            r2 = 1 - (ss_res / ss_tot)

            result[0, n] = r2
            result[1:, n] = popt
        except BaseException:
            pass
    
    return result

def blob_detect(img, plot=False, title="", filter=None, **kwargs):
    """
    Detect blobs in an image.

    Wraps :class:`cv2.SimpleBlobDetector` [1]_. See also [2]_.
    Default parameters are optimized for bright spot detection on dark background,
    but can be changed with ``**kwargs``.

    Parameters
    ----------
    img : numpy.ndarray
        The image to perform blob detection on.
    filter : str or None
        One of ``dist_to_center`` or ``max_amp``.
    title : str
        Plot title.
    plot : bool
        Whether to show a debug plot.
    kwargs
       Extra arguments for :class:`cv2.SimpleBlobDetector`.

    Returns
    -------
    blobs : ndarray
        List of blobs found by  ``detector``.
    detector : :class:`cv2.SimpleBlobDetector`
        A blob detector with customized parameters.

    References
    ~~~~~~~~~~
    .. [1] https://docs.opencv.org/3.4/d8/da7/structcv_1_1SimpleBlobDetector_1_1Params.html
    .. [2] https://learnopencv.com/blob-detection-using-opencv-python-c/
    """
    cv2img = make8bit(np.copy(img))
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
    blobs = detector.detect(cv2img)

    if len(blobs) == 0:
        blurred = cv2.GaussianBlur(cv2img, (3, 3), 0)

        plt.imshow(blurred)
        plt.colorbar()
        plt.title('gaussian blurred')
        plt.show()

        blobs = detector.detect(blurred)
        if len(blobs) == 0:
            raise Exception("No blobs found!")

        print([blob.size for blob in blobs])

    # Downselect blobs if desired by kwargs:
    if filter == 'dist_to_center':
        dist_to_center = [np.linalg.norm(
            np.array(blob.pt) - np.array(img.shape[::-1]) / 2) for blob in blobs]
        blobs = [blobs[np.argmin(dist_to_center)]]
    elif filter == 'max_amp':
        bin_size = int(np.mean([blob.size for blob in blobs]))
        for i, blob in enumerate(blobs):
            # Try fails when blob is on edge of camera.
            try:
                blobs[i].response = float(cv2img[
                    np.ix_(int(blob.pt[1]) + np.arange(-bin_size, bin_size),
                           int(blob.pt[0]) + np.arange(-bin_size, bin_size))].sum()
                )
            except Exception:
                blobs[i].response = float(0)
        blobs = [blobs[np.argmax([blob.response for blob in blobs])]]

    if plot:
        for blob in blobs:
            cv2.circle(
                cv2img, (int(
                    blob.pt[0]), int(
                    blob.pt[1])), int(
                    blob.size * 4), (255, 0, 0), 5)
        plt.figure(dpi=300)
        plt.imshow(cv2img)
        plt.title(title)
        plt.show()

    return blobs, detector

def blob_array_detect(img, size, orientation=None, parity_check=True, plot=False):
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
    parity_check : bool
        If enabled, looks for missing spots at corners to check rotation. Used by
        :meth:`~slmsuite.hardware.cameraslms.FourierSLM.fourier_calibrate`.
    plot : bool
        Whether or not to plot debug plots. Default is ``False``.

    Returns
    --------
    dict
        Orientation dictionary with the following keys, corresponding to
        the affine transformation:

         - ``"M"`` : 2x2 ``numpy.ndarray``
         - ``"b"`` : 2x1 ``numpy.ndarray``.
    """
    cv2img = make8bit(img)
    start_orientation = orientation

    if orientation is None:
        # 1) Threshold to eliminate noise
        thresh = 10
        im_thresh = threshold(
            cv2img,
            thresh=thresh,
            thresh_type=cv2.THRESH_BINARY)

        # 2) FFT to find array pitch and orientation
        # Take the largest dimension rounded up to nearest power of 2.
        fftsize = int(2 ** np.ceil(np.log2(np.max(np.shape(img)))))
        dft = np.fft.fftshift(np.fft.fft2(im_thresh, s=[fftsize, fftsize]))
        fft_blur_size = int(2*np.ceil(fftsize/1000)) + 1 # Future: Make not arbitrary.
        # dft_amp = np.abs(dft) # cv2.GaussianBlur(np.abs(dft), (fft_blur_size, fft_blur_size), 0)
        dft_amp = cv2.GaussianBlur(np.abs(dft), (fft_blur_size, fft_blur_size), 0)

        # Need copy for some reason:
        # https://github.com/opencv/opencv/issues/18120
        minThreshold=50
        thresholdStep=10
        blobs, _ = blob_detect(dft_amp.copy(), plot=False,
            minThreshold=minThreshold, thresholdStep=thresholdStep)
        blobs = np.array(blobs)

        # Debugging plots
        if len(blobs) < 5:
            if plot:
                blobs, _ = blob_detect(
                    dft_amp.copy(), plot=True,
                    minThreshold=minThreshold, thresholdStep=thresholdStep
                )

                plt.imshow(im_thresh)
                plt.show()

                plt_img = make8bit(dft_amp.copy())

                plt.imshow(plt_img)
                plt.title('DFT peaks for scale/rotation')
                plt.colorbar()
                plt.xticks([])
                plt.yticks([])
                plt.show()

            raise RuntimeError(
                "Not enough spots found in FT, check illumination "
                "or try again with higher threshold")

        # Future: improve this part of the algorithm. It sometimes makes mistakes.

        # 2.1) Get the max point (DTF center) and its next four neighbors.
        blob_dist = np.zeros(len(blobs))
        k = np.zeros((len(blobs), 2))
        for i, blob in enumerate(blobs):
            k[i, 0] = -1 / 2 + blob.pt[0] / dft_amp.shape[1]
            k[i, 1] = -1 / 2 + blob.pt[1] / dft_amp.shape[0]
            blob_dist[i] = np.linalg.norm(
                np.array([k[i, 0], k[i, 1]]))  # Assumes max at center

        sort_ind = np.argsort(blob_dist)[:5]
        blobs = blobs[sort_ind]
        blob_dist = blob_dist[sort_ind]
        k = k[sort_ind]

        # 2.2) Calculate array metrics
        left =      np.argmin([k[:, 0]])  # Smallest x
        right =     np.argmax([k[:, 0]])  # Largest x
        bottom =    np.argmin([k[:, 1]])  # Smallest y
        top =       np.argmax([k[:, 1]])  # Largest y

        # 2.3) Calculate the vectors in the imaging domain
        x = (2 * (k[right, :] - k[left, :]) /
            (blob_dist[right] + blob_dist[left])**2)
        y = (2 * (k[top, :] - k[bottom, :]) /
            (blob_dist[top] + blob_dist[bottom])**2)

        M = np.array([[x[0], y[0]], [x[1], y[1]]])
    else:
        M = orientation["M"]

    # 3) Make the array kernel for convolutional detection of the array center.
    # Make lists that we will use to make the kernel: the array...
    x_list = np.arange(-(size[0]-1)/2., (size[0]+1)/2.)
    y_list = np.arange(-(size[1]-1)/2., (size[1]+1)/2.)

    x_centergrid, y_centergrid = np.meshgrid(x_list, y_list)
    centers = np.vstack((x_centergrid.ravel(), y_centergrid.ravel()))

    # ...and the array padded by one.
    pad = 1
    p = int(pad*2)

    x_list_larger = np.arange(-(size[0]+p-1)/2., (size[0]+p+1)/2.)
    y_list_larger = np.arange(-(size[1]+p-1)/2., (size[1]+p+1)/2.)

    x_centergrid_larger, y_centergrid_larger = np.meshgrid( x_list_larger,
                                                            y_list_larger)
    centers_larger = np.vstack((x_centergrid_larger.ravel(),
                                y_centergrid_larger.ravel()))

    # If we're not sure about how things are flipped, consider alternatives
    if size[0] != size[1] and orientation is None:
        M_alternative = np.array([[M[0,1], M[0,0]], [M[1,1], M[1,0]]])
        M_options = [M, M_alternative]
    else:
        M_options = [M]

    results = []

    # Iterate through these alternatives
    for M_trial in M_options:
        # Find the position of the centers for this trial matrix.
        rotated_centers = np.matmul(M_trial, centers)
        rotated_centers_larger = np.matmul(M_trial, centers_larger)

        # Make the kernel
        max_pitch = int(np.amax([   np.linalg.norm(M_trial[:,0]),
                                    np.linalg.norm(M_trial[:,1])]))

        mask = np.zeros((
            int(np.amax(rotated_centers_larger[1, :])
              - np.amin(rotated_centers_larger[1, :]) + max_pitch),
            int(np.amax(rotated_centers_larger[0, :])
              - np.amin(rotated_centers_larger[0, :]) + max_pitch)))

        rotated_centers +=          np.flip(mask.shape)[:, np.newaxis] / 2
        rotated_centers_larger +=   np.flip(mask.shape)[:, np.newaxis] / 2

        # Pixels to use for the kernel.
        x_array = rotated_centers[0,:].astype(np.int)
        y_array = rotated_centers[1,:].astype(np.int)

        x_larger = rotated_centers_larger[0,:].astype(np.int)
        y_larger = rotated_centers_larger[1,:].astype(np.int)

        # Make a mask with negative power at the border, positive
        # at the array, with integrated intensity of 0.
        area = size[0] * size[1]
        perimeter = 2*(size[0] + size[1]) + 4

        mask[y_larger, x_larger] = -area
        mask[y_array, x_array] = perimeter

        mask = make8bit(mask)

        # 4) Do the autocorrelation
        try:
            res = cv2.matchTemplate(cv2img, mask, cv2.TM_CCOEFF)
            _, max_val, _, max_loc = cv2.minMaxLoc(res)
        except:
            max_val = 0
            max_loc = [0,0]

        b_fixed =   np.array(max_loc)[:, np.newaxis] + \
                    np.flip(mask.shape)[:, np.newaxis] / 2

        # Parity check
        if orientation is None and parity_check:
            try:
                cam_array_ind = np.ix_( max_loc[1] + np.arange(mask.shape[0]),
                                        max_loc[0] + np.arange(mask.shape[1]))
                match = cv2img[cam_array_ind]

                wmask = .1
                w = np.max([1, int(wmask * max_pitch)])
                edge = np.arange(-w, w + 1)

                integration_x, integration_y = np.meshgrid(edge, edge)

                rotated_integration_x = np.add(
                            integration_x.ravel()[:, np.newaxis].T,
                            rotated_centers[:][0][:, np.newaxis]).astype(np.int)
                rotated_integration_y = np.add(
                            integration_y.ravel()[:, np.newaxis].T,
                            rotated_centers[:][1][:, np.newaxis]).astype(np.int)

                spotpowers = np.reshape(np.sum(match[rotated_integration_y,
                                                     rotated_integration_x], 1),
                                        np.flip(size))

                # Find the two dimmest pixels.
                spotbooleans = spotpowers <= np.sort(spotpowers.ravel())[1]

                assert np.sum(spotbooleans) == 2

                # Find whether the corners are dimmest.
                corners = spotbooleans[[-1,-1,0,0],[-1,0,0,-1]]

                assert np.sum(corners) == 1

                # We want a dim corner at -1, -1.
                rotation_parity = np.where(corners)[0][0]
                spotbooleans_rotated = np.rot90(spotbooleans, rotation_parity)

                theta = rotation_parity * np.pi / 2
                c= np.cos(theta)
                s = np.sin(theta)
                rotation = np.array([[c, -s], [s, c]])

                # Look for the second missing spot.
                flip_parity = ( int(spotbooleans_rotated[-1,-2]) -
                                int(spotbooleans_rotated[-2,-1]) )

                assert abs(flip_parity) == 1

                if  flip_parity == 1:
                    flip = np.array([[1,0], [0,1]])
                else:
                    flip = np.array([[0,1], [1,0]])

                M_fixed = np.matmul(M_trial, np.matmul(rotation, flip))
                parity_success = True
            except Exception as e:
                print(e)

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

    if plot:
        array_center = orientation["b"]
        true_centers = np.matmul(orientation["M"], centers) + orientation["b"]

        if start_orientation is None:
            _, axs = plt.subplots(1, 2, constrained_layout=True, figsize=(12, 6))

            plt_img = make8bit(dft_amp.copy())

            # Determine the bounds of the zoom region, padded by zoom_pad
            zoom_pad = 50

            x = np.array([blob.pt[0] for blob in blobs])
            xl = [  np.clip(np.amin(x) - zoom_pad,
                            0, dft_amp.shape[1]),
                    np.clip(np.amax(x) + zoom_pad,
                            0, dft_amp.shape[1])]

            y = np.array([blob.pt[1] for blob in blobs])
            yl = [  np.clip(np.amin(y) - zoom_pad,
                    0, dft_amp.shape[0]),
                    np.clip(np.amax(y) + zoom_pad,
                    0, dft_amp.shape[0])]

            axs[1].imshow(plt_img)
            axs[1].set_title('DFT Result - Zoom')
            axs[1].set_xticks([])
            axs[1].set_yticks([])
            axs[1].set_xlim(xl)
            axs[1].set_ylim(np.flip(yl))
            axs[1].scatter(x, y, facecolors='none', edgecolors='r',
                                marker='o', s=1000, linewidths=1)

            # Plot the unzoomed figure
            axs[0].imshow(plt_img)

            # Plot a red rectangle to show the extents of the zoom region
            rect = plt.Rectangle(   [xl[0], yl[0]],
                                    np.diff(xl), np.diff(yl),
                                    ec='r', fc='none')
            axs[0].add_patch(rect)
            axs[0].set_title('DFT Result - Full')
            axs[0].set_xticks([])
            axs[0].set_yticks([])

            plt.show()

    def hone():
        guess_positions = np.matmul(orientation["M"], centers) + orientation["b"]

        # Odd helper parameters.
        psf = 2*int(np.floor(np.amin(np.amax(orientation["M"], axis=0)))/2) + 1
        blur = 2*int(psf/16)+1

        regions = take( img, guess_positions, psf,
                        centered=True, integrate=False, clip=True)

        # Filter the images, but not the stack.
        sp_gaussian_filter1d(regions, blur, axis=1, output=regions)
        sp_gaussian_filter1d(regions, blur, axis=2, output=regions)

        # Future: fit gaussians instead of taking the (integer) max point for floating accuracy.
        shift_x = np.argmax(np.amax(regions, axis=1, keepdims=True), axis=2) - (psf-1)/2
        shift_y = np.argmax(np.amax(regions, axis=2, keepdims=True), axis=1) - (psf-1)/2
        shift_error = np.sqrt(np.square(shift_x) + np.square(shift_y))

        thresh = np.mean(shift_error)

        shift_x[shift_error > thresh] = np.nan
        shift_y[shift_error > thresh] = np.nan

        if False:
            fig, axs = plt.subplots(2, 2, figsize=(12,12))
            im = axs[0,0].imshow(shift_x.reshape(np.flip(size)))
            plt.colorbar(im, ax=axs[0,0])
            im = axs[1,0].imshow(shift_y.reshape(np.flip(size)))
            plt.colorbar(im, ax=axs[1,0])

            axs[0,1].hist(shift_x.ravel(), bins=30)
            axs[1,1].hist(shift_y.ravel(), bins=30)

            plt.show()

        orientation["b"] += clean_2vectors([np.nanmean(shift_x), np.nanmean(shift_y)])

    hone()
    hone()

    if plot:
        array_center = orientation["b"]
        true_centers = np.matmul(orientation["M"], centers) + orientation["b"]

        showmatch = False

        _, axs = plt.subplots(1, 2 + showmatch,
                    constrained_layout=True, figsize=(12, 6))

        # Determine the bounds of the zoom region, padded by 50
        x = true_centers[0, :]
        xl = [  np.clip(np.amin(x) - max_pitch,
                        0, img.shape[1]),
                np.clip(np.amax(x) + max_pitch,
                        0, img.shape[1])]
        y = true_centers[1, :]
        yl = [  np.clip(np.amin(y) - max_pitch,
                        0, img.shape[0]),
                np.clip(np.amax(y) + max_pitch,
                        0, img.shape[0])]

        axs[1].imshow(img)
        axs[1].scatter(x, y, facecolors='none', edgecolors='r',
                            marker='o', s=80, linewidths=.5)

        for i, ctr in enumerate(true_centers.T):
            if i < 2 or i > len(true_centers.T) - 3:
                axs[1].annotate(i, (ctr[0] + 4, ctr[1] - 4), c='r', size='x-small')

        axs[1].scatter(array_center[0], array_center[1], c='r', marker='x', s=10)
        axs[1].set_title('Result - Zoom')
        axs[1].set_xlim(xl)
        axs[1].set_ylim(np.flip(yl))

        axs[0].imshow(im_thresh)
        axs[0].scatter(array_center[0], array_center[1], c='r', marker='x', s=10)

        # Plot a red rectangle to show the extents of the zoom region
        rect = plt.Rectangle([xl[0], yl[0]], np.diff(xl),
                             np.diff(yl), ec='r', fc='none')
        axs[0].add_patch(rect)
        axs[0].set_title('Result - Full')

        if showmatch:
            axs[2].imshow(match)
            axs[2].set_title('Result - Match')

        plt.show()

    return orientation


def make8bit(img):
    """
    Convert an image to ``numpy.uint8``, scaling to the limits.

    This function is useful to convert float or larger bitdepth images to
    8-bit, which cv2 accepts and can speedily process.

    Parameters
    ----------
    img : numpy.ndarray
        The image in question.

    Returns
    -------
    ndarray
        img as an 8-bit image.
    """
    img = img.astype(np.float)

    img -= np.amin(img)
    img = img / np.amax(img) * (2**8 - 1)

    return img.astype(np.uint8)

def threshold(img, thresh=50, thresh_type=cv2.THRESH_TOZERO):
    """
    Threshold an image to a certain percentage of the maximum value.

    Parameters
    ----------
    img : numpy.ndarray
        The image in question.
    thresh : float
        Threshold in percent.
    thresh_type : int
        :mod:`cv2` threshold type.

    Returns
    -------
    numpy.ndarray
        Thresholded image.
    """
    thresh = int(thresh / 100. * np.amax(img))
    _, thresh = cv2.threshold(img, thresh, np.amax(img), thresh_type)
    return thresh

def get_transform(rot="0", fliplr=False, flipud=False):
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
    function
        Compiled image transformation.
    """
    transforms = list()

    if fliplr == True:
        transforms.append(np.fliplr)
    if flipud == True:
        transforms.append(np.flipud)

    if   rot == "90"  or rot == 1:
        transforms.append(lambda img: np.rot90(img, 1))
    elif rot == "180" or rot == 2:
        transforms.append(lambda img: np.rot90(img, 2))
    elif rot == "270" or rot == 3:
        transforms.append(lambda img: np.rot90(img, 3))

    return reduce(lambda f, g: lambda x: f(g(x)), transforms, lambda x: x)
