"""Helper functions for processing images."""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from functools import reduce
from scipy.ndimage import gaussian_filter1d as sp_gaussian_filter1d

from slmsuite.holography.lcos_toolbox import clean_2vectors

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

def take(imgs, points, size, centered=False, integrate=False, clip=False, plot=False):
    """
    Crop integration regions around an array of ``points``.
    
    Each integration region is a rectangle of the same ``size``. Similar to but more 
    general than :meth:`numpy.take`; useful for gathering data from spots in spot arrays.

    Parameters
    ----------
    imgs : array_like
        2D image or array of 2D images.
    points : array_like
        2-vector (or 2-vector array) listing location(s) of integration region(s).
        See :meth:`~slmsuite.holography.lcos_toolbox.clean_2vectors`.
    size : int or (int, int)
        Size of the rectangular integration region in ``(w,h)`` format. If a scalar is given,
        assume square ``(w,w)``.
    centered : bool
        Whether to center the integration region on the ``points``.
        If False, lower left corner is used.
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
        If ``integrate`` is ``False``, returns a list containing the images cropped
        from the regions. If ``integrate`` is ``True``, instead returns a list of floats
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
        mask =  (integration_x < 0) | (integration_x >= shape[-1]) | \
                (integration_y < 0) | (integration_y >= shape[-2])

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
        plot_taken(np.reshape(result, (points.shape[1], size[1], size[0])))

    if integrate:   # Sum over the integration axis
        return np.sum(result, -1)
    else:           # Reshape the integration axis
        return np.reshape(result, (points.shape[1], size[1], size[0]))

def plot_taken(taken):
    """
    Plots non-integrated results of :meth:`.take()` in a square array of subplots.
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

def blob_detect(img, plot=False, title="", filter_=None, **kwargs):
    """
    Detect blobs in an image.

    Default parameters are optimized for bright spot detection on dark background,
    but can be changed with **kwargs.

    Parameters
    ----------
    img : numpy.ndarray
        The image to perform blob detection on.
    filter_ : str or None
        One of ``dist_to_center`` or ``max_amp``.
    title : str
        Plot title.
    kwargs
       Extra arguments for :class:`cv2.SimpleBlobDetector`, see [0].

    Returns
    -------
    blobs : ndarray
        List of blobs found by  ``detector``.
    detector : :class:`cv2.SimpleBlobDetector`
        A blob detector with customized parameters.

    Notes
    -----
    - List of `cv2.SimpleBlobDetector` params [0].
    - Helpful but basic tutorial [1].
    [0] https://docs.opencv.org/3.4/d8/da7/structcv_1_1SimpleBlobDetector_1_1Params.html
    [1] https://learnopencv.com/blob-detection-using-opencv-python-c/
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
    if filter_ == 'dist_to_center':
        dist_to_center = [np.linalg.norm(
            np.array(blob.pt) - np.array(img.shape[::-1]) / 2) for blob in blobs]
        blobs = [blobs[np.argmin(dist_to_center)]]
    elif filter_ == 'max_amp':
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
    r"""Detect an array of spots and return orientation. Primarily used for calibration.

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
        x = 2 * (k[right, :] - k[left, :]) / \
            (blob_dist[right] + blob_dist[left])**2
        y = 2 * (k[top, :] - k[bottom, :]) / \
            (blob_dist[top] + blob_dist[bottom])**2

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

    x_centergrid_larger, y_centergrid_larger = np.meshgrid(x_list_larger, y_list_larger)
    centers_larger = np.vstack((x_centergrid_larger.ravel(), y_centergrid_larger.ravel()))

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

                rotated_integration_x = np.add(integration_x.ravel()[:, np.newaxis].T, 
                                               rotated_centers[:][0][:, np.newaxis]).astype(np.int)
                rotated_integration_y = np.add(integration_y.ravel()[:, np.newaxis].T, 
                                               rotated_centers[:][1][:, np.newaxis]).astype(np.int)

                spotpowers = np.reshape(np.sum(match[rotated_integration_y,
                                                     rotated_integration_x], 1), np.flip(size))

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
                flip_parity = int(spotbooleans_rotated[-1,-2]) - int(spotbooleans_rotated[-2,-1])

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

            # Determine the bounds of the zoom region, padded by `zoom_pad`
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

        regions = take(img, guess_positions, psf, centered=True, integrate=False, clip=True)

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

        _, axs = plt.subplots(1, 2 + showmatch, constrained_layout=True, figsize=(12, 6))

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
