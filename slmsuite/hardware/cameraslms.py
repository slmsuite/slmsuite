"""
Datastructures, methods, and calibrations for an SLM monitored by a camera.
"""

import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy import optimize
from tqdm import tqdm

from slmsuite.holography import analysis
from slmsuite.holography.algorithms import SpotHologram
from slmsuite.holography.toolbox import blaze, imprint, clean_2vectors
from slmsuite.misc.files import read_h5, write_h5, generate_path, latest_path
from slmsuite.misc.fitfunctions import cos_fitfun


class CameraSLM:
    """
    Base class for an SLM with camera feedback.

    Attributes
    ----------
    cam : object
        Instance of :class:`~slmsuite.hardware.cameras.camera.Camera`
        which interfaces with a camera. This camera is
        used to provide closed-loop feedback to an SLM for calibration and holography.
    slm : object
        Instance of :class:`~slmsuite.hardware.slms.slm.SLM`
        which interfaces with a phase display.
    """

    def __init__(self, cam, slm):
        """
        Parameters
        ----------
        cam
            See :attr:`~CameraSLM.cam`.
        slm
            See :attr:`~CameraSLM.slm`.
        """
        self.cam = cam
        self.slm = slm


class NearfieldSLM(CameraSLM):
    """
    **(NotImplemented)** Class for an SLM which is not nearly in the Fourier domain of a camera.
    """

    def __init__(self, *args, **kwargs):
        """See :meth:`CameraSLM.__init__`."""
        super().__init__(*args, **kwargs)
        raise NotImplementedError()


class FourierSLM(CameraSLM):
    """
    Class for an SLM and camera separated by a Fourier transform.
    This class includes methods for Fourier space and wavefront calibration.

    Attributes
    ----------
    fourier_calibration : dict or None
        Information for the affine transformation that maps between
        the k-space of the SLM (kxy) and the pixel-space of the camera (ij).
        See :meth:`~slmsuite.hardware.cameraslms.FourierSLM.fourier_calibrate`.
        In the future, including pincushion or other distortions might be implemented
        using :meth:`scipy.ndimage.spline_filter`, though this mapping would be seperable.
    wavefront_calibration_raw : dict or None
        Raw data for wavefront calibration, which corrects for aberrations
        in the optical system (phase_correction) and measures the amplitude distribution
        of power on the SLM (measured_amplitude).
        See :meth:`~slmsuite.hardware.cameraslms.FourierSLM.wavefront_calibrate`.
    wavefront_calibration : dict or None
        Processed derivative of
        :attr:`~slmsuite.hardware.cameraslms.FourierSLM.wavefront_calibration_raw`.
        See :meth:`~slmsuite.hardware.cameraslms.FourierSLM.process_wavefront_calibration`.
    """

    def __init__(self, cam, slm):
        """
        Parameters
        ----------
        cam
            See :attr:`CameraSLM.cam`.
        slm
            See :attr:`CameraSLM.slm`.
        """
        super().__init__(cam, slm)

        self.fourier_calibration = None
        self.wavefront_calibration = None
        self.wavefront_calibration_raw = None

    ### Fourier Calibration ###

    def fourier_calibrate(
        self,
        array_shape=10,
        array_pitch=10,
        array_center=(0, 0),
        plot=True,
        autofocus=False,
        autoexposure=False,
        **kwargs
    ):
        """
        Project and fit a SLM Fourier space (``"kxy"``) grid onto 
        camera pixel space (``"ij"``) for affine fitting.
        Sets :attr:`~slmsuite.hardware.cameraslms.FourierSLM.fourier_calibration`.
        A an array produced by 
        :meth:`~slmsuite.holography.algorithms.SpotHologram.make_rectangular_array()`
        is projected for analysis by 
        :meth:`~slmsuite.holography.analysis.blob_array_detect()`.

        Parameters
        ----------
        array_shape, array_pitch, array_center
            See :meth:`~slmsuite.holography.algorithms.SpotHologram.make_rectangular_array()`.
        plot : bool
            Enables debug plots.
        autofocus : bool or dict
            Whether or not to autofocus the camera.
            If a dictionary is passed, autofocus is performed,
            and the dictionary is passed to
            :meth:`~slmsuite.hardware.cameras.camera.Camera.autofocus()`.
        autoexpose : bool or dict
            Whether or not to automatically set the camera exposure.
            If a dictionary is passed, autoexposure is performed,
            and the dictionary is passed to
            :meth:`~slmsuite.hardware.cameras.camera.Camera.autoexposure()`.
        **kwargs : dict
            Passed to :meth:`.project_fourier_grid()`.

        Returns
        -------
        dict
            :attr:`~slmsuite.hardware.cameraslms.FourierSLM.fourier_calibration`
        """
        # Parse variables
        if isinstance(array_shape, (int, float)):
            array_shape = [int(array_shape), int(array_shape)]
        if isinstance(array_pitch, (int, float)):
            array_pitch = [array_pitch, array_pitch]
            
        self.fourier_calibration = None

        # Make and project a GS hologram across a normal grid of kvecs
        hologram = self.project_fourier_grid(
            array_shape=array_shape,
            array_pitch=array_pitch,
            array_center=array_center,
            **kwargs
        )

        if plot:
            hologram.plot_farfield()
            hologram.plot_nearfield()

        # Optional step -- autofocus and autoexpose the spots
        if autofocus or isinstance(autofocus, dict):
            if autoexposure or isinstance(autoexposure, dict):
                if isinstance(autoexposure, dict):
                    self.cam.autoexposure(**autoexposure)
                else:
                    self.cam.autoexposure()

            if isinstance(autofocus, dict):
                self.cam.autofocus(plot=plot, **autofocus)
            else:
                self.cam.autofocus(plot=plot)

        if autoexposure or isinstance(autoexposure, dict):
            if isinstance(autoexposure, dict):
                self.cam.autoexposure(**autoexposure)
            else:
                self.cam.autoexposure()

        self.cam.flush()
        img = self.cam.get_image()

        # 2) Get orientation of projected array
        orientation = analysis.blob_array_detect(img, array_shape, plot=plot)

        a = clean_2vectors(array_center)
        M = np.array(orientation["M"])
        b = clean_2vectors(orientation["b"])

        knm_conv = (
            np.array((self.slm.dx, self.slm.dy))
            * np.flip(np.squeeze(hologram.shape))
            / np.squeeze(array_pitch)
        )

        M = np.array(
            [
                [M[0, 0] * knm_conv[0], M[0, 1] * knm_conv[1]],
                [M[1, 0] * knm_conv[0], M[1, 1] * knm_conv[1]],
            ]
        )

        self.fourier_calibration = {"M": M, "b": b, "a": a}

        return self.fourier_calibration

    def name_fourier_calibration(self):
        """
        Creates ``"<cam.name>-<slm.name>-fourier-calibration"``.

        Returns
        -------
        name : str
            The generated name.
        """
        return "{}-{}-fourier-calibration".format(self.cam.name, self.slm.name)

    def save_fourier_calibration(self, path=".", name=None):
        """
        Saves :attr:`~slmsuite.hardware.cameraslms.FourierSLM.fourier_calibration`
        to a file like ``"path/name_id.h5"``.

        Parameters
        ----------
        path : str
            Path to directory to save in. Default is current directory.
        name : str or None
            Name of the save file. If `None`, will use :meth:`name_fourier_calibration`.

        Returns
        -------
        str
            The file path that the fourier calibration was saved to.
        """
        if name is None:
            name = self.name_fourier_calibration()
        file_path = generate_path(path, name, extension="h5")
        write_h5(file_path, self.fourier_calibration)

        return file_path

    def load_fourier_calibration(self, file_path=None):
        """
        Loads :attr:`~slmsuite.hardware.cameraslms.FourierSLM.fourier_calibration`
        from a file.

        Parameters
        ----------
        file_path : str or None
            Full path to the Fourier calibration file. If `None`, will
            search the current directory for a file with a name like
            the one returned by :meth:`name_fourier_calibration`.

        Returns
        -------
        str
            The file path that the fourier calibration was loaded from.

        Raises
        ------
        FileNotFoundError
            If a file is not found.
        """
        if file_path is None:
            path = os.path.abspath(".")
            name = self.name_fourier_calibration()
            file_path = latest_path(path, name, extension="h5")
            if file_path is None:
                raise FileNotFoundError(
                    "Unable to find a Fourier calibration file like\n{}"
                    "".format(os.path.join(path, name))
                )

        self.fourier_calibration = read_h5(file_path)

        return file_path

    def project_fourier_grid(
        self, array_shape=10, array_pitch=10, array_center=(0, 0), **kwargs
    ):
        """
        Projects a Fourier space grid onto pixel space.

        Parameters
        ----------
        array_shape, array_pitch, array_center
            Passed to :meth:`~slmsuite.holography.algorithms.SpotHologram.make_rectangular_array()`.
        **kwargs
            Passed to :meth:`~slmsuite.holography.algorithms.SpotHologram.optimize()`.

        Returns
        -------
        ~slmsuite.holography.algorithms.SpotHologram
            Optimized hologram.
        """

        # Make the spot array
        shape = SpotHologram.calculate_padding(
            self, padding_order=1, square_padding=True
        )
        hologram = SpotHologram.make_rectangular_array(
            shape,
            array_shape=array_shape,
            array_pitch=array_pitch,
            array_center=array_center,
            basis="knm",
            parity_check=True,
            cameraslm=self,
        )

        # Default optimize settings.
        if "maxiter" not in kwargs:
            kwargs["maxiter"] = 50

        # Optimize and project the hologram
        hologram.optimize(**kwargs)

        self.slm.write(hologram.extract_phase())

        return hologram

    ### Fourier Calibration Helpers ###

    def kxyslm_to_ijcam(self, kxy):
        r"""
        Converts SLM Fourier space (``"kxy"``) to camera pixel space (``"ij"``).
        For blaze vectors :math:`\vec{x}` and camera pixel indices :math:`\vec{y}`, computes:

        .. math:: \vec{y} = M \cdot (\vec{x} - \vec{a}) + \vec{b}

        where :math:`M`, :math:`\vec{b}`, and :math:`\vec{a}` are stored in 
        :attr:`~slmsuite.hardware.cameraslms.FourierSLM.fourier_calibration`.

        Parameters
        ----------
        kxy : array_like
            2-vector or array of 2-vectors to convert.
            Cleaned with :meth:`~slmsuite.holography.toolbox.clean_2vectors()`.

        Returns
        -------
        ij : numpy.ndarray
            2-vector or array of 2-vectors in camera coordinates.

        Raises
        ------
        AssertionError
            If the fourier plane calibration does not exist.
        """
        assert self.fourier_calibration is not None
        return np.matmul(   self.fourier_calibration["M"],
                            clean_2vectors(kxy) - self.fourier_calibration["a"] ) \
                    + self.fourier_calibration["b"]

    def ijcam_to_kxyslm(self, ij):
        r"""
        Converts camera pixel space (``"ij"``) to SLM Fourier space (``"kxy"``).
        For camera pixel indices :math:`\vec{y}` and blaze vectors :math:`\vec{x}`, computes:

        .. math:: \vec{x} = M^{-1} \cdot (\vec{y} - \vec{b}) + \vec{a}

        where :math:`M`, :math:`\vec{b}`, and :math:`\vec{a}` are stored in 
        :attr:`~slmsuite.hardware.cameraslms.FourierSLM.fourier_calibration`.

        Parameters
        ----------
        ij : array_like
            2-vector or array of 2-vectors to convert.
            Cleaned with :meth:`~slmsuite.holography.toolbox.clean_2vectors()`.

        Returns
        -------
        kxy : numpy.ndarray
            2-vector or array of 2-vectors in slm coordinates.

        Raises
        ------
        AssertionError
            If the fourier plane calibration does not exist.
        """
        assert self.fourier_calibration is not None
        return np.matmul(   np.linalg.inv(self.fourier_calibration["M"]),
                            clean_2vectors(ij) - self.fourier_calibration["b"]  ) \
                    + self.fourier_calibration["a"]

    def calc_spot_size(self, size, basis="kxy"):
        """
        Calculates the spot size in the given basis for a given SLM patch of ``size``
        SLM pixels. Uses
        :attr:`~slmsuite.hardware.cameraslms.FourierSLM.fourier_calibration`.

        Parameters
        ----------
        size : (float, float) or int or float
            Size on SLM. An scalar is taken as the width and height of a square.
        basis : {"kxy", "ij"}
            Basis of size; ``"kxy"`` for SLM size, ``"ij"`` for camera size.

        Returns
        -------
        (float, float)
            Size in x and y of the spot in camera pixels.

        Raises
        ------
        ValueError
            If the basis argument was malformed.
        """
        if isinstance(size, (int, float)):
            size = (size, size)

        size_kxy = (1 / self.slm.dx / size[0], 1 / self.slm.dy / size[1])

        ret = None
        if basis == "kxy":
            ret = size_kxy
        elif basis == "ij":
            ret = np.abs(self.kxyslm_to_ijcam([0, 0]) - self.kxyslm_to_ijcam(size_kxy))
        else:
            raise ValueError("Unrecgonized basis \"{}\".".format(basis))

        return ret

    ### Wavefront Calibration ###

    def wavefront_calibrate(
        self,
        interference_point,
        field_point,
        superpixel_size=50,
        phase_steps=10,
        exclude_superpixels=(0, 0),
        autoexposure=False,
        test_superpixel=None,
        reference_superpixel=None,
        plot=0,
    ):
        """
        Perform wavefront calibration.
        This procedure involves iteratively interfering light diffracted from
        superpixels across an SLM with a reference superpixel  [1]_. Interference
        occurs at a given ``interference_point`` in the camera's imaging plane.
        It is at this point where the computed correction is ideal; the further away
        from this point, the less ideal the correction is.
        Sets :attr:`~slmsuite.hardware.cameraslms.FourierSLM.wavefront_calibration_raw`.
        Run :meth:`~slmsuite.hardware.cameraslms.FourierSLM.process_wavefront_calibration`
        after to produce the usable
        :attr:`~slmsuite.hardware.cameraslms.FourierSLM.wavefront_calibration`.

        References
        ----------
        .. [1] In situ wavefront correction and its application to micromanipulation

        Parameters
        ----------
        interference_point : (float, float)
            Position in the camera domain where interference occurs.
        field_point : (float, float)
            Position in the camera domain where pixels not included in superpixels are
            blazed toward in order to reduce light in the camera's field. Suggested
            approach is to set this outside the field of view of the camera and make
            sure that other diffraction orders are far from the `interference_point`.
        superpixel_size : int
            The width and height in pixels of each SLM superpixel.
            If this is not a devisor of both dimensions of the SLM's :attr:`shape`,
            then superpixels at the edge of the SLM may be cropped and give undefined results.
        phase_steps : int
            The number of phases measured for the interference pattern.
        exclude_superpixels : (int, int)
            Optionally exclude superpixels from the margin, in ``(nx, ny)`` form. As power is
            typically concentrated in the center of the SLM, this function is useful for
            excluding points that are known to be blocked, or for quickly testing calibration
            at the most relevant points.
        autoexposure : bool
            Whether or not to perform autoexposure on the camera.
            See :meth:`~slmsuite.hardware.cameras.camera.Camera.autoexposure`.
        test_superpixel : (int, int) OR None
            Test an iteration of wavefront calibration using the given superpixel.
            If ``None``, do not test.
        reference_superpixel : None or (int, int)
            The superpixel to reference from. Defaults to the center of the SLM.
        plot : int or bool
            Whether to provide visual feedback, options are:

            -1
              No plots or tqdm prints.
            0 / ``False``
               No plots, but tqdm prints.
            1 / ``True``
               Plots on fits and essentials.
            2
               Plots on everything.

        Returns
        -------
        dict
            :attr:`~slmsuite.hardware.cameraslms.FourierSLM.wavefront_calibration_raw`

        Raises
        ------
        AssertionError
            If the fourier plane calibration does not exist.
        """
        # Interpret the plot command.
        verbose = plot >= 0
        plot_fits = plot >= 1 or test_superpixel is not None
        plot_everything = plot >= 2

        # Clean the points
        base_point = self.kxyslm_to_ijcam([0, 0]).astype(np.int)
        interference_point = clean_2vectors(interference_point).astype(np.int)
        field_point = clean_2vectors(field_point).astype(np.int)

        # Use the Fourier calibration to help find points/sizes in the imaging plane.
        assert self.fourier_calibration is not None, \
            "Fourier calibration must be done before wavefront calibration."
        interference_blaze = self.ijcam_to_kxyslm(interference_point)
        field_blaze = self.ijcam_to_kxyslm(field_point)

        # Determine how many rows and columns of superpixels we will use.
        [NY, NX] = np.ceil(np.array(self.slm.shape) / superpixel_size).astype(np.int)

        if reference_superpixel is None:
            # Set the reference superpixel to be centered on the SLM.
            [nxref, nyref] = np.floor(  np.flip(self.slm.shape) \
                                        / superpixel_size / 2).astype(np.int)
        else:
            (nxref, nyref) = reference_superpixel

        interference_size = np.array(self.calc_spot_size(   superpixel_size, 
                                                            basis="ij")).astype(np.int)

        correction_dict = {
            "NX": NX,
            "NY": NY,
            "nxref": nxref,
            "nyref": nyref,
            "superpixel_size": superpixel_size,
            "interference_point": interference_point,
            "interference_size": interference_size,
        }

        keys = [
            "power",
            "normalization",
            "background",
            "phase",
            "kx",
            "ky",
            "amp_fit",
            "contrast_fit",
            "r2_fit",
        ]

        for key in keys:
            if key not in correction_dict.keys():
                correction_dict.update({key: np.zeros((NY, NX), dtype=np.float32)})

        def superpixels(index,
                        reference=None,
                        reference_blaze=interference_blaze,
                        target=None,
                        target_blaze=interference_blaze,
                        plot=False):
            """
            Helper function for making superpixel phase masks.

            Parameters
            ----------
            reference, target : float or None
                Phase of reference/target superpixel; not rendered if None.
            reference_blaze, target_blaze : (float, float)
                Blaze vector for the given superpixel.
            """
            matrix = blaze(self.slm, field_blaze)

            if reference is not None:
                imprint(matrix,
                        np.array([nxref, 1, nyref, 1]) * superpixel_size,
                        self.slm,
                        blaze,
                        vector=reference_blaze,
                        offset=0)

            if target is not None:
                imprint(matrix,
                        np.array([index[0], 1, index[1], 1]) * superpixel_size,
                        self.slm,
                        blaze,
                        vector=target_blaze,
                        offset=target)

            if plot_everything or plot:
                plt.figure(figsize=(20, 25))
                plt.imshow(np.mod(matrix, 2 * np.pi), interpolation="none")
                plt.show()

            return matrix

        def mask(img, center, lengths):
            """
            Take a matrix img and CUT everything outside a rectangle, defined by
            its center point and the lengths

            Parameters
            ----------
            img : numpy.ndarray
                Matrix to mask.
            center : (int, int)
                Center (x, y) of rectangle.
            lengths : (int, int)
                (length_x, length_y) of rectangle.

            Returns
            -------
            numpy.ndarray
                Masked image.
            """
            return img[
                int(center[1] - lengths[1] / 2) : int(center[1] + lengths[1] / 2),
                int(center[0] - lengths[0] / 2) : int(center[0] + lengths[0] / 2),
            ]

        def fit_phase(phases, intensities):
            """
            Fits a sine function to the Intensity Vs. phase, and extracts best phase and amplitude
            that give the constructive interference.
            If fit fails return 0 on all values

            Parameters
            ----------
            phases : numpy.ndarray
                Phase measurements.
            intensities : numpy.ndarray
                Intensity measurements.

            Returns
            -------
            best_phase :
                b
            amp :
                a
            r2 :
                r^2 of fit
            contrast :
                a / (a + c)
            """
            guess = [
                np.max(intensities) - np.min(intensities),
                phases[np.argmax(intensities)],
                np.min(intensities),
            ]

            try:
                p, _ = optimize.curve_fit(cos_fitfun, phases, intensities, p0=guess)
            except BaseException:
                return 0, 0, 0, 0

            # extract phase and amplitude from fit
            best_phase = p[1]
            amp = p[0]
            contrast = p[0] / (p[0] + p[2])

            # residual and total sum of squares, producing the r^2 metric.
            ss_res = np.sum((intensities - cos_fitfun(phases, *p)) ** 2)
            ss_tot = np.sum((intensities - np.mean(intensities)) ** 2)
            r2 = 1 - (ss_res / ss_tot)

            if plot_fits:
                plt.scatter(phases / np.pi, intensities, label="Data")

                phases_fine = np.linspace(0, 2 * np.pi, 100)

                plt.plot(phases_fine / np.pi, cos_fitfun(phases_fine, *p), label="Fit")
                plt.plot(
                    phases_fine / np.pi, cos_fitfun(phases_fine, *guess), label="Guess"
                )
                plt.plot(best_phase / np.pi, p[0] + p[2], "xr", label="Phase")

                plt.legend(loc="best")
                plt.title("Interference (r^2={:.2f})".format(r2))
                plt.grid()
                plt.xlabel(r"$\phi$ $[\pi]$")
                plt.ylabel("Signal")

                plt.show()

            return best_phase, amp, r2, contrast

        def plot_labeled(img, plot=False, title=""):
            if plot_everything or plot:
                _, ax = plt.subplots(1, 1)
                ax.imshow(np.log10(img + 1), cmap="Blues")

                dpoint = field_point - base_point

                points = [(base_point + N * dpoint) for N in range(-2, 3)]
                points.append(interference_point)
                labels = ["-2nd", "-1st", "0th", "1st", "2nd", "Target"]

                wh = int(interference_size[0])
                hh = int(interference_size[1])

                for point, label in zip(points, labels):
                    if label == "Target":
                        wh *= 2
                        hh *= 2
                    rect = plt.Rectangle(
                        [point[0] - wh, point[1] - hh],
                        2 * wh,
                        2 * hh,
                        ec="r",
                        fc="none",
                    )
                    ax.add_patch(rect)
                    ax.annotate(
                        label,
                        (point[0], point[1] + 2 * interference_size[1] + hh),
                        c="r",
                        size="x-small",
                        ha="center",
                    )

                plt.title(title)

                plt.show()

        def find_center(img, plot=False):
            masked_pic_mode = mask(img, interference_point, 8 * interference_size)

            if plot_everything or plot:
                plt.imshow(masked_pic_mode, cmap="Blues")
                plt.show()

            # Blur a lot and assume the maximum corresponds to the center.
            blur = 2 * int(np.min(interference_size)) + 1
            masked_pic_mode = analysis.make8bit(masked_pic_mode)
            masked_pic_mode = cv2.GaussianBlur(masked_pic_mode, (blur, blur), 0)
            _, _, _, max_loc = cv2.minMaxLoc(masked_pic_mode)
            found_center = (
                clean_2vectors(max_loc)
                - clean_2vectors(np.flip(masked_pic_mode.shape)) / 2
                + interference_point
            )

            return found_center

        def measure(index, plot=False):
            self.cam.flush()

            # Step 0: Measure the background.
            self.slm.write( superpixels(index, reference=None, target=None), 
                            wait_for_settle=True    )
            background_image = self.cam.get_image()
            plot_labeled(background_image, plot=plot)
            back = mask(background_image, interference_point, 2 * interference_size).sum()

            # Step 0.5: Measure the power in the reference mode.
            self.slm.write( superpixels(index, reference=0, target=None), 
                            wait_for_settle=True    )
            normalization_image = self.cam.get_image()
            plot_labeled(normalization_image, plot=plot)
            norm = mask(normalization_image, interference_point, 2 * interference_size).sum()

            # Step 1: Add a blaze to the target mode so that it overlaps with
            # reference mode.
            self.slm.write( superpixels(index, reference=None, target=0), 
                            wait_for_settle=True    )
            position_image = self.cam.get_image()
            plot_labeled(position_image, plot=plot)
            found_center = find_center(position_image)

            blaze_difference = self.ijcam_to_kxyslm(found_center) - interference_blaze
            target_blaze_fixed = interference_blaze - blaze_difference

            # Step 1.5: Measure the power in the corrected target mode.
            self.slm.write( superpixels(index, reference=None, target=0, 
                                        target_blaze=target_blaze_fixed),
                            wait_for_settle=True)
            fixed_image = self.cam.get_image()
            plot_labeled(fixed_image, plot=plot)
            pwr = mask(fixed_image, interference_point, 2 * interference_size).sum()

            # Step 2: Measure interference and find relative phase
            phases = np.linspace(0, 2 * np.pi, phase_steps)
            results = []  # list for recording the intensity of the reference point

            # Determine whether to use a progress bar.
            if verbose:
                description = "nx={}, ny={}".format(index[0], index[1])
                prange = tqdm(phases, position=0, leave=False, desc=description)
            else:
                prange = phases

            # Step 3: Measure phase
            for phase in prange:
                self.slm.write(
                    superpixels(
                        index,
                        reference=0,
                        target=phase,
                        target_blaze=target_blaze_fixed,
                    ),
                    flatmap=True,
                    wait_for_settle=True,
                )
                interference_image = self.cam.get_image()
                results.append(
                    interference_image[
                        int(interference_point[1]), int(interference_point[0])
                    ]
                )

            plot_labeled(interference_image, plot=plot)

            # Step 4: Fit to sine and return.
            phase_fit, amp_fit, r2_fit, contrast_fit = fit_phase(phases, results)

            return {
                "power": pwr,
                "normalization": norm,
                "background": back,
                "phase": phase_fit,
                "kx": -blaze_difference[0],
                "ky": -blaze_difference[1],
                "amp_fit": amp_fit,
                "contrast_fit": contrast_fit,
                "r2_fit": r2_fit,
            }

        # Correct exposure and position of the reference mode.
        self.slm.write( superpixels((0, 0), reference=0, target=None), 
                        wait_for_settle=True)
        self.cam.flush()

        if autoexposure:
            window = [  interference_point[0],
                        2 * interference_size[0],
                        interference_point[1],
                        2 * interference_size[1]    ]
            self.cam.autoexposure(set_fraction=0.1, window=window)

        base_image = self.cam.get_image()
        plot_labeled(base_image, plot=plot_fits)
        found_center = find_center(base_image)

        # Correct the original blaze using the measured result.
        blaze_difference = self.ijcam_to_kxyslm(found_center) - interference_blaze
        reference_blaze_fixed = interference_blaze - blaze_difference

        if plot_fits:
            self.slm.write(
                superpixels(
                    (0, 0),
                    reference=0,
                    target=None,
                    reference_blaze=reference_blaze_fixed),
                wait_for_settle=True)
            fixed_image = self.cam.get_image()
            plot_labeled(fixed_image, plot=plot_fits)
            found_center = find_center(fixed_image)

        # If we just want to debug/test one region, then do so.
        if test_superpixel is not None:
            return measure(test_superpixel, plot=plot_fits)

        # Otherwise, proceed with all of the superpixels.
        for n in tqdm(range(NX * NY), position=0, leave=False):
            nx = int(n % NX)
            ny = int(n / NX)

            # Exclude the reference mode.
            if nx == nxref and ny == nyref:
                continue

            # Exclude margin superpixels, if desired.
            if nx < exclude_superpixels[0]:
                continue
            if nx > self.slm.shape[1] - exclude_superpixels[0]:
                continue
            if ny < exclude_superpixels[1]:
                continue
            if ny > self.slm.shape[0] - exclude_superpixels[1]:
                continue

            # Measure!
            measurement = measure((nx, ny))

            # Update dictionary.
            for key in measurement:
                correction_dict[key][ny, nx] = measurement[key]

        self.wavefront_calibration_raw = correction_dict

        return correction_dict

    def process_wavefront_calibration(self, smooth=True, r2_thresh=0.99, plot=True):
        """
        Processes :attr:`~slmsuite.hardware.cameraslms.FourierSLM.wavefront_calibration_raw`
        into the desired phase correction and amplitude measurement. Sets
        :attr:`~slmsuite.hardware.cameraslms.FourierSLM.wavefront_calibration`.

        Parameters
        ----------
        smooth : bool
            Whether to blur the correction data to avoid aliasing.
        r2_thresh : float
            Threshold for a "good fit". Proxy for whether a datapoint should be used or
            ignored in the final data, depending upon the rsquared value of the fit.
            Should be within [0, 1].
        plot : bool
            Whether to enable debug plots.

        Returns
        -------
        dict
            :attr:`~slmsuite.hardware.cameraslms.FourierSLM.wavefront_calibration`.
        """
        data = self.wavefront_calibration_raw

        NX = data["NX"]
        NY = data["NY"]
        nxref = data["nxref"]
        nyref = data["nyref"]

        size_blur_k = 1

        def average_neighbors(matrix):
            matrix[nyref, nxref] = (
                  np.sum([matrix[nyref + i, nxref + 1] for i in [-1, 0, 1]])
                + np.sum([matrix[nyref + i, nxref]     for i in [-1,    1]])
                + np.sum([matrix[nyref + i, nxref - 1] for i in [-1, 0, 1]])
            ) / 8

        # def average_neighbors_phase(matrix):
        #     matrix[nyref, nxref] = (
        #           np.sum([matrix[nyref, nxref + i] for i in [-1, 1]])
        #         + np.sum([matrix[nyref + i, nxref] for i in [-1, 1]])
        #     )

        # Load the amplitude and norm
        # Fix the reference pixel by averaging the 8 surrounding pixels
        pwr = np.copy(data["power"])
        pwr[pwr == np.inf] = np.amax(pwr)
        average_neighbors(pwr)
        if smooth:
            pwr = cv2.GaussianBlur(pwr, (size_blur_k, size_blur_k), 0)

        norm = np.copy(data["normalization"])
        average_neighbors(norm)
        if smooth:
            norm = cv2.GaussianBlur(norm, (size_blur_k, size_blur_k), 0)

        back = np.copy(data["background"])
        average_neighbors(back)
        if smooth:
            back = cv2.GaussianBlur(back, (size_blur_k, size_blur_k), 0)

        pwr -= back
        norm -= back

        # Normalize and resize
        superpixel_size = int(data["superpixel_size"])
        w = superpixel_size * NX
        h = superpixel_size * NY

        pwr_norm = np.divide(pwr, norm)

        pwr_norm[np.isnan(pwr_norm)] = 0
        pwr_norm[~np.isfinite(pwr_norm)] = 0
        pwr_norm[pwr_norm < 0] = 0

        pwr_large = cv2.resize(pwr_norm, (w, h), interpolation=cv2.INTER_CUBIC)
        pwr_large = pwr_large[:self.slm.shape[0], :self.slm.shape[1]]

        pwr_large[np.isnan(pwr_large)] = 0
        pwr_large[~np.isfinite(pwr_large)] = 0
        pwr_large[pwr_large < 0] = 0

        if smooth:
            size_blur = 4 * superpixel_size + 1
            pwr_large = cv2.GaussianBlur(pwr_large, (size_blur, size_blur), 0)

        amp = np.sqrt(pwr_norm)
        amp_large = np.sqrt(pwr_large)

        # Process r^2
        r2 = np.copy(data["r2_fit"])
        r2[nyref, nxref] = 1
        r2s = cv2.GaussianBlur(r2, (3, 3), 0)

        r2s_large = cv2.resize(r2s, (w, h), interpolation=cv2.INTER_NEAREST)
        r2s_large = r2s_large[:self.slm.shape[0], :self.slm.shape[1]]

        # Process the wavefront
        kx = np.copy(data["kx"])
        ky = np.copy(data["ky"])

        average_neighbors(kx)
        average_neighbors(ky)

        offset = np.copy(data["phase"])

        real = np.cos(offset)
        imag = np.sin(offset)
        
        average_neighbors(real)
        average_neighbors(imag)

        offset = np.arctan2(imag, real) + np.pi

        kx[r2s < r2_thresh] = 0
        ky[r2s < r2_thresh] = 0
        offset[r2s < r2_thresh] = 0
        pathing = 0 * r2s

        for ny in range(NY):
            for nx in list(range(NX)) + list(range(NX-1, -1, -1)):
                if r2s[ny, nx] >= r2_thresh:
                    pass
                else:
                    kx2 = []
                    ky2 = []
                    offset2 = []

                    for (ax, ay) in [(1, 0), (-1, 0), (0, 1), (0, -1), 
                                     (1, -1), (-1, -1), (1, 1), (-1, 1)]:
                        (tx, ty) = (nx + ax, ny + ay)
                        (dx, dy) = (2 * np.pi * (nx - nxref) * superpixel_size * self.slm.dx, 
                                    2 * np.pi * (ny - nyref) * superpixel_size * self.slm.dy)

                        if (tx >= 0 and tx < NX and ty >= 0 and ty < NY and
                            (r2s[ty, tx] >= r2_thresh or pathing[ty, tx] == ny)):

                            kx3 = kx[ty, tx]
                            ky3 = ky[ty, tx]

                            kx2.append(kx3)
                            ky2.append(ky3)
                            offset2.append(offset[ty, tx] + dx * kx3 + dy * ky3) # + kx3 * ax + ky3 * ay)

                    if len(kx2) > 0:
                        kx[ny, nx] = 0 #np.mean(kx2)
                        ky[ny, nx] = 0 #np.mean(ky2)

                        minstd = np.inf
                        for phi in range(4):
                            shift = phi * np.pi / 2
                            offset3 = np.mod(np.array(offset2) + shift, 2 * np.pi)
                            
                            if minstd > np.std(offset3):
                                minstd = np.std(offset3)
                                offset[ny, nx] = np.mod(np.mean(offset3) - shift, 2 * np.pi)

                        pathing[ny, nx] = ny

        phase = np.zeros(self.slm.shape)
        for nx in range(NX):
            for ny in range(NY):
                imprint(
                    phase,
                    np.array([nx, 1, ny, 1]) * superpixel_size,
                    self.slm,
                    blaze,
                    vector=(kx[ny, nx], ky[ny, nx]),
                    offset=offset[ny, nx],
                )

        real = np.cos(phase)
        imag = np.sin(phase)

        # Blur the phase to smooth it out
        if smooth:
            real = cv2.GaussianBlur(real, (size_blur, size_blur), 0)
            imag = cv2.GaussianBlur(imag, (size_blur, size_blur), 0)

        phase = np.arctan2(imag, real) + np.pi
        
        mindiff = np.inf
        phase_fin = []
        for phi in range(8):
            shift = phi * np.pi / 4

            modthis = np.mod(phase + shift, 2 * np.pi)

            fom = np.sum(np.abs(np.diff(modthis, axis=0))) + \
                  np.sum(np.abs(np.diff(modthis, axis=1)))

            if fom < mindiff:
                phase_fin = modthis
                min = np.amin(phase_fin)
                mean = np.mean(phase_fin)
                max = np.amax(phase_fin)
                if mean - min < max - mean:
                    phase_fin -= min
                else:
                    phase_fin -= max - 2 * np.pi

                phase_fin = np.mod(phase_fin, 2 * np.pi)

                mindiff = fom

        self.wavefront_calibration = {  "phase_correction":phase_fin, 
                                        "measured_amplitude":amp_large}

        # Load the correction to the SLM
        self.slm.phase_correction = phase_fin
        self.slm.measured_amplitude = amp_large

        # Plot the result
        if plot:
            plt.figure(figsize=(16, 8))

            plt.subplot(1, 3, 1)
            plt.imshow(
                phase_fin,
                # cmap=plt.get_cmap("twilight"),
                interpolation="none",
            )
            plt.title("SLM Flatfield Phase Correction")
            plt.xlabel("SLM $x$ [pix]")
            plt.ylabel("SLM $y$ [pix]")

            plt.subplot(1, 3, 2)
            plt.imshow(amp_large)
            plt.title("Measured Beam Amplitude")
            plt.xlabel("SLM $x$ [pix]")
            plt.ylabel("SLM $y$ [pix]")

            plt.subplot(1, 3, 3)
            plt.imshow(r2)
            plt.title("r^2")
            plt.xlabel("SLM $x$ [superpix]")
            plt.ylabel("SLM $y$ [superpix]")

            plt.show()

        return self.wavefront_calibration

    def name_wavefront_calibration(self):
        """
        Creates ``"<cam.name>-<slm.name>-wavefront-calibration"``.

        Returns
        -------
        name : str
            The generated name.
        """
        return "{}-{}-wavefront-calibration".format(self.cam.name, self.slm.name)

    def save_wavefront_calibration(self, path=".", name=None):
        """
        Saves :attr:`~slmsuite.hardware.cameraslms.FourierSLM.wavefront_calibration_raw`
        to a file like ``"path/name_id.h5"``.

        Parameters
        ----------
        path : str
            Path to the save location. Default is current directory.
        name : str or None
            Name of the save file. If `None`, will use :meth:`name_wavefront_calibration`.

        Returns
        -------
        str
            The file path that the fourier calibration was saved to.
        """
        if name is None:
            name = self.name_wavefront_calibration()
        file_path = generate_path(path, name, extension="h5")
        write_h5(file_path, self.wavefront_calibration_raw)

        return file_path

    def load_wavefront_calibration(self, file_path=None, process=True):
        """
        Loads :attr:`~slmsuite.hardware.cameraslms.FourierSLM.wavefront_calibration_raw`
        from a file.

        Parameters
        ----------
        file_path : str or None
            Full path to the wavefront calibration file. If `None`, will
            search the current directory for a file with a name like
            the one returned by
            :meth:`~slmsuite.hardware.cameraslms.FourierSLM.name_wavefront_calibration`.
        process : bool
            Whether to immediately process the wavefront calibration.
            See
            :meth:`~slmsuite.hardware.cameraslms.FourierSLM.process_wavefront_calibration`.

        Returns
        -------
        str
            The file path that the wavefront calibration was loaded from.
        """
        if file_path is None:
            path = os.path.abspath(".")
            name = self.name_wavefront_calibration()
            file_path = latest_path(path, name, extension="h5")
            if file_path is None:
                raise FileNotFoundError(
                    "Unable to find a Fourier calibration file like\n{}"
                    "".format(os.path.join(path, name))
                )

        self.wavefront_calibration_raw = read_h5(file_path)

        if process:
            self.process_wavefront_calibration()

        return file_path
