"""
Datastructures, methods, and calibrations for an SLM monitored by a camera.
"""

import os
import time
import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy import optimize
from tqdm import tqdm

from slmsuite.holography import analysis
from slmsuite.holography import toolbox
from slmsuite.holography.algorithms import SpotHologram
from slmsuite.holography.toolbox import imprint, format_2vectors
from slmsuite.holography.toolbox.phase import blaze
from slmsuite.misc.files import read_h5, write_h5, generate_path, latest_path
from slmsuite.misc.fitfunctions import cos
from slmsuite.misc.math import REAL_TYPES


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
        mag
            See :attr:`~CameraSLM.mag`.
        """
        self.cam = cam
        self.slm = slm


class NearfieldSLM(CameraSLM):
    """
    **(NotImplemented)** Class for an SLM which is not nearly in the Fourier domain of a camera.

    Parameters
    ----------
    mag : number OR None
        Magnification between the plane where the SLM image is created
        and the camera sensor plane.
    """

    def __init__(self, cam, slm, mag=None):
        """See :meth:`CameraSLM.__init__`."""
        super().__init__(cam, slm)
        self.mag = mag


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
        using :meth:`scipy.ndimage.spline_filter`, though this mapping would be separable.
    wavefront_calibration_raw : dict or None
        Raw data for wavefront calibration, which corrects for aberrations
        in the optical system (phase_correction) and measures the amplitude distribution
        of power on the SLM (measured_amplitude).
        See :meth:`~slmsuite.hardware.cameraslms.FourierSLM.wavefront_calibrate`.
    """

    def __init__(self, *args, **kwargs):
        """See :attr:`CameraSLM.__init__`."""
        super().__init__(*args, **kwargs)

        self.fourier_calibration = None
        self.wavefront_calibration_raw = None

    ### Settle Time Measurement ###

    def measure_settle(
        self,
        vector=(.1,.1),
        basis="kxy",
        size=None,
        times=None,
        settle_time_s=1,
        plot=True
    ):
        """
        **(NotImplemented)**
        Approximates the :math`1/e` settle time of the SLM.
        This is done by successively removing and applying a blaze to the SLM,
        measuring the intensity at the first order spot versus time.

        Parameters
        ----------
        vector : array_like
            Point to TODO
        basis : {"ij", "kxy"}
            Basis of vector. This is the vector TODO
        size : int
            Size in pixels of the integration region. If ``None``, sets to sixteen
            times the approximate size of a diffraction limited spot.
        times : array_like
            List of times to sweep over in search of the :math:`1/e` settle time.
        settle_time_s : float
            Time inbetween measurements to allow the SLM to re-settle without
        plot : bool
            Whether to print debug plots.
        """
        if basis == "ij":
            vector = self.ijcam_to_kxyslm(vector)

        point = self.kxyslm_to_ijcam(vector)

        blaze = toolbox.blaze(grid=self.slm, vector=vector)

        results = []

        for t in times:
            self.slm.write(None, settle=False)
            time.sleep(settle_time_s)

            self.slm.write(blaze, settle=False)
            time.sleep(t)

            image = self.cam.get_image()

            print(np.sum(image))
            print(np.max(image))
            # nonzero = image.ravel()
            # nonzero = nonzero[nonzero != 0]
            # print(nonzero)

            _, axs = plt.subplots(1, 2, figsize=(10,4))
            axs[0].imshow(image)
            axs[1].imshow(np.squeeze(analysis.take(image, point, size, centered=True, integrate=False)))
            plt.show()

            results.append(analysis.take(image, point, size, centered=True, integrate=True))

        if plot:
            plt.plot(times, np.squeeze(results), 'k*')
            plt.ylabel("Signal [a.u.]")
            plt.xlabel("Time [sec]")
            plt.show()

        return results

    ### Fourier Calibration ###

    def fourier_calibrate(
        self,
        array_shape=10,
        array_pitch=10,
        array_center=None,
        plot=False,
        autofocus=False,
        autoexposure=False,
        **kwargs
    ):
        """
        Project and fit a SLM Fourier space (``"knm"``) grid onto
        camera pixel space (``"ij"``) for affine fitting.
        Sets :attr:`~slmsuite.hardware.cameraslms.FourierSLM.fourier_calibration`.
        An array produced by
        :meth:`~slmsuite.holography.algorithms.SpotHologram.make_rectangular_array()`
        is projected for analysis by
        :meth:`~slmsuite.holography.analysis.blob_array_detect()`.

        Important
        ~~~~~~~~~
        For best results, array_pitch should be integer data. Otherwise non-uniform
        rounding to the SLM's computational :math:`k`-space will result in non-uniform pitch and
        a bad fit.

        Parameters
        ----------
        array_shape, array_pitch
            Passed to :meth:`~slmsuite.holography.algorithms.SpotHologram.make_rectangular_array()`
            **in the** ``"knm"`` **basis.**
        array_center
            Passed to :meth:`~slmsuite.holography.algorithms.SpotHologram.make_rectangular_array()`
            **in the** ``"knm"`` **basis.**  ``array_center`` is not passed directly, and is
            processed as being relative to the center of ``"knm"`` space, the position
            of the 0th order.
        plot : bool
            Enables debug plots.
        autofocus : bool OR dict
            Whether or not to autofocus the camera.
            If a dictionary is passed, autofocus is performed,
            and the dictionary is passed to
            :meth:`~slmsuite.hardware.cameras.camera.Camera.autofocus()`.
        autoexpose : bool OR dict
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
        if isinstance(array_shape, REAL_TYPES):
            array_shape = [int(array_shape), int(array_shape)]
        if isinstance(array_pitch, REAL_TYPES):
            array_pitch = [array_pitch, array_pitch]

        self.fourier_calibration = None

        # Make and project a GS hologram across a normal grid of kvecs
        hologram = self.project_fourier_grid(
            array_shape=array_shape,
            array_pitch=array_pitch,
            array_center=array_center,
            **kwargs
        )

        # The rounding of the values might cause the center to shift from the desired
        # value. To compensate for this, we find the true written center.
        # The first two points are ignored for balance against the parity check omission
        # of the last two points.
        array_center = np.mean(hologram.spot_kxy_rounded[:, 2:], axis=1)

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

        # Get orientation of projected array
        orientation = analysis.blob_array_detect(img, array_shape, plot=plot)

        a = format_2vectors(array_center)
        M = np.array(orientation["M"])
        b = format_2vectors(orientation["b"])

        # blob_array_detect returns the calibration from ij to the space of the array, so
        # as a last step we must convert from the array to (centered) knm space, and then
        # one step further to kxy space. This is done by a simple scaling.
        scaling =   ( np.array((self.slm.dx, self.slm.dy))
                    * np.flip(np.squeeze(hologram.shape))
                    / np.squeeze(array_pitch) )

        M = np.array([  [M[0, 0] * scaling[0], M[0, 1] * scaling[1]],
                        [M[1, 0] * scaling[0], M[1, 1] * scaling[1]]  ])

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
            Name of the save file. If ``None``, will use :meth:`name_fourier_calibration`.

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
            Full path to the Fourier calibration file. If ``None``, will
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

    def project_fourier_grid(self, array_shape=10, array_pitch=10, array_center=None, **kwargs):
        """
        Projects a Fourier space grid (``"knm"``) onto pixel space (``"ij"``).
        The chosen computational k-space ``"knm"`` uses a computational shape
        generated by :meth:`SpotHologram.calculate_padded_shape` corresponding to the
        smallest square shape with power-of-two sidelength that is larger than the SLM's
        shape.

        Parameters
        ----------
        array_shape, array_pitch
            Passed to :meth:`~slmsuite.holography.algorithms.SpotHologram.make_rectangular_array()`
            **in the** ``"knm"`` **basis.**
        array_center
            Passed to :meth:`~slmsuite.holography.algorithms.SpotHologram.make_rectangular_array()`
            **in the** ``"knm"`` **basis.**  ``array_center`` is not passed directly, and is
            processed as being relative to the center of ``"knm"`` space, the position
            of the 0th order.
        **kwargs
            Passed to :meth:`~slmsuite.holography.algorithms.SpotHologram.optimize()`.

        Returns
        -------
        ~slmsuite.holography.algorithms.SpotHologram
            Optimized hologram.
        """

        # Make the spot array
        shape = SpotHologram.calculate_padded_shape(
            self, padding_order=1, square_padding=True
        )
        hologram = SpotHologram.make_rectangular_array(
            shape,
            array_shape=array_shape,
            array_pitch=array_pitch,
            array_center=None if array_center is None else (
                format_2vectors(array_center) +
                format_2vectors(((shape[1]) / 2.0, (shape[0]) / 2.0))
            ),
            basis="knm",
            orientation_check=True,
            cameraslm=self,
        )

        # Default optimize settings.
        if "maxiter" not in kwargs:
            kwargs["maxiter"] = 50

        # Optimize and project the hologram
        hologram.optimize(**kwargs)

        self.slm.write(hologram.extract_phase(), settle=True)

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
            Cleaned with :meth:`~slmsuite.holography.toolbox.format_2vectors()`.

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
        return (
            np.matmul(
                self.fourier_calibration["M"],
                format_2vectors(kxy) - self.fourier_calibration["a"]
            ) + self.fourier_calibration["b"]
        )

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
            Cleaned with :meth:`~slmsuite.holography.toolbox.format_2vectors()`.

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
        return (
            np.matmul(
                np.linalg.inv(self.fourier_calibration["M"]),
                format_2vectors(ij) - self.fourier_calibration["b"]
            ) + self.fourier_calibration["a"]
        )

    def get_farfield_spot_size(self, slm_size=None, basis="kxy"):
        """
        Calculates the size of a spot produced by blazed patch of size ``slm_size`` on the SLM.
        If this patch is the size of the SLM, then we will find in the farfield (camera)
        domain, the size of a diffraction-limited spot for a fully-illuminated surface.
        As the ``slm_size`` of the patch on the SLM decreases, the diffraction limited
        spot size in the farfield domain will of course increase. This calculation
        is accomplished using the calibration produced by
        :attr:`~slmsuite.hardware.cameraslms.FourierSLM.fourier_calibrate()`
        and stored in
        :attr:`~slmsuite.hardware.cameraslms.FourierSLM.fourier_calibration`.

        Parameters
        ----------
        slm_size : (float, float) OR int OR float OR None
            Size of patch on the SLM in normalized units.
            A scalar is interpreted as the width and height of a square.
            If ``None``, defaults to the ``shape`` of the SLM.
        basis : {"kxy", "ij"}
            Basis of the returned size; ``"kxy"`` for SLM size, ``"ij"`` for camera size.

        Returns
        -------
        (float, float)
            Size in x and y of the spot in the desired ``basis``.

        Raises
        ------
        ValueError
            If the basis argument was malformed.
        """
        if slm_size is None:
            slm_size = self.slm.shape
        if isinstance(slm_size, REAL_TYPES):
            slm_size = (slm_size, slm_size)

        size_kxy = (1 / slm_size[0], 1 / slm_size[1])

        ret = None
        if basis == "kxy":
            ret = size_kxy
        elif basis == "ij":
            ret = np.abs(self.kxyslm_to_ijcam([0, 0]) - self.kxyslm_to_ijcam(size_kxy))
        else:
            raise ValueError("Unrecognized basis \"{}\".".format(basis))

        return ret

    ### Wavefront Calibration ###

    def wavefront_calibrate(
        self,
        interference_point,
        field_point,
        field_point_units="ij",
        superpixel_size=50,
        phase_steps=10,
        exclude_superpixels=(0, 0),
        autoexposure=False,
        test_superpixel=None,
        reference_superpixel=None,
        fresh_calibration=True,
        plot=0,
    ):
        """
        Perform wavefront calibration.
        This procedure involves `iteratively interfering light diffracted from
        superpixels across an SLM with a reference superpixel <https://doi.org/10.1038/nphoton.2010.8>`_.
        Interference occurs at a given ``interference_point`` in the camera's imaging plane.
        It is at this point where the computed correction is ideal; the further away
        from this point, the less ideal the correction is.
        Sets :attr:`~slmsuite.hardware.cameraslms.FourierSLM.wavefront_calibration_raw`.
        Run :meth:`~slmsuite.hardware.cameraslms.FourierSLM.process_wavefront_calibration`
        after to produce the usable calibration which is written to the SLM.
        This procedure measures the wavefront phase and amplitude.
        If only amplitude calibration is desired,
        set ``phase_steps=0`` to omit the phase calibration.

        Parameters
        ----------
        interference_point : (float, float)
            Position in the camera domain where interference occurs.
        field_point : (float, float)
            Position in the camera domain where pixels not included in superpixels are
            blazed toward in order to reduce light in the camera's field. Suggested
            approach is to set this outside the field of view of the camera and make
            sure that other diffraction orders are far from the `interference_point`.
        field_point_units : str
            Default to ``"ij"`` which moves first diffraction order
             to the camera pixel ``field_point``.
            If it is instead a unit compatible with
            :meth:`~slmsuite.holography.toolbox.convert_blaze_vector()`, then the
            ``field_point`` value is interpreted as a shifting blaze vector.
            In this case, setting one coordinate of ``field_point`` to zero is suggested
            to minimize higher order diffraction.
        superpixel_size : int
            The width and height in pixels of each SLM superpixel.
            If this is not a devisor of both dimensions of the SLM's :attr:`shape`,
            then superpixels at the edge of the SLM may be cropped and give undefined results.
        phase_steps : int
            The number of phases measured for the interference pattern.
            If phase_steps is not strictly positive, phase is not measured:
            only amplitude is measured.
        exclude_superpixels : (int, int)
            Optionally exclude superpixels from the margin, in ``(nx, ny)`` form.
            That is, the ``nx`` superpixels are omitted from the left and right sides
            of the SLM, with the same for ``ny``. As power is
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
        fresh_calibration : bool
            If ``True``, the calibration is performed without an existing calibration
            (any old calibration is wiped from the :class:`SLM` and :class:`CameraSLM`).
            If ``False``, the calibration is performed on top of any existing
            calibration. This is useful to determine the quality of a previous
            calibration, as a new calibration should yield zero phase correction needed
            if the previous was perfect.
        plot : int or bool
            Whether to provide visual feedback, options are:

             - ``-1`` : No plots or tqdm prints.
             - ``0``, ``False`` : No plots, but tqdm prints.
             - ``1``, ``True`` : Plots on fits and essentials.
             - ``2`` : Plots on everything.
             - ``3`` : ``test_superpixel`` not ``None`` only: returns image frames
               to make a movie from the phase measurement (not for general use).

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
        return_movie = plot == 3 and test_superpixel is not None
        if return_movie:
            plot = 1
            if phase_steps <= 0:
                raise ValueError(
                    "cameraslms.py: Must have strictly positive phase_steps to produce a movie."
                )
        verbose = plot >= 0
        plot_fits = plot >= 1 or test_superpixel is not None
        plot_everything = plot >= 2

        # Clean the points
        base_point = np.around(self.kxyslm_to_ijcam([0, 0])).astype(int)
        interference_point = np.around(format_2vectors(interference_point)).astype(int)
        field_point = format_2vectors(field_point)

        # Use the Fourier calibration to help find points/sizes in the imaging plane.
        assert self.fourier_calibration is not None, \
            "Fourier calibration must be done before wavefront calibration."
        interference_blaze = self.ijcam_to_kxyslm(interference_point)
        if field_point_units == "ij":
            field_blaze = self.ijcam_to_kxyslm(field_point)
        else:
            field_blaze = toolbox.convert_blaze_vector(
                field_point,
                from_units=field_point_units,
                to_units="kxy",
                slm=self.slm
            )

            field_point = self.kxyslm_to_ijcam(field_blaze)

        field_point = np.around(format_2vectors(field_point)).astype(int)

        # Determine how many rows and columns of superpixels we will use.
        [NY, NX] = np.ceil(np.array(self.slm.shape) / superpixel_size).astype(int)

        if reference_superpixel is None:
            # Set the reference superpixel to be centered on the SLM.
            [nxref, nyref] = np.floor(  np.flip(self.slm.shape)
                                        / superpixel_size / 2).astype(int)

            reference_superpixel = [nxref, nyref]
        else:
            (nxref, nyref) = reference_superpixel

        interference_size = np.around(np.array(
            self.get_farfield_spot_size(
                (superpixel_size * self.slm.dx, superpixel_size * self.slm.dy),
                basis="ij"
            )
        )).astype(int)

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

        # Remove the current calibration
        measured_amplitude = self.slm.measured_amplitude
        phase_correction = self.slm.measured_amplitude

        if fresh_calibration:
            self.slm.measured_amplitude = None
            self.slm.phase_correction = None

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
                imprint(
                    matrix,
                    np.array([nxref, 1, nyref, 1]) * superpixel_size,
                    blaze,
                    self.slm,
                    vector=reference_blaze,
                    offset=0
                )

            if target is not None:
                imprint(
                    matrix,
                    np.array([index[0], 1, index[1], 1]) * superpixel_size,
                    blaze,
                    self.slm,
                    vector=target_blaze,
                    offset=target
                )

            if plot_everything or plot:
                plt.figure(figsize=(20, 25))
                plt.imshow(np.mod(matrix, 2 * np.pi), interpolation="none")
                plt.show()

            return matrix

        def mask(img, center, lengths):
            """
            Take a matrix img and cut everything outside a rectangle, defined by
            its center point and the lengths.

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
            return img[int(center[1] - lengths[1] / 2):int(center[1] + lengths[1] / 2),
                       int(center[0] - lengths[0] / 2):int(center[0] + lengths[0] / 2)]

        def fit_phase(phases, intensities):
            """
            Fits a sine function to the Intensity Vs. phase, and extracts best phase and amplitude
            that give the constructive interference.
            If fit fails return 0 on all values.

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
                R^2 of fit
            contrast :
                a / (a + c)
            """
            guess = [
                phases[np.argmax(intensities)],
                np.max(intensities) - np.min(intensities),
                np.min(intensities),
            ]

            try:
                popt, _ = optimize.curve_fit(cos, phases, intensities, p0=guess)
            except BaseException:
                return 0, 0, 0, 0

            # extract phase and amplitude from fit
            best_phase = popt[0]
            amp = popt[1]
            contrast = popt[1] / (popt[1] + popt[2])

            # residual and total sum of squares, producing the R^2 metric.
            ss_res = np.sum((intensities - cos(phases, *popt)) ** 2)
            ss_tot = np.sum((intensities - np.mean(intensities)) ** 2)
            r2 = 1 - (ss_res / ss_tot)

            if plot_fits:
                plt.scatter(phases / np.pi, intensities, color="k", label="Data")

                phases_fine = np.linspace(0, 2 * np.pi, 100)

                plt.plot(phases_fine / np.pi, cos(phases_fine, *popt), "k-", label="Fit")
                plt.plot(phases_fine / np.pi, cos(phases_fine, *guess), "k--",
                    label="Guess")
                plt.plot(best_phase / np.pi, popt[1] + popt[2], "xr", label="Phase")

                plt.legend(loc="best")
                plt.title("Interference ($R^2$={:.3f})".format(r2))
                plt.grid()
                plt.xlim([0, 2])
                plt.xlabel(r"$\phi$ $[\pi]$")
                plt.ylabel("Signal")

                plt.show()

            return best_phase, amp, r2, contrast

        def plot_labeled(img, plot=False, title="", plot_zoom=False):
            if plot_everything or plot:
                if return_movie:
                    fig, axs = plt.subplots(1, 3, figsize=(16,4), facecolor="white")
                else:
                    fig, axs = plt.subplots(1, 3, figsize=(16,4))
                axs[0].imshow(
                    np.mod(self.slm.phase, 2*np.pi),
                    cmap=plt.get_cmap("twilight"),
                    interpolation='none'
                )

                if plot_zoom:
                    for a in [0, 1]:
                        ref = reference_superpixel[a] * superpixel_size
                        test = test_superpixel[a] * superpixel_size

                        lim = [min(ref, test) - .5 * superpixel_size, max(ref, test) + 1.5 * superpixel_size]

                        if a:
                            axs[0].set_ylim([lim[1], lim[0]])
                        else:
                            axs[0].set_xlim(lim)

                color = "r"

                im = axs[1].imshow(np.log10(img + .1))
                im.set_clim(0, np.log10(self.cam.bitresolution))

                dpoint = field_point - base_point

                points = [(base_point + N * dpoint) for N in range(-2, 3)]
                points.append(interference_point)
                labels = [
                    "Field -2nd",
                    "Field -1st",
                    "Field 0th",
                    "Field 1st",
                    "Field 2nd",
                    "Interference\nPoint"
                ]

                wh = int(interference_size[0])
                hh = int(interference_size[1])

                for point, label in zip(points, labels):
                    if label == "Interference\nPoint":
                        wh *= 2
                        hh *= 2
                    rect = plt.Rectangle(
                        (float(point[0] - wh), float(point[1] - hh)),
                        float(2 * wh), float(2 * hh),
                        ec=color, fc="none"
                    )
                    axs[1].add_patch(rect)
                    axs[1].annotate(label,
                        (point[0], point[1] + 2 * interference_size[1] + hh),
                        c=color, size="x-small", ha="center")

                im = axs[2].imshow(np.log10(img + .1))
                axs[2].set_xlim(point[0] - wh, point[0] + wh)
                axs[2].set_ylim(point[1] + hh, point[1] - hh)
                im.set_clim(0, np.log10(self.cam.bitresolution))

                for spine in ["top", "bottom", "right", "left"]:
                    axs[2].spines[spine].set_color(color)
                    axs[2].spines[spine].set_linewidth(1.5)

                if self.cam.bitdepth > 10:
                    step = 2
                else:
                    step = 1

                bitresolution_list = np.power(2, np.arange(0, self.cam.bitdepth+1, step))

                cbar = fig.colorbar(im, ax=axs[2])
                cbar.ax.set_yticks(np.log10(bitresolution_list))
                cbar.ax.set_yticklabels(bitresolution_list)

                axs[0].set_title("SLM Phase")
                axs[1].set_title("Camera Result")
                axs[2].set_title(title)

                if plot_zoom and return_movie:
                    fig.tight_layout()
                    fig.canvas.draw()
                    image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
                    image_from_plot = image_from_plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                    plt.close()

                    return image_from_plot
                else:
                    plt.show()

        def find_center(img, plot=False):
            masked_pic_mode = mask(img, interference_point, 4 * interference_size)

            if plot_everything or plot:
                plt.imshow(masked_pic_mode)
                plt.show()

            # Future: use the below
            # found_center = analysis.image_positions([masked_pic_mode]) + interference_point

            # Blur a lot and assume the maximum corresponds to the center.
            blur = 2 * int(np.min(interference_size)) + 1
            masked_pic_mode = analysis._make_8bit(masked_pic_mode)
            masked_pic_mode = cv2.GaussianBlur(masked_pic_mode, (blur, blur), 0)
            _, _, _, max_loc = cv2.minMaxLoc(masked_pic_mode)
            found_center = (format_2vectors(max_loc)
                            - format_2vectors(np.flip(masked_pic_mode.shape)) / 2
                            + interference_point)

            return found_center

        def measure(index, plot=False):
            self.cam.flush()

            # Step 0: Measure the background.
            self.slm.write( superpixels(index, reference=None, target=None),
                            settle=True    )
            background_image = self.cam.get_image()
            plot_labeled(background_image, plot=plot, title="Background")
            back = mask(background_image, interference_point,
                        2 * interference_size).sum()

            # Step 0.5: Measure the power in the reference mode.
            self.slm.write( superpixels(index, reference=0, target=None),
                            settle=True    )
            normalization_image = self.cam.get_image()
            plot_labeled(normalization_image, plot=plot, title="Reference Diffraction")
            norm = mask(normalization_image, interference_point,
                        2 * interference_size).sum()

            # Step 1: Add a blaze to the target mode so that it overlaps with
            # reference mode.
            self.slm.write( superpixels(index, reference=None, target=0),
                            settle=True    )
            position_image = self.cam.get_image()
            plot_labeled(position_image, plot=plot, title="Base Target Diffraction")
            found_center = find_center(position_image)

            blaze_difference = self.ijcam_to_kxyslm(found_center) - interference_blaze
            target_blaze_fixed = interference_blaze - blaze_difference

            # Step 1.25: Stop here if we don't need to measure the phase.
            if phase_steps <= 0:
                pwr = mask(position_image, interference_point, 2 * interference_size).sum()
                return {
                    "power": pwr,
                    "normalization": norm,
                    "background": back,
                    "phase": np.nan,
                    "kx": np.nan,
                    "ky": np.nan,
                    "amp_fit": np.nan,
                    "contrast_fit": np.nan,
                    "r2_fit": np.nan,
                }

            # Step 1.5: Measure the power in the corrected target mode.
            self.slm.write( superpixels(index, reference=None, target=0,
                                        target_blaze=target_blaze_fixed),
                            settle=True)
            fixed_image = self.cam.get_image()
            plot_labeled(fixed_image, plot=plot, title="Corrected Target Diffraction")
            pwr = mask(fixed_image, interference_point, 2 * interference_size).sum()

            # Step 2: Measure interference and find relative phase
            phases = np.linspace(0, 2 * np.pi, phase_steps, endpoint=False)
            results = []  # list for recording the intensity of the reference point

            # Determine whether to use a progress bar.
            if verbose:
                description = "superpixel=({},{})".format(index[0], index[1])
                prange = tqdm(phases, position=0, leave=False, desc=description)
            else:
                prange = phases

            if return_movie: frames = []

            # Step 3: Measure phase
            for phase in prange:
                self.slm.write( superpixels(index, reference=0,target=phase,
                                            target_blaze=target_blaze_fixed),
                                settle=True)
                interference_image = self.cam.get_image()
                results.append(
                    interference_image[
                        int(interference_point[1]), int(interference_point[0])
                    ]
                )

                if return_movie:
                    frames.append(
                        plot_labeled(
                            interference_image,
                            plot=plot,
                            title="Phase = ${:1.2f}\pi$".format(phase / np.pi),
                            plot_zoom=True
                        )
                    )

            # Step 4: Fit to sine and return.
            phase_fit, amp_fit, r2_fit, contrast_fit = fit_phase(phases, results)


            self.slm.write( superpixels(index, reference=0, target=phase_fit,
                                        target_blaze=target_blaze_fixed),
                            settle=True)
            interference_image = self.cam.get_image()
            if plot:
                plot_labeled(interference_image, plot=plot, title="Best Interference")

            if test_superpixel is not None:
                self.slm.measured_amplitude = measured_amplitude
                self.slm.phase_correction = phase_correction

            if return_movie: return frames

            return {
                "power": pwr,
                "normalization": norm,
                "background": back,
                "phase": phase_fit,
                "kx": -float(blaze_difference[0]),
                "ky": -float(blaze_difference[1]),
                "amp_fit": amp_fit,
                "contrast_fit": contrast_fit,
                "r2_fit": r2_fit,
            }

        # Correct exposure and position of the reference mode.
        self.slm.write( superpixels((0, 0), reference=0, target=None),
                        settle=True)
        self.cam.flush()

        if autoexposure:
            window = [  interference_point[0], 2 * interference_size[0],
                        interference_point[1], 2 * interference_size[1] ]
            self.cam.autoexposure(set_fraction=0.1, window=window)

        base_image = self.cam.get_image()
        plot_labeled(base_image, plot=plot_everything, title="Base Reference Diffraction")
        found_center = find_center(base_image)

        # Correct the original blaze using the measured result.
        blaze_difference = self.ijcam_to_kxyslm(found_center) - interference_blaze
        reference_blaze_fixed = interference_blaze - blaze_difference

        if plot_fits:
            self.slm.write( superpixels((0, 0), reference=0, target=None,
                                        reference_blaze=reference_blaze_fixed),
                            settle=True)
            fixed_image = self.cam.get_image()
            plot_labeled(fixed_image, plot=plot_everything, title="Corrected Reference Diffraction")
            found_center = find_center(fixed_image)

        # If we just want to debug/test one region, then do so.
        if test_superpixel is not None:
            return measure(test_superpixel, plot=plot_fits)

        # Otherwise, proceed with all of the superpixels.
        for n in tqdm(range(NX * NY), position=0, leave=False, desc="calibration"):
            nx = int(n % NX)
            ny = int(n / NX)

            # Exclude the reference mode.
            if nx == nxref and ny == nyref:
                continue

            # Exclude margin superpixels, if desired.
            if nx < exclude_superpixels[0]:
                continue
            if nx > NX - exclude_superpixels[0]:
                continue
            if ny < exclude_superpixels[1]:
                continue
            if ny > NY - exclude_superpixels[1]:
                continue

            # Measure!
            measurement = measure((nx, ny))

            # Update dictionary.
            for key in measurement:
                correction_dict[key][ny, nx] = measurement[key]

        self.wavefront_calibration_raw = correction_dict

        return correction_dict

    def process_wavefront_calibration(
            self,
            smooth=True,
            r2_threshold=0.9,
            apply=True,
            plot=False
        ):
        """
        Processes :attr:`~slmsuite.hardware.cameraslms.FourierSLM.wavefront_calibration_raw`
        into the desired phase correction and amplitude measurement. Applies these
        parameters to the respective variables in the SLM if ``apply`` is ``True``.

        Parameters
        ----------
        smooth : bool
            Whether to blur the correction data to avoid aliasing.
        r2_threshold : float
            Threshold for a "good fit". Proxy for whether a datapoint should be used or
            ignored in the final data, depending upon the rsquared value of the fit.
            Should be within [0, 1].
        apply : bool
            Whether to apply the processed calibration to the associated SLM.
            Otherwise, this function only returns and maybe
            plots these results. Defaults to ``True``.
        plot : bool
            Whether to enable debug plots.

        Returns
        -------
        dict
            A dictionary consisting of the ``measured_amplitude`` and
            ``phase_correction``. With the same names as keys.
        """
        # Step 0: Initialize helper variables and functions.
        data = self.wavefront_calibration_raw

        if len(data) == 0:
            raise RuntimeError("No raw wavefront data to process. Either load data or calibrate.")

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

        # Step 1: Process the measured amplitude
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
            size_blur = 4*int(superpixel_size) + 1
            pwr_large = cv2.GaussianBlur(pwr_large, (size_blur, size_blur), 0)

        amp = np.sqrt(pwr_norm)
        amp_large = np.sqrt(pwr_large)

        # Step 2: Process R^2
        r2 = np.copy(data["r2_fit"])
        r2[nyref, nxref] = 1
        r2s = r2

        r2s_large = cv2.resize(r2s, (w, h), interpolation=cv2.INTER_NEAREST)
        r2s_large = r2s_large[:self.slm.shape[0], :self.slm.shape[1]]

        # Step 3: Process the wavefront
        # Load data.
        kx = np.copy(data["kx"])
        ky = np.copy(data["ky"])

        offset = np.copy(data["phase"])

        real = np.cos(offset)
        imag = np.sin(offset)

        # Fill in the reference pixel with surrounding data.
        average_neighbors(kx)
        average_neighbors(ky)

        average_neighbors(real)
        average_neighbors(imag)

        # Cleanup the phase.
        offset = np.arctan2(imag, real) + np.pi

        kx[r2s < r2_threshold] = 0
        ky[r2s < r2_threshold] = 0
        offset[r2s < r2_threshold] = 0
        pathing = 0 * r2s

        # Step 3.5: Infer phase for superpixels which do satisfy the R^2 threshold.
        # For each row...
        for ny in range(NY):
            # Go forward and then back along each row.
            for nx in list(range(NX)) + list(range(NX-1, -1, -1)):
                if r2s[ny, nx] >= r2_threshold:
                    # Superpixels exceeding the threshold need no correction.
                    pass
                else:
                    # Otherwise, do a majority-vote with adjacent superpixels.
                    kx2 = []
                    ky2 = []
                    offset2 = []

                    # Loop through the adjacent superpixels (including diagonals).
                    for (ax, ay) in [(1,  0), (-1,  0), (0, 1), (0, -1),
                                     (1, -1), (-1, -1), (1, 1), (-1, 1)]:
                        (tx, ty) = (nx + ax, ny + ay)
                        (dx, dy) = (
                            2 * np.pi * (nx - nxref) * superpixel_size * self.slm.dx,
                            2 * np.pi * (ny - nyref) * superpixel_size * self.slm.dy)

                        # Make sure our adjacent pixel under test is within range and above threshold.
                        if (tx >= 0 and tx < NX and ty >= 0 and ty < NY and
                            (r2s[ty, tx] >= r2_threshold)): # or pathing[ty, tx] == ny)):

                            kx3 = kx[ty, tx]
                            ky3 = ky[ty, tx]

                            kx2.append(kx3)
                            ky2.append(ky3)
                            offset2.append(offset[ty, tx] + dx * kx3 + dy * ky3)

                    # Do a majority vote (within std) for the phase.
                    if len(kx2) > 0:
                        kx[ny, nx] = 0 #np.mean(kx2)
                        ky[ny, nx] = 0 #np.mean(ky2)

                        minstd = np.inf
                        for phi in range(4):
                            shift = phi * np.pi / 2
                            offset3 = np.mod(np.array(offset2) + shift, 2 * np.pi)

                            if minstd > np.std(offset3):
                                minstd = np.std(offset3)
                                offset[ny, nx] = np.mod(np.mean(offset3) - shift,
                                                        2 * np.pi)

                        pathing[ny, nx] = ny

        # Step 3.75: Make the SLM-sized correction using the compressed data from each superpixel.
        phase = np.zeros(self.slm.shape)
        for nx in range(NX):
            for ny in range(NY):
                imprint(
                    phase,
                    np.array([nx, 1, ny, 1]) * superpixel_size,
                    blaze,
                    self.slm,
                    vector=(kx[ny, nx], ky[ny, nx]),
                    offset=offset[ny, nx],
                )

        if smooth:
            # Iterative smoothing helps to preserve slopes while avoiding superpixel boundaries.
            # Consider, for instance, a fine blaze.
            for _ in range(16):
                real = np.cos(phase)
                imag = np.sin(phase)

                # Blur the phase to smooth it out
                size_blur = 2*int(superpixel_size/4) + 1
                real = cv2.GaussianBlur(real, (size_blur, size_blur), 0)
                imag = cv2.GaussianBlur(imag, (size_blur, size_blur), 0)

                phase = np.arctan2(imag, real) + np.pi
        else:
            real = np.cos(phase)
            imag = np.sin(phase)
            phase = np.arctan2(imag, real) + np.pi

        # Shift the final phase to the nearest pi/4 phase offset which
        # minimizes 2pi -> 0pi shearing.
        mindiff = np.inf
        phase_fin = []
        for phi in range(8):
            shift = phi * np.pi / 4

            modthis = np.mod(phase + shift, 2 * np.pi)

            fom = (np.sum(np.abs(np.diff(modthis, axis=0))) +
                   np.sum(np.abs(np.diff(modthis, axis=1))))

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

        wavefront_calibration = {   "phase_correction":phase_fin,
                                    "measured_amplitude":amp_large}

        # Step 4: Load the correction to the SLM
        if apply:
            self.slm.phase_correction = phase_fin
            self.slm.measured_amplitude = amp_large

        # Plot the result
        if plot:
            plt.figure(figsize=(16, 8))

            plt.subplot(1, 3, 1)
            plt.imshow(
                phase_fin,
                cmap=plt.get_cmap("twilight"),
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
            plt.title("$R^2$")
            plt.xlabel("SLM $x$ [superpix]")
            plt.ylabel("SLM $y$ [superpix]")

            plt.show()

        return wavefront_calibration

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
            Name of the save file. If ``None``, will use :meth:`name_wavefront_calibration`.

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

    def load_wavefront_calibration(self, file_path=None, process=True, **kwargs):
        """
        Loads :attr:`~slmsuite.hardware.cameraslms.FourierSLM.wavefront_calibration_raw`
        from a file.

        Parameters
        ----------
        file_path : str or None
            Full path to the wavefront calibration file. If ``None``, will
            search the current directory for a file with a name like
            the one returned by
            :meth:`~slmsuite.hardware.cameraslms.FourierSLM.name_wavefront_calibration`.
        process : bool
            Whether to immediately process the wavefront calibration.
            See
            :meth:`~slmsuite.hardware.cameraslms.FourierSLM.process_wavefront_calibration`.
        **kwargs
            Passed to :meth:`~slmsuite.hardware.cameraslms.FourierSLM.process_wavefront_calibration`,
            if ``process`` is true.

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
            self.process_wavefront_calibration(**kwargs)

        return file_path
