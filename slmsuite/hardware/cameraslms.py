"""
Datastructures, methods, and calibrations for an SLM monitored by a camera.
"""

import os
import time
import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy import optimize
from tqdm.autonotebook import tqdm
import warnings

from slmsuite.holography import analysis
from slmsuite.holography import toolbox
from slmsuite.holography.algorithms import SpotHologram
from slmsuite.holography.toolbox import imprint, format_2vectors, smallest_distance, fit_3pt
from scipy.spatial.distance import chebyshev, euclidean
from slmsuite.holography.toolbox.phase import blaze
from slmsuite.misc.files import read_h5, write_h5, generate_path, latest_path
from slmsuite.misc.fitfunctions import cos, sinc2d
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

        # Size of the calibration point window relative to the spot.
        self._wavefront_calibration_window_multiplier = 4

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
        Approximates the :math:`1/e` settle time of the SLM.
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

        If the vectors are three-dimensional, the third depth dimension is treated according to:

        .. math:: y_z = \frac{f_{eff}^2}{f_{slm}} = f_{eff}^2x_z

        where :math:`x_z`, equivalent to focal power,
        is converted into depth :math:`y_z` in pixels.

        Parameters
        ----------
        kxy : array_like
            Vector or array of vectors to convert. Can be 2D or 3D.
            Cleaned with :meth:`~slmsuite.holography.toolbox.format_2vectors()`.

        Returns
        -------
        ij : numpy.ndarray
            Vector or array of vectors in camera spatial coordinates. Can be 2D or 3D.

        Raises
        ------
        RuntimeError
            If the fourier plane calibration does not exist.
        """
        if self.fourier_calibration is None:
            raise RuntimeError("Fourier calibration must exist to be used.")

        kxy = format_2vectors(kxy, handle_dimension="pass")

        # Apply the xy transformation.
        ij = np.matmul(
            self.fourier_calibration["M"],
            kxy[:2, :] - self.fourier_calibration["a"]
        ) + self.fourier_calibration["b"]

        # Handle z if needed.
        if ij.shape[0] == 3:
            f_eff = self.get_effective_focal_length("norm")
            pix2um = self._get_camera_pix2um()
            z = ij[[2], :] * (self.slm.wav_um * f_eff * f_eff / pix2um)
            return np.vstack((ij, z))
        else:
            return ij

    def ijcam_to_kxyslm(self, ij):
        r"""
        Converts camera pixel space (``"ij"``) to SLM Fourier space (``"kxy"``).
        For camera pixel indices :math:`\vec{y}` and blaze vectors :math:`\vec{x}`, computes:

        .. math:: \vec{x} = M^{-1} \cdot (\vec{y} - \vec{b}) + \vec{a}

        where :math:`M`, :math:`\vec{b}`, and :math:`\vec{a}` are stored in
        :attr:`~slmsuite.hardware.cameraslms.FourierSLM.fourier_calibration`.

        If the vectors are three-dimensional, the third depth dimension is treated according to:

        .. math:: x_z = \frac{1}{f_{slm}} = \frac{y_z}{f_{eff}^2}

        where this factor, equivalent to focal power, is used as the normalized prefactor of
        the quadratic term of a simple thin lens.

        Parameters
        ----------
        ij : array_like
            Vector or array of vectors to convert. Can be 2D or 3D.
            Cleaned with :meth:`~slmsuite.holography.toolbox.format_2vectors()`.

        Returns
        -------
        kxy : numpy.ndarray
            Vector or array of vectors in slm angular coordinates. Can be 2D or 3D.

        Raises
        ------
        RuntimeError
            If the fourier plane calibration does not exist.
        """
        if self.fourier_calibration is None:
            raise RuntimeError("Fourier calibration must exist to be used.")

        ij = format_2vectors(ij, handle_dimension="pass")

        # Apply the xy transformation.
        kxy = np.matmul(
            np.linalg.inv(self.fourier_calibration["M"]),
            ij[:2, :] - self.fourier_calibration["b"]
        ) + self.fourier_calibration["a"]

        # Handle z if needed.
        if ij.shape[0] == 3:
            f_eff = self.get_effective_focal_length("norm")
            pix2um = self._get_camera_pix2um()
            z = ij[[2], :] * (pix2um / (self.slm.wav_um * f_eff * f_eff))
            return np.vstack((kxy, z))
        else:
            return kxy

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

    def _get_camera_pix2um(self):
        if self.cam.dx_um is None or self.cam.dy_um is None:
            raise ValueError("Camera dx_um or dy_um are not set.")
        if self.cam.dx_um != self.cam.dy_um:
            warnings.warn("Camera does not have square pitch. Odd behavior might result.")
        return np.mean([self.cam.dx_um, self.cam.dy_um])

    def get_effective_focal_length(self, units="norm"):
        """
        Uses the Fourier calibration to estimate the effective focal length of the
        optical train separating the Fourier-domain SLM from the camera.
        This currently assumes an isotropic imaging train without cylindrical optics.

        Parameters
        ----------
        units : {"pix", "um", "norm"}
            Units for the focal length.

        Returns
        -------
        f_eff : float
            Effective focal length.
        """
        if self.fourier_calibration is None:
            raise RuntimeError("Fourier calibration must exist to be used.")

        # Gather f_eff in pix/rad.
        f_eff = np.sqrt(np.abs(np.linalg.det(self.fourier_calibration["M"])))

        # Gather other conversions.
        pix2um = self._get_camera_pix2um()

        # Convert.
        if units == "pix":
            pass
        elif units == "um":
            f_eff *= pix2um
        elif units == "norm":
            f_eff *= pix2um / self.slm.wav_um

        return f_eff

    ### Wavefront Calibration ###

    def get_wavefront_calibration_points(
        self,
        pitch,
        field_exclusion=None,
        field_point=(0,0),
        field_point_units="kxy",
        avoid_points=None,
        plot=False,
    ):
        """
        Generates a grid of points to perform wavefront calibration at.

        Parameters
        ----------
        pitch : float OR (float, float)
            The grid of points must have pitch greater than this value.
        field_exclusion : float OR None
            If ``None``, defaults to ``pitch``.
        field_point : (float, float)
            Position in the camera domain where pixels not included in superpixels are
            blazed toward in order to reduce light in the camera's field. Suggested
            approach is to set this outside the field of view of the camera and make
            sure that other diffraction orders are far from the `calibration_points`.
            Defaults to no blaze.
        field_point_units : str
            Default to ``"ij"`` which moves first diffraction order
             to the camera pixel ``field_point``.
            If it is instead a unit compatible with
            :meth:`~slmsuite.holography.toolbox.convert_blaze_vector()`, then the
            ``field_point`` value is interpreted as a shifting blaze vector.
            In this case, setting one coordinate of ``field_point`` to zero is suggested
            to minimize higher order diffraction.
        avoid_points : numpy.ndarray
            Additional points to avoid

        Returns
        -------
        numpy.ndarray
            List of points of dimension ``2 x N`` to calibrate at.

        Raises
        ------
        AssertionError
            If the fourier plane calibration does not exist.
        """
        # Parse field_point
        if field_point_units != "ij":
            field_blaze = toolbox.convert_blaze_vector(
                format_2vectors(field_point),
                from_units=field_point_units,
                to_units="kxy",
                slm=self.slm
            )

            field_point = self.kxyslm_to_ijcam(field_blaze)

        field_point = np.around(format_2vectors(field_point)).astype(int)

        # Parse field_exclusion
        if field_exclusion is None:
            field_exclusion = pitch
        if not np.isscalar(field_exclusion):
            field_exclusion = np.mean(field_exclusion)

        # Gather other information
        base_point = np.around(self.kxyslm_to_ijcam([0, 0])).astype(int)
        
        # Generate the intiial grid
        plane = format_2vectors(self.cam.shape[::-1])
        grid = np.floor(plane / pitch).astype(int)
        spacing = plane / grid

        calibration_points = fit_3pt(
            spacing/2,
            (1.5*spacing[0], spacing[1]/2),
            (spacing[0]/2, 1.5*spacing[1]),
            np.squeeze(grid)
        )

        # Sort by proximity to the center, avoiding the 0th order
        distance = np.sum(np.square(calibration_points - base_point), axis=0)
        I = np.argsort(distance)
        calibration_points = calibration_points[:, I]

        # Prune points within field_exclusion from a given order (-2, ..., 2).
        dorder = field_point - base_point

        order_points = np.hstack([base_point + dorder * i for i in range(-2, 3)])

        if avoid_points is None:
            avoid_points = order_points
        else:
            avoid_points = np.hstack((format_2vectors(avoid_points), order_points))

        for i in range(avoid_points.shape[1]):
            point = avoid_points[:, [i]]
            distance = np.sum(np.square(calibration_points - point), axis=0)
            calibration_points = np.delete(
                calibration_points, 
                distance < field_exclusion*field_exclusion, 
                axis=1
            )

            if plot: plt.scatter(point[0], point[1], c="r")

        if plot: 
            plt.scatter(
                calibration_points[0,:],
                calibration_points[1,:], 
                c=np.arange(calibration_points.shape[1])
            )
            plt.xlim([0, self.cam.shape[1]])
            plt.ylim([0, self.cam.shape[0]])
            plt.show()

        return calibration_points

    def get_wavefront_calibration_window(self, superpixel_size):
        """TODO"""
        interference_size = np.around(np.array(
            self.get_farfield_spot_size(
                (superpixel_size * self.slm.dx, superpixel_size * self.slm.dy),
                basis="ij"
            )
        )).astype(int)

        return self._wavefront_calibration_window_multiplier * interference_size

    def wavefront_calibrate(
        self,
        calibration_points=None,
        superpixel_size=50,
        reference_superpixels=None,
        test_superpixel=None,
        exclude_superpixels=(0, 0),
        field_point=(0,0),
        field_point_units="kxy",
        phase_steps=1,
        fresh_calibration=True,
        measure_background=False,
        corrected_amplitude=False,
        plot=0,
    ):
        """
        Perform wavefront calibration.
        This procedure involves `iteratively interfering light diffracted from
        superpixels across an SLM with a reference superpixel <https://doi.org/10.1038/nphoton.2010.8>`_.
        Interference occurs at a given ``calibration_points`` in the camera's imaging plane.
        It is at each point where the computed correction is ideal; the further away
        from each point, the less ideal the correction is.
        Correction at many points over the plane permits a better understanding of the
        aberration and greater possibility of compensation.
        Sets :attr:`~slmsuite.hardware.cameraslms.FourierSLM.wavefront_calibration_raw`.
        Run :meth:`~slmsuite.hardware.cameraslms.FourierSLM.process_wavefront_calibration`
        after to produce the usable calibration which is written to the SLM.
        This procedure measures the wavefront phase and amplitude.

        Tip
        ~~~
        If only amplitude calibration is desired,
        set ``phase_steps=None`` to omit the more time-consuming phase calibration.

        Tip
        ~~~
        Set ``phase_steps=1`` for faster calibration. This fits the phase fringes of an
        image rather than scanning the fringes over a single camera pixel over many
        ``phase_steps``. This is usually optimal except in cases with excessive noise.

        Parameters
        ----------
        calibration_points : (float, float) OR None
            Position(s) in the camera domain where interference occurs.
            This is naturally in the ``"ij"`` basis.
            If None, densely fills the camera field of view with calibration points.
        superpixel_size : int
            The width and height in pixels of each SLM superpixel.
            If this is not a devisor of both dimensions in the SLM's :attr:`shape`,
            then superpixels at the edge of the SLM may be cropped and give undefined results.
            Currently, superpixels are forced to be square, and this value must be a scalar.
        reference_superpixels : (int, int) OR [(int, int)] OR None
            The superpixel to reference from. Defaults to the center of the SLM.
            If multiple calibration points are desired, then the references are clustered at the center.
        test_superpixels : (int, int) OR [(int, int)] OR int OR None
            Test an iteration of wavefront calibration using the given superpixels.
            If ``(int, int)``, then tests the first calibration point
            If ``[(int, int)]``, then
            If ``int``, then tests the scheduled frame corresponding to this index.
            Defaults to ``None``, which does not test and instead runs the full calibration.
        exclude_superpixels : (int, int) OR numpy.ndarray OR None
            If in ``(nx, ny)`` form, optionally exclude superpixels from the margin,
            That is, the ``nx`` superpixels are omitted from the left and right sides
            of the SLM, with the same for ``ny``. As power is
            typically concentrated in the center of the SLM, this function is useful for
            excluding points that are known to be blocked, or for quickly testing calibration
            at the most relevant points.
            Otherwise, if exclude_superpixels is an image with the same dimension as the
            superpixeled SLM, this image is interpreted as a denylist.
            Defaults to ``None``, where no superpixels are excluded.
        field_point : (float, float)
            Position in the camera domain where pixels not included in superpixels are
            blazed toward in order to reduce light in the camera's field. Suggested
            approach is to set this outside the field of view of the camera and make
            sure that other diffraction orders are far from the `calibration_points`.
            Defaults to no blaze.
        field_point_units : str
            Default to ``"ij"`` which moves first diffraction order
             to the camera pixel ``field_point``.
            If it is instead a unit compatible with
            :meth:`~slmsuite.holography.toolbox.convert_blaze_vector()`, then the
            ``field_point`` value is interpreted as a shifting blaze vector.
            In this case, setting one coordinate of ``field_point`` to zero is suggested
            to minimize higher order diffraction.
        phase_steps : int
            The number of interference phases to measure.
            If ``phase_steps`` is ``None`, phase is not measured;
            only amplitude is measured.
            If ``phase_steps`` is one, does a one-shot fit on the interference
            pattern to improve speed.
        fresh_calibration : bool
            If ``True``, the calibration is performed without an existing calibration
            (any old global calibration is wiped from the :class:`SLM` and :class:`CameraSLM`).
            If ``False``, the calibration is performed on top of any existing
            calibration. This is useful to determine the quality of a previous
            calibration, as a new calibration should yield zero phase correction needed
            if the previous was perfect. The old calibration will be stored in the
            :attr:`wavefront_calibration_raw` under ``"previous_phase_correction"``,
            so keep in mind that this (uncompressed) image will take up significantly
            more space.
        measure_background : bool
            Whether to measure the background at each point.
        corrected_amplitude : bool
            If ``False``, the power in the uncorrected target beam
            is used as the source of amplitude measurement.
            If ``True``, adds another step to measure the power of the corrected target beam.
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
        RuntimeError
            If the Fourier plane calibration does not exist.
        ValueError
            If various points are out of range.
        """
        # Parse the superpixel size and derived quantities.
        superpixel_size = int(superpixel_size)

        slm_supershape = tuple(
            np.ceil(np.array(self.slm.shape) / superpixel_size).astype(int)
        )
        num_superpixels = slm_supershape[0] * slm_supershape[1]

        # Next, we get the size of the window necessary to measure a spot
        # produced by the given superpixel size.
        interference_window = self.get_wavefront_calibration_window(superpixel_size)
        interference_size = interference_window / self._wavefront_calibration_window_multiplier

        # Now that we have the supershape, we label each of the pixels with an index.
        # It's sometimes useful to map that index to the xy coordinates of the pixel,
        # hence this function.
        def index2coord(index):
            return format_2vectors(
                np.stack((index % slm_supershape[1], index // slm_supershape[1]), axis=0)
            )
        
        # It's also useful to make an image showing the given indices.
        def index2image(index):
            image = np.zeros(slm_supershape)
            image.ravel()[index] = True
            return image
        
        # Parse exclude_superpixels
        exclude_superpixels = np.array(exclude_superpixels)

        if exclude_superpixels.shape == slm_supershape:
            exclude_superpixels = exclude_superpixels != 0
        elif exclude_superpixels.size == 2:
            exclude_margin = exclude_superpixels.astype(int)

            # Make the image based on margin values
            exclude_superpixels = np.zeros(slm_supershape)
            exclude_superpixels[:, :exclude_margin[0]] = True
            exclude_superpixels[:, slm_supershape[1]-exclude_margin[0]:] = True
            exclude_superpixels[:exclude_margin[1], :] = True
            exclude_superpixels[slm_supershape[0]-exclude_margin[1]:, :] = True
        else:
            raise ValueError("Did not recognize type for exclude_superpixels")

        num_active_superpixels = int(np.sum(np.logical_not(exclude_superpixels)))

        # Parse calibration_points.
        if calibration_points is None:
            # If None, then use the built-in generator.
            calibration_points = self.generate_wavefront_calibration_points(
                1.5*np.max(interference_window),
                np.max(interference_window),
                field_point,
                field_point_units,
                plot=False
            )

        calibration_points = np.around(format_2vectors(calibration_points)).astype(int)

        num_points = calibration_points.shape[1]

        # Clean the base and field points.
        base_point = np.around(self.kxyslm_to_ijcam([0, 0])).astype(int)

        if field_point_units != "ij":
            field_blaze = toolbox.convert_blaze_vector(
                format_2vectors(field_point),
                from_units=field_point_units,
                to_units="kxy",
                slm=self.slm
            )

            field_point = self.kxyslm_to_ijcam(field_blaze)
        else:
            field_blaze = toolbox.convert_blaze_vector(
                field_point,
                from_units="ij",
                to_units="kxy",
                slm=self
            )

        field_point = np.around(format_2vectors(field_point)).astype(int)

        # Use the Fourier calibration to help find points/sizes in the imaging plane.
        if self.fourier_calibration is None:
            raise RuntimeError("Fourier calibration must be done before wavefront calibration.")
        calibration_blazes = self.ijcam_to_kxyslm(calibration_points)
        reference_blazes = calibration_blazes.copy()

        # Set the reference superpixels to be centered on the SLM if undefined.
        if reference_superpixels is None:
            all_superpixels = np.arange(num_superpixels)
            all_superpixels_coords = index2coord(all_superpixels)

            distance = np.sum(np.square(all_superpixels_coords - format_2vectors(slm_supershape[::-1])/2), axis=0)
            I = np.argsort(distance)

            reference_superpixels = I[:num_points]
        else:
            raise NotImplementedError("TODO")

        # Error check the reference superpixels.
        reference_superpixels_coords = index2coord(reference_superpixels)
        reference_superpixels_image = index2image(reference_superpixels)
        if (np.any(np.logical_and(reference_superpixels_image, exclude_superpixels))):
            raise ValueError("reference_superpixels out of range of calibration.")
            
        # Now we have to solve the challenge of when to measure each target-reference pair.
        num_measurements = num_active_superpixels + ((2*num_points - 2) if phase_steps is not None else 0)

        index_image = np.reshape(np.arange(num_superpixels, dtype=int), slm_supershape)
        active_superpixels = index_image[np.logical_not(exclude_superpixels)].ravel()

        # The base schedule cycles through all indices apart from the base reference index.
        scheduling = np.zeros((num_points, num_measurements), dtype=int)

        scheduling[:, :(num_active_superpixels-1)] = np.mod(
            np.repeat(np.arange(num_active_superpixels-1, dtype=int)[np.newaxis, :] + 1, num_points, axis=0) +
            np.repeat(reference_superpixels[:, np.newaxis], num_active_superpixels-1, axis=1),
            num_active_superpixels
        )

        # Account for some superpixels being excluded.
        scheduling = active_superpixels[scheduling]
        scheduling[:, (num_active_superpixels-1):] = -1

        # Remove conflicts where other calibration pairs are targeting another
        # reference superpixel. Only do this when we are measuring relative phase
        # (if phase_steps is None, then we never write reference superpixels).
        if phase_steps is not None:
            for i in range(num_points):     # TODO: Make more efficient.
                # For each calibration point, determine where the reference index is being overwritten.
                reference_index = reference_superpixels[i]

                conflicts = scheduling == reference_index
                conflict_indices = np.array(np.where(conflicts))

                for j in range(int(np.sum(conflicts))):
                    # For each time that the index is overwritten, 
                    # reassign the target to empty space.
                    c_index = conflict_indices[:, j]

                    # This is the overwritten target index.
                    displaced_index = scheduling[i, c_index[1]]
                    scheduling[i, c_index[1]] = -1

                    # Find a point in empty space to resettle the index, if it is not already unused.
                    # This algorithm is currently quite slow. Consider speeding?
                    if displaced_index != -1:
                        for k in range(num_active_superpixels-1, num_measurements+1):
                            if k == num_measurements:
                                raise RuntimeError("TODO")
                            elif (
                                scheduling[i, k] == -1 
                                and not np.any(scheduling[:, k] == reference_index) 
                                and not np.any(scheduling[:, k] == displaced_index)
                            ):
                                scheduling[i, k] = displaced_index
                                break

        # Cleanup the scheduling.
        empty_schedules = np.all(scheduling == -1, axis=0)
        scheduling = scheduling[:, np.logical_not(empty_schedules)]
        num_measurements = scheduling.shape[1]

        # Error check whether we expect to be able to see fringes.
        # max_dist_slmpix = np.max(np.hstack((
        #     reference_superpixels - exclude_superpixels,
        #     slm_supershape[::-1, [0]] - exclude_superpixels - reference_superpixels
        # )), axis=1)
        # max_r_slmpix = np.sqrt(np.sum(np.square(max_dist_slmpix)))
        # fringe_period = np.mean(interference_size) / max_r_slmpix / 2

        # if fringe_period < 2:
        #     warnings.warn(
        #         "Non-resolvable interference fringe period "
        #         "for the given SLM calibration extent. "
        #         "Either exclude more of the SLM or magnify the field on the camera."
        #     )

        # Error-check if we're measuring multiple sites at once.
        if num_points > 1:
            interference_distance = smallest_distance(calibration_points, euclidean)
            if np.max(interference_window) > interference_distance:
                raise ValueError(
                    "Requested interference points are too close together. "
                    "The minimum distance {} pix is smaller than twice the window size {} pix."
                    .format(interference_distance, 2 * interference_window)
                )

        # Error check interference point proximity to the 0th order.
        dorder = field_point - base_point
        interference_distance_orders = smallest_distance(
            np.hstack((
                calibration_points,
                base_point,             #  0th order
                field_point,            # +1st order
                field_point + dorder,   # +2nd order
                base_point - dorder     # -1st order
            )), 
            euclidean
        )

        if np.mean(interference_window) > interference_distance_orders:
            warnings.warn(
                "The requested interference point(s) are close to the expected positions of "
                "the field diffractive orders. Consider moving interference regions further away."
            )

        # Save the current calibration in case we are just testing (test_superpixel != None)
        measured_amplitude = self.slm.measured_amplitude
        phase_correction =   self.slm.phase_correction

        # If we're starting fresh, remove the old calibration such that this does not
        # muddle things. If we're only testing, the stored data above will be reinstated.
        if fresh_calibration:
            self.slm.measured_amplitude = None
            self.slm.phase_correction = None

        # Parse phase_steps
        if phase_steps is not None:
            if not np.isclose(phase_steps, int(phase_steps)):
                raise ValueError(f"Expected integer phase_steps. Received {phase_steps}.")
            phase_steps = int(phase_steps)
            if phase_steps <= 0:
                raise ValueError(f"Expected positive phase_steps. Received {phase_steps}.")

        # Interpret the plot command.
        return_movie = plot == 3 and test_superpixel is not None
        if return_movie:
            plot = 1
            if phase_steps is None or phase_steps == 1:
                raise ValueError(
                    "cameraslms.py: Must have phase_steps > 1 to produce a movie."
                )
        verbose = plot >= 0
        plot_fits = plot >= 1 or test_superpixel is not None
        plot_everything = plot >= 2

        # Build the calibration dict.
        calibration_dict = {
            "calibration_points" : calibration_points,
            "superpixel_size" : superpixel_size,
            "slm_supershape" : slm_supershape,
            "reference_superpixels" : reference_superpixels,
            "phase_steps" : phase_steps,
            "interference_size" : interference_size,
            "interference_window" : interference_window,
            "previous_phase_correction" : (
                False
                if self.slm.phase_correction is None else
                np.copy(self.slm.phase_correction)
            ),
            "scheduling" : scheduling,
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
            calibration_dict.update(
                {key: np.full(slm_supershape + (num_points,), np.nan, dtype=np.float32)}
            )

        def superpixels(
                schedule=None,
                reference_phase=None,
                target_phase=None,
                reference_blaze=reference_blazes,
                target_blaze=calibration_blazes,
                plot=False
            ):
            """
            Helper function for making superpixel phase masks.

            Parameters
            ----------
            schedule : list of int
                Defines which superpixels to source targets from.
            reference_phase, target_phase : float OR None
                Phase of reference/target superpixel; not rendered if None.
            reference_blaze, target_blaze : (float, float)
                Blaze vector(s) for the given superpixel.
            """
            matrix = blaze(self.slm, field_blaze)

            if reference_phase is not None:
                for i in range(num_points):
                    if schedule is None or schedule[i] != -1:
                        imprint(
                            matrix,
                            np.array([
                                reference_superpixels_coords[0, i], 1,
                                reference_superpixels_coords[1, i], 1
                            ]) * superpixel_size,
                            blaze,
                            self.slm,
                            vector=reference_blaze[:, [i]],
                            offset=reference_phase
                        )

            if target_phase is not None and schedule is not None:
                target_coords = index2coord(schedule)
                for i in range(num_points):
                    if schedule[i] != -1:
                        imprint(
                            matrix,
                            np.array([
                                target_coords[0, i], 1,
                                target_coords[1, i], 1
                            ]) * superpixel_size,
                            blaze,
                            self.slm,
                            vector=target_blaze[:, [i]],
                            offset=target_phase if np.isscalar(target_phase) else target_phase[i]
                        )

            if plot:
                plt.figure(figsize=(20, 25))
                plt.imshow(np.mod(matrix, 2 * np.pi), interpolation="none")
                plt.show()

            self.slm.write(matrix, settle=True)
            return self.cam.get_image()

        def fit_phase(phases, intensities, plot_fits=False):
            """
            Fits a sine function to the intensity vs phase, and extracts best phase and amplitude
            that give constructive interference.
            If fit fails return 0 on all values.

            Parameters
            ----------
            phases : numpy.ndarray
                Phase measurements.
            intensities : numpy.ndarray
                Intensity measurements.
            plot_fits : bool
                Whether to plot fit results.

            Returns
            -------
            best_phase : float
                b
            amp : float
                a
            r2 : float
                R^2 of fit
            contrast : float
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

            # Extract phase and amplitude from fit.
            best_phase = popt[0]
            amp = popt[1]
            contrast = popt[1] / (popt[1] + popt[2])

            # Residual and total sum of squares, producing the R^2 metric.
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

        def fit_phase_image(img, dsuperpixel, plot_fits=True):
            """
            Fits a modulated 2D sinc function to an image, and extracts best phase and
            amplitude that give constructive interference.
            If fit fails return 0 on all values.

            Parameters
            ----------
            img : numpy.ndarray
                2D image centered on the interference point.

            Returns
            -------
            best_phase : float
                b
            amp : float
                a
            r2 : float
                R^2 of fit
            contrast : float
                a / (a + c)
            """
            # Future: Cache this outside to avoid repeating memory allocation.
            xy = np.meshgrid(
                *[np.arange(-(img.shape[1-a]-1)/2, +(img.shape[1-a]-1)/2+.5) for a in range(2)]
            )
            xyr = [l.ravel() for l in xy]

            # Process dsuperpixel by rotating it according to the Fourier calibration.
            M = self.fourier_calibration["M"]
            M_norm = 2 * M / np.trace(M)            # trace is sum of eigenvalues.
            dsuperpixel = np.squeeze(np.matmul(M_norm, format_2vectors(dsuperpixel)))

            # Make the guess and bounds.
            d = np.amin(img)
            c = 0
            a = np.amax(img) - c

            R = np.mean(img.shape)/4

            guess = [
                R, a, 0, c, d,
                8 * np.pi * dsuperpixel[0] / img.shape[1],
                8 * np.pi * dsuperpixel[1] / img.shape[0]
            ]
            lb = [
                .9*R, 0, -4*np.pi, 0, 0,
                guess[5]-1,
                guess[6]-1
            ]
            ub = [
                1.1*R, 2*a, 4*np.pi, a, a,
                guess[5]+1,
                guess[6]+1
            ]

            # Restrict sinc2d to be centered (as expected).
            def sinc2d_local(xy, R, a=1, b=0, c=0, d=0, kx=1, ky=1):
                return sinc2d(xy, 0, 0, R, a, b, c, d, kx, ky)

            # Determine the guess phase by overlapping shifted guesses with the image.
            differences = []
            N = 20
            phases = np.arange(N) * 2 * np.pi / N

            for phase in phases:
                guess[2] = phase
                differences.append(np.sum(np.square(img - sinc2d_local(xy, *guess))))

            guess[2] = phases[int(np.min(np.argmin(differences)))]

            # Try the fit!
            try:
                popt, _ = optimize.curve_fit(sinc2d_local, xyr, img.ravel(), p0=guess, bounds=(lb, ub))
            except BaseException:
                return 0, 0, 0, 0

            # Extract phase and amplitude from fit.
            best_phase = popt[2]
            amp = np.abs(popt[1])
            contrast = np.abs(popt[1] / (np.abs(popt[1]) + np.abs(popt[3])))

            # Remove the sinc term when doing the rsquared.
            popt_nomod = np.copy(popt)
            popt_nomod[3] += popt_nomod[1]/2
            popt_nomod[1] = 0
            img0 = img - sinc2d_local(xy, *popt_nomod)
            fit0 = sinc2d_local(xy, *popt) - sinc2d_local(xy, *popt_nomod)

            # Residual and total sum of squares, producing the R^2 metric.
            ss_res = np.sum((img0 - fit0) ** 2)
            ss_tot = np.sum((img0 - np.mean(img0)) ** 2)
            r2 = 1 - (ss_res / ss_tot)

            final = (np.mod(-best_phase, 2*np.pi), amp, r2, contrast)

            # Plot the image, guess, and fit, if desired.
            if plot_fits:
                _, axs = plt.subplots(1, 3)

                axs[0].imshow(img)
                axs[1].imshow(sinc2d_local(xy, *guess))
                axs[2].imshow(sinc2d_local(xy, *popt))

                for index, title in enumerate(["Image", "Guess", "Fit"]):
                    axs[index].set_title(title)

                plt.show()

            return final

        def plot_labeled(schedule, img, phase=None, plot=False, title="", plot_zoom=False, focus=None):
            if plot_everything or plot:
                def plot_labeled_rects(ax, points, labels, colors, wh, hh):
                    for point, label, color in zip(points, labels, colors):
                        rect = plt.Rectangle(
                            (float(point[0] - wh/2), float(point[1] - hh/2)),
                            float(wh), float(hh),
                            ec=color, fc="none"
                        )
                        ax.add_patch(rect)
                        ax.annotate(
                            label, (point[0], point[1]),
                            c=color, size="x-small", ha="center", va="center"
                        )

                if return_movie:
                    fig, axs = plt.subplots(1, 3, figsize=(16,4), facecolor="white")
                else:
                    fig, axs = plt.subplots(1, 3, figsize=(16,4))

                # Plot phase on the first axis.
                if phase is None:
                    phase = self.slm.phase
                axs[0].imshow(
                    np.mod(phase, 2*np.pi),
                    cmap=plt.get_cmap("twilight"),
                    interpolation='none'
                )

                points = []
                labels = []
                colors = []
                center_offset = format_2vectors((superpixel_size/2, superpixel_size/2))

                for i in range(num_points):
                    if schedule is None or schedule[i] != -1:
                        if focus is None:
                            focus = i
                        points.append(reference_superpixels_coords[:, [i]] * superpixel_size + center_offset)
                        if schedule is not None: points.append(index2coord(schedule[i]) * superpixel_size + center_offset)
                        if num_points > 1:
                            labels.append("{}".format(i))
                            if schedule is not None: labels.append("{}".format(i))
                        else:
                            labels.append("Reference\nSuperpixel")
                            if schedule is not None: labels.append("Test\nSuperpixel")
                        c1 = (1 if i == focus else .5, .2, 0)
                        colors.append(c1)
                        c2 = (1 if i == focus else .5, 0, .2)
                        if schedule is not None: colors.append(c2)

                plot_labeled_rects(axs[0], points, labels, colors, superpixel_size, superpixel_size)

                # TODO: fix for multiple
                # if plot_zoom:
                #     for a in [0, 1]:
                #         ref = reference_superpixels[a] * superpixel_size
                #         test = test_superpixel[a] * superpixel_size

                #         lim = [min(ref, test) - .5 * superpixel_size, max(ref, test) + 1.5 * superpixel_size]

                #         if a:
                #             axs[0].set_ylim([lim[1], lim[0]])
                #         else:
                #             axs[0].set_xlim(lim)

                if img is not None:
                    im = axs[1].imshow(np.log10(img + .1))
                    im.set_clim(0, np.log10(self.cam.bitresolution))

                dpoint = field_point - base_point

                # Assemble points and labels.
                points = [(base_point + N * dpoint) for N in range(-2, 3)]
                labels = ["-2nd", "-1st", "0th", "1st", "2nd"]
                colors = ["b"] * 5

                focus_point = None

                for i in range(num_points):
                    if schedule is None or schedule[i] != -1:
                        points.append(calibration_points[:, [i]])
                        if num_points > 1:
                            labels.append("{}".format(i))
                        else:
                            labels.append("Calibration\nPoint")
                        c = (1 if i == focus else .5, 0, 0)
                        colors.append(c)
                        if i == focus:
                            focus_point = calibration_points[:, [i]]

                # Plot points and labels.
                wh = int(interference_window[0])
                hh = int(interference_window[1])

                plot_labeled_rects(axs[1], points, labels, colors, wh, hh)

                if img is not None:
                    im = axs[2].imshow(np.log10(img + .1))
                    im.set_clim(0, np.log10(self.cam.bitresolution))

                    if self.cam.bitdepth > 10:
                        step = 2
                    else:
                        step = 1

                    bitresolution_list = np.power(2, np.arange(0, self.cam.bitdepth+1, step))

                    cbar = fig.colorbar(im, ax=axs[2])
                    cbar.ax.set_yticks(np.log10(bitresolution_list))
                    cbar.ax.set_yticklabels(bitresolution_list)

                point = focus_point

                axs[2].scatter([point[0]], [point[1]], 5, "r", "*")
                axs[2].set_xlim(point[0] - wh/2, point[0] + wh/2)
                axs[2].set_ylim(point[1] + hh/2, point[1] - hh/2)

                # Axes coloring and colorbar.
                for spine in ["top", "bottom", "right", "left"]:
                    axs[2].spines[spine].set_color("r")
                    axs[2].spines[spine].set_linewidth(1.5)

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

        def take_interference_regions(img, integrate=True):
            """Helper function for grabbing the data at the calibration points."""
            return analysis.take(
                img,
                calibration_points,
                interference_window,
                clip=True,
                integrate=integrate
            )

        def find_centers(img):
            """Helper function for finding the center of images around the calibration points."""
            imgs = take_interference_regions(img, integrate=False)
            return analysis.image_positions(imgs) + calibration_points

        def measure(schedule, plot=False):
            self.cam.flush()

            # Step 0: Measure the background.
            if measure_background:
                back_image = superpixels(schedule, None, None)
                plot_labeled(schedule, back_image, plot=plot, title="Background")
                back = take_interference_regions(back_image)
            else:
                back = [np.nan] * num_points

            # Step 0.5: Measure the power in the reference mode.
            norm_image = superpixels(schedule, 0, None)
            plot_labeled(schedule, norm_image, plot=plot, title="Reference Diffraction")
            norm = take_interference_regions(norm_image)

            # Step 1: Measure the position of the target mode.
            position_image = superpixels(schedule, None, 0)
            plot_labeled(schedule, position_image, plot=plot, title="Base Target Diffraction")
            found_centers = find_centers(position_image)

            # Step 1.25: Add a blaze to the target mode so that it overlaps with reference mode.
            blaze_differences = self.ijcam_to_kxyslm(found_centers) - calibration_blazes
            target_blaze_fixed = calibration_blazes - blaze_differences

            # Step 1.5: Measure the power...
            if corrected_amplitude:      # ...in the corrected target mode.
                fixed_image = superpixels(schedule, None, 0, target_blaze=target_blaze_fixed)
                plot_labeled(schedule, fixed_image, plot=plot, title="Corrected Target Diffraction")
                pwr = take_interference_regions(fixed_image)
            else:                       # ...in the uncorrected target mode.
                pwr = take_interference_regions(position_image)

            # Step 1.75: Stop here if we don't need to measure the phase (only save powers).
            if phase_steps is None:
                return {
                    "power": pwr,
                    "normalization": norm,
                    "background": back,
                    "phase": [np.nan] * num_points,
                    "kx": [np.nan] * num_points,
                    "ky": [np.nan] * num_points,
                    "amp_fit": [np.nan] * num_points,
                    "contrast_fit": [np.nan] * num_points,
                    "r2_fit": [np.nan] * num_points,
                }
            
            results = []
            first_index = np.where(schedule != -1)[0][0]

            # Step 2: Measure interference and find relative phase TODO: vectorize
            if phase_steps == 1:
                # Step 2.1: Gather a single image.
                result_img = superpixels(schedule, 0, 0, target_blaze=target_blaze_fixed)
                cropped_img = take_interference_regions(result_img, integrate=False)

                # Step 2.2: Fit the data and return.
                coord_difference = index2coord(schedule) - index2coord(reference_superpixels)

                results = [
                    (
                        fit_phase_image(
                            cropped_img[i],
                            coord_difference[:,i],
                            plot_fits=plot and i == first_index
                        ) 
                        if schedule[i] != -1 else
                        [np.nan] * 4
                    )
                    for i in range(num_points)
                ]
            else:
                # Gather multiple images at different phase offsets.
                phases = np.linspace(0, 2 * np.pi, phase_steps, endpoint=False)
                iresults = []  # list for recording the intensity of the reference point

                # Determine whether to use a progress bar.
                if verbose:
                    description = "phase_measurement"
                    prange = tqdm(phases, position=0, leave=False, desc=description)
                else:
                    prange = phases

                if return_movie: frames = []

                # Step 2.1: Measure phases
                for phase in prange:
                    interference_image = superpixels(schedule, 0, phase, target_blaze=target_blaze_fixed)
                    iresults.append(
                        [
                            interference_image[calibration_points[1, i], calibration_points[0, i]]
                            for i in range(num_points)
                        ]
                    )

                    if return_movie:
                        frames.append(
                            plot_labeled(
                                schedule, 
                                interference_image,
                                plot=plot,
                                title="Phase = ${:1.2f}\pi$".format(phase / np.pi),
                                plot_zoom=True
                            )
                        )

                iresults = np.array(iresults)

                # Step 2.2: Fit to sine and return.
                for i in range(num_points):
                    results.append(fit_phase(phases, iresults[:, i], plot and i == first_index))

            results = np.array(results)

            phase_fit =     results[:, 0]
            amp_fit =       results[:, 1]
            contrast_fit =  results[:, 2]
            r2_fit =        results[:, 3]

            # Step 2.5: maybe plot a picture of the correct phase.
            if plot:
                interference_image = superpixels(schedule, 0, phase_fit, target_blaze=target_blaze_fixed)
                plot_labeled(schedule, interference_image, plot=plot, title="Best Interference")

            # Step 3: Return the result.
            if return_movie: return frames

            return {
                "power": pwr,
                "normalization": norm,
                "background": back,
                "phase": phase_fit,
                "kx": -blaze_differences[0, :],
                "ky": -blaze_differences[1, :],
                "amp_fit": amp_fit,
                "contrast_fit": contrast_fit,
                "r2_fit": r2_fit,
            }

        # Correct exposure and position of the reference mode(s).
        self.cam.flush()
        base_image = superpixels(None, 0, None)
        plot_labeled(None, base_image, plot=plot_everything, title="Base Reference Diffraction")
        found_centers = find_centers(base_image)

        # Correct the original blaze using the measured result.
        reference_blaze_differences = self.ijcam_to_kxyslm(found_centers) - reference_blazes
        np.subtract(reference_blazes, reference_blaze_differences, out=reference_blazes)

        if plot_fits:
            fixed_image = superpixels(None, 0, None)
            plot_labeled(None, fixed_image, plot=plot_everything, title="Corrected Reference Diffraction")

        # If we just want to debug/test one region, then do so.
        if test_superpixel is not None:
            result = measure(scheduling[:, test_superpixel], plot=plot_fits)

            # Reset the phase and amplitude of the SLM to the stored data.
            self.slm.measured_amplitude = measured_amplitude
            self.slm.phase_correction = phase_correction

            return result

        # Proceed with all of the superpixels.
        for n in tqdm(range(num_measurements), position=1, leave=True, desc="calibration"):
            schedule = scheduling[:, n]

            # Measure!
            measurement = measure(schedule)

            # Update dictionary.
            coords = index2coord(schedule)
            for i in range(num_points):
                if schedule[i] != -1:
                    for key in measurement:
                        calibration_dict[key][coords[1, i], coords[0, i], i] = measurement[key][i]

        self.wavefront_calibration_raw = calibration_dict

        return calibration_dict

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
        back[np.isnan(back)] = 0
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

        # Add the old phase correction if it's there.
        if (
            "previous_phase_correction" in data and
            data["previous_phase_correction"] is not None
        ):
            phase_fin_fin = phase_fin + data["previous_phase_correction"]
        else:
            phase_fin_fin = phase_fin

        # Build the final dict.
        wavefront_calibration = {
            "phase_correction":phase_fin_fin,
            "measured_amplitude":amp_large,
            "r2":r2
        }

        # Step 4: Load the correction to the SLM
        if apply:
            self.slm.phase_correction = phase_fin_fin
            self.slm.measured_amplitude = amp_large

        # Plot the result
        if plot:
            plt.figure(figsize=(16, 8))

            plt.subplot(1, 3, 1)
            plt.imshow(
                phase_fin,
                clim=(0,2*np.pi),
                cmap=plt.get_cmap("twilight"),
                interpolation="none",
            )
            plt.title("SLM Flatfield Phase Correction")
            plt.xlabel("SLM $x$ [pix]")
            plt.ylabel("SLM $y$ [pix]")

            plt.subplot(1, 3, 2)
            plt.imshow(amp_large, clim=(0,1))
            plt.title("Measured Beam Amplitude")
            plt.xlabel("SLM $x$ [pix]")
            plt.ylabel("SLM $y$ [pix]")

            plt.subplot(1, 3, 3)
            plt.imshow(r2, clim=(0,1))
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
