"""
Hardware control for Santec SLMs.
Santec LCoS SLM-200, SLM-210, SLM-300, ...

Note
~~~~
:class:`.Santec` requires files from Santec to be present in the
:mod:`~slmsuite.hardware.slms` directory:

 - A header file (_slm_win.py) and
 - Dynamically linked libraries (SLMFunc.dll and FTD3XX.dll).

These files should be copied in before use.

Note
~~~~
Santec provides base wavefront correction accounting for the curvature of the SLM surface.
Consider loading these files via :meth:`.SLM..load_vendor_phase_correction()`
"""
import os
import ctypes
import numpy as np
import cv2
import warnings

from .slm import SLM

if hasattr(os, "add_dll_directory"):
    try:
        # Python >= 3.8 requires the search path for .dll loading.
        os.add_dll_directory(os.path.dirname(os.path.abspath(__file__)))
        # Load Santec's header file.
        from . import _slm_win as slm_funcs
    except BaseException as e:
        # Provide an informative error should something go wrong.
        print("Santec DLLs not installed. Install these to use Santec SLMs.")
        print(  "  Files from Santec must be present in the slms directory:\n"
                "  - A header file (_slm_win.py) and\n"
                "  - Dynamically linked libraries (SLMFunc.dll and FTD3XX.dll).\n"
                "Check that theses files are present and are error-free.\n{}".format(e))
else:
    print("santec.py: os has no attribute add_dll_directory.")


class Santec(SLM):
    """
    Interfaces with Santec SLMs.

    Attributes
    ----------
    slm_number : int
        USB port number assigned by Santec SDK.
    display_number : int
        Display number assigned by Santec SDK.
    optionboard_id : str
        ID of the option board
    driveboard_id : str
        ID of the drive board
    product_code_id : str
        Product code of the device
    """

    def __init__(self, slm_number=1, display_number=2, verbose=True, **kwargs):
        """
        Initializes an instance of a Santec SLM.

        Arguments
        ------
        slm_number
            See :attr:`slm_number`.
        display_number
            See :attr:`display_number`.
        verbose : bool
            Whether to print extra information.
        kwargs
            See :meth:`.SLM.__init__` for permissible options.

        Note
        ----
        Santec sets the voltage lookup table to correspond to the wavelength
        supplied to :attr:`.SLM.wav_um`. This allows :attr:`.SLM.phase_scaling` to
        always be one, and make use of optimized routines (see :meth`.write()`)

        Caution
        ~~~~~~~
        :class:`.Santec` defaults to 8 micron SLM pixel size
        (:attr:`.SLM.dx_um` = :attr:`.SLM.dy_um` = 8)
        and 10-bit :attr:`.SLM.bitdepth`.
        This is valid for SLM-200, SLM-210, and SLM-300, but may not be valid for future
        Santec models.
        """
        # Default max phase. Maybe this should be opened to the user in the future.
        # Otherwise, wav_um and wav_design_um have the same functionality.
        max_phase = 2 * np.pi

        self.slm_number = int(slm_number)
        self.display_number = int(display_number)

        # Default wavelength is 0.780um.
        wav_um = kwargs.pop("wav_um", 0.780)

        # By default, target wavelength is the design wavelength
        wav_design_um = kwargs.pop("wav_design_um", None)
        if wav_design_um is None:
            wav_design_um = wav_um

        if verbose:
            print("Santec slm_number={} initializing... ".format(self.slm_number), end="")
        self.parse_status(slm_funcs.SLM_Ctrl_Open(self.slm_number))
        self.parse_status(slm_funcs.SLM_Ctrl_WriteVI(self.slm_number, 1))   # 0:Memory 1:DVI
        if verbose:
            print("success")

        # Update wavelength if needed
        wav_current_nm = ctypes.c_uint32(0)
        phase_current = ctypes.c_ulong(0)
        self.parse_status(
            slm_funcs.SLM_Ctrl_ReadWL(self.slm_number, wav_current_nm, phase_current)
        )

        wav_desired_nm = int(1e3 * wav_design_um)
        phase_desired = int(np.floor(max_phase * 100 / np.pi))

        if wav_current_nm.value != wav_desired_nm:
            if verbose:
                print("Current wavelength table: wav = {0} nm, maxphase = {1:.2f}pi"
                    .format(wav_current_nm.value, phase_current.value / 100.0))
                print("Desired wavelength table: wav = {0} nm, maxphase = {1:.2f}pi"
                        .format(wav_desired_nm, phase_desired / 100.0))
                print("     ...Updating wavelength table (this may take 30 seconds)...")

            # Set wavelength (nm) and maximum phase (100 * [float pi])
            self.parse_status(
                slm_funcs.SLM_Ctrl_WriteWL(self.slm_number, wav_desired_nm, phase_desired)
            )
            # Save wavelength
            self.parse_status(slm_funcs.SLM_Ctrl_WriteAW(self.slm_number))

            # Verify wavelength
            self.parse_status(
                slm_funcs.SLM_Ctrl_ReadWL(self.slm_number, wav_current_nm, phase_current)
            )
            if verbose:
                print("Updated wavelength table: wav = {0} nm, maxphase = {1:.2f}pi"
                        .format(wav_current_nm.value, phase_current.value / 100.0))

        # Check for the SLM parameters and save them
        width = ctypes.c_ushort(0)
        height = ctypes.c_ushort(0)
        display_name = ctypes.create_string_buffer(128)

        if verbose:
            print("Looking for display... ", end="")
        self.parse_status(
            slm_funcs.SLM_Disp_Info2(self.display_number, width, height, display_name)
        )

        # For instance, "LCOS-SLM,SOC,8001,2018021001"
        # Format is "UserFriendlyName,ManufacterName,ProductCodeID,SerialNumberID"
        name = display_name.value.decode("mbcs")
        if verbose:
            print("success")

        names = name.split(",")

        # If the target display_number is not found, then print some debug:
        if names[0] != "LCOS-SLM":
            # Don't parse status around this one...
            slm_funcs.SLM_Ctrl_Close(self.slm_number)

            raise ValueError(   "SLM not found at display_number={}. " \
                                "Use .info() to find the correct display!" \
                                .format(self.display_number)    )

        # Populate some info
        driveboard_id = ctypes.create_string_buffer(16)
        optionboard_id = ctypes.create_string_buffer(16)
        self.parse_status(
            slm_funcs.SLM_Ctrl_ReadSDO(self.slm_number, driveboard_id, optionboard_id)
        )

        self.driveboard_id = driveboard_id.value.decode("mbcs")
        self.optionboard_id = optionboard_id.value.decode("mbcs")
        self.product_code_id = names[2]

        # Open SLM
        if verbose:
            print("Opening {}... ".format(name), end="")
        self.parse_status(slm_funcs.SLM_Disp_Open(self.display_number))
        if verbose:
            print("success")

        super().__init__(
            int(width.value),
            int(height.value),
            bitdepth=10,
            name=names[-1], # SerialNumberID
            wav_um=wav_um,
            wav_design_um=wav_design_um,
            dx_um=8,
            dy_um=8,
            **kwargs
        )

        self.write(None)

    @staticmethod
    def info(verbose=True):
        """
        Discovers the names of all the displays.
        Checks all 8 possible supported by Santec's SDK.

        Parameters
        ----------
        verbose : bool
            Whether to print the discovered information.

        Returns
        --------
        list of (int, str) tuples
            The number and name of each SLM.
        """
        # Check for the SLM parameters and save them
        display_list = []

        if verbose:
            print("display_number, display_name:")
            print("#,  Name")

        for display_number in range(1, 9):
            width = ctypes.c_ushort(0)
            height = ctypes.c_ushort(0)
            display_name = ctypes.create_string_buffer(128)

            self.parse_status(
                slm_funcs.SLM_Disp_Info2(display_number, width, height, display_name)
            )
            name = display_name.value.decode("mbcs")
            if verbose:
                print("{},  {}".format(display_number, name))

            display_list.append((display_number, name))

        return display_list

    def load_vendor_phase_correction(self, file_path, wav_correction_um=0.830, smooth=False):
        """
        Load phase correction provided by Santec from file,
        setting :attr:`~slmsuite.hardware.slms.slm.SLM.phase_correction`.

        Parameters
        ----------
        file_path : str
            File path for the vendor-provided phase correction.
        wav_correction_um : float
            The wavelength the phase correction was taken at in um. Default is 830nm.
        smooth : bool
            Whether to apply a Gaussian blur to smooth the data.

        Returns
        ----------
        numpy.ndarray
            :attr:`~slmsuite.hardware.slms.slm.SLM.phase_correction`,
            the Santec-provided phase correction.
        """
        try:
            # Load from .csv, skipping the first row and column
            # (corresponding to X and Y coordinates).
            map = np.loadtxt(file_path, skiprows=1, dtype=int, delimiter=",")[:, 1:]
            phase = (-2 * np.pi / self.bitresolution * self.wav_um / wav_correction_um) * map.astype(float)

            # Smooth the map
            if smooth:
                real = np.cos(phase)
                imag = np.sin(phase)
                size_blur = 15

                real = cv2.GaussianBlur(real, (size_blur, size_blur), 0)
                imag = cv2.GaussianBlur(imag, (size_blur, size_blur), 0)

                # Recombine the components
                phase = np.arctan2(imag, real) + np.pi

            self.phase_correction = phase
        except BaseException as e:
            print("Error while loading phase correction.\n{}".format(e))

        return self.phase_correction

    def close(self):
        """See :meth:`.SLM.close`."""
        slm_funcs.SLM_Disp_Close(self.display_number)
        slm_funcs.SLM_Ctrl_Close(self.slm_number)

    def _write_hw(self, phase):
        """See :meth:`.SLM._write_hw`."""
        matrix = phase.astype(slm_funcs.USHORT)

        n_h, n_w = self.shape

        # Write to SLM
        c = matrix.ctypes.data_as(ctypes.POINTER((slm_funcs.USHORT * n_h) * n_w)).contents
        self.parse_status(slm_funcs.SLM_Disp_Data(self.display_number, n_w, n_h, 0, c))

    # Extra functionality

    def read_temperature(self):
        """
        **(Untested)**
        Read the drive board and option board temperatures.

        Returns
        -------
        (float, float)
            Temperature in Celsius of the drive and option board
        """
        drive_temp = ctypes.c_int32(0)
        option_temp = ctypes.c_int32(0)

        self.parse_status(slm_funcs.SLM_Ctrl_ReadT(self.slm_number, drive_temp, option_temp))

        return (drive_temp.value / 10., option_temp.value / 10.)

    def check_error(self, raise_error=True):
        """
        **(Untested)**
        Read the drive board and option board errors.

        Returns
        -------
        list of str
            List of errors.
        """
        drive_error = ctypes.c_uint32(0)
        option_error = ctypes.c_uint32(0)

        slm_funcs.SLM_Ctrl_ReadEDO(self.slm_number, drive_error, option_error)

        # Check the resulting bitstrings for errors (0 ==> all good).
        errors = []

        for drive_error_bit in slm_funcs.SLM_DRIVEBOARD_ERROR.keys():
            if drive_error.value & drive_error_bit:
                errors.append(slm_funcs.SLM_DRIVEBOARD_ERROR[drive_error_bit])

        for option_error_bit in slm_funcs.SLM_OPTIONBOARD_ERROR.keys():
            if option_error.value & option_error_bit:
                errors.append(slm_funcs.SLM_OPTIONBOARD_ERROR[option_error_bit])

        if len(errors) > 0:
            raise RuntimeError("Santec error: " + ", ".join(["'" + err + "'" for err in errors]))

        return errors

    def parse_status(self, status=None, raise_error=True):
        """
        **(Untested)**
        Parses the meaning of a ``SLM_STATUS`` return from the Santec SLM.

        Parameters
        ----------
        status : int
            ``SLM_STATUS`` return. If ``None``, the SLM is polled for status instead.
        raise_error : bool
            Whether to raise an error (if True) or a warning (if False) when status is not ``SLM_OK``.

        Returns
        -------
        (str, str)
            Status in ``(name, note)`` form.
        """
        # Parse status
        if status is None:
            status = slm_funcs.SLM_Ctrl_ReadSU(self.slm_number)

        status = int(status)

        if not status in slm_funcs.SLM_STATUS_DICT.keys():
            raise ValueError("SLM status '{}' not recognized.".format(status))

        # Recover the meaning of status
        (name, note) = slm_funcs.SLM_STATUS_DICT[status]

        status_str = "Santec error {}; '{}'".format(name, note)

        if status != 0:
            if raise_error:
                raise RuntimeError(status_str)
            else:
                warnings.warn(status_str)

        return (name, note)

    def write_csv(self, filename):
        """
        Write the phase image contained in a .csv file to the SLM.
        This image should have the size of the SLM.

        Parameters
        ----------
        filename : str
            Path to the .csv file.
        """
        slm_funcs.SLM_Disp_ReadCSV(self.display_number, 0, filename)
