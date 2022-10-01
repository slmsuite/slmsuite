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
        max_phase = 2 * np.pi

        self.slm_number = int(slm_number)
        self.display_number = int(display_number)

        # Default wavelength is 0.780um.
        wav_um = kwargs.pop("wav_um", 0.780)

        if verbose:
            print("Santec initializing... ", end="")
        assert (
            slm_funcs.SLM_Ctrl_Open(self.slm_number) == 0
        ), "Opening Santec slm_number={} failed.".format(self.slm_number)
        slm_funcs.SLM_Ctrl_WriteVI(self.slm_number, 1)  # 0:Memory 1:DVI
        if verbose:
            print("success")

        # Update wavelength if needed
        wav_current = ctypes.c_uint32(0)
        phase_current = ctypes.c_ulong(0)
        slm_funcs.SLM_Ctrl_ReadWL(self.slm_number, wav_current, phase_current)

        wav_desired = int(1e3 * wav_um)
        phase_desired = int(np.floor(max_phase * 100 / np.pi))

        if wav_current.value != 1e3 * wav_um:
            if verbose:
                print("Current wavelength table: wav = {0} nm, maxphase = {1:.2f}pi"
                    .format(wav_current.value, phase_current.value / 100.0))
                print("Desired wavelength table: wav = {0} nm, maxphase = {1:.2f}pi"
                        .format(wav_desired, phase_desired / 100.0))
                print("     ...Updating wavelength table (this may take 30 seconds)...")

            # Set wavelength (nm) and maximum phase (100 * number pi)
            slm_funcs.SLM_Ctrl_WriteWL(self.slm_number, wav_desired, phase_desired)
            # Save wavelength
            slm_funcs.SLM_Ctrl_WriteAW(self.slm_number)

            # Verify wavelength
            slm_funcs.SLM_Ctrl_ReadWL(self.slm_number, wav_current, phase_current)
            if verbose:
                print("Updated wavelength table: wav = {0} nm, maxphase = {1:.2f}pi"
                        .format(wav_current.value, phase_current.value / 100.0))

        # Check for the SLM parameters and save them
        width = ctypes.c_ushort(0)
        height = ctypes.c_ushort(0)
        display_name = ctypes.create_string_buffer(64)

        if verbose:
            print("Looking for display... ", end="")
        slm_funcs.SLM_Disp_Info2(self.display_number, width, height, display_name)

        # For instance, "LCOS-SLM,SOC,8001,2018021001"
        name = display_name.value.decode("mbcs")
        if verbose:
            print("success")

        names = name.split(",")

        # If the target display_number is not found, then print some debug:
        if names[0] != "LCOS-SLM":
            slm_funcs.SLM_Ctrl_Close(self.slm_number)

            raise ValueError(   "SLM not found at display_number {}. " \
                                "Use .info() to find the correct display!" \
                                .format(self.display_number)    )

        # Open SLM
        if verbose:
            print("Opening {}... ".format(name), end="")
        slm_funcs.SLM_Disp_Open(self.display_number)
        if verbose:
            print("success")

        super().__init__(
            int(width.value),
            int(height.value),
            bitdepth=10,
            name="Santec_" + names[-1],
            wav_um=wav_um,
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
        width = ctypes.c_ushort(0)
        height = ctypes.c_ushort(0)
        display_name = ctypes.create_string_buffer(64)

        display_list = []

        if verbose:
            print("display_number, display_name:")
            print("#,  Name")

        for display_number in range(1, 8):
            slm_funcs.SLM_Disp_Info2(display_number, width, height, display_name)
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
        matrix = phase.astype(ctypes.c_uint16)

        n_h, n_w = self.shape

        # Write to SLM
        c = matrix.ctypes.data_as(ctypes.POINTER((ctypes.c_int16 * n_h) * n_w)).contents
        ret = slm_funcs.SLM_Disp_Data(self.display_number, n_w, n_h, 0, c)
        if ret != 0:
            raise RuntimeError("SLM returned error {}.".format(ret))

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
