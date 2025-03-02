"""
Hardware control for Meadowlark SLMs.

Tested with:
- Meadowlark P1920-400-800-HDMI-T.
- Meadowlark HSPDM512-1064-PCIe8-ODP
- Meadowlark HSP1920-1064-HSP8

Note
~~~~
Check that the Blink SDK, including DLL files etc, are in the default folder
or otherwise pass the correct directory in the constructor.
"""

import os
import ctypes
import warnings
from enum import IntEnum
from pathlib import Path
import numpy as np
from typing import Callable
from slmsuite.hardware.slms.slm import SLM
from functools import partial
from platform import system

#: str: Default location in which Meadowlark Optics software is installed
_DEFAULT_MEADOWLARK_PATH = "C:\\Program Files\\Meadowlark Optics\\"

class _SDK_MODE(IntEnum):
    #: No connection
    NULL = 0
    #: HDMI connection
    HDMI = 1
    #: High-Speed PCIe connection (512x512 Models Only)
    PCIE_LEGACY = 2
    #: High-Speed PCIe connection
    PCIE_MODERN = 3


class Meadowlark(SLM):
    """
    Interfaces with Meadowlark SLMs.

    Attributes
    ----------
    slm_lib : ctypes.CDLL
        Connection to the Meadowlark library.
    sdk_path : str
        Path of the Blink SDK folder.
    """

    def __init__(  # noqa: R0913, R0917
        self,
        slm_number: int = 1,
        sdk_path: str | None = None,
        lut_path: str | None = None,
        wav_um: float = 1,
        pitch_um: tuple[float, float] | None = None,
        verbose: bool = True,
        **kwargs,
    ):
        r"""
        Initializes an instance of a Meadowlark SLM.

        Caution
        ~~~~~~~
        :class:`.Meadowlark` defaults to 8 micron SLM pixel size.
        This is valid for most Meadowlark models, but not true for all!

        Arguments
        ---------
        verbose : bool
            Whether to print extra information.
        slm_number : int
            The board number of the SLM to connect to, in the case of PCIe SLMs.
            Defaults to 1.
        sdk_path : str
            Path of the Blink SDK installation folder. Stored in :attr:`sdk_path`.

            Important
            ~~~~~~~~~
            If the installation is not in the default folder,
            then this path needs to be specified
        lut_path : str OR None
            Passed to :meth:`load_lut`. Looks for the voltage 'look-up table' data
            which is necessary to run the SLM.

            Tip
            ~~~
            See :meth:`load_lut` for how the default
            argument and other options are parsed.
        wav_um : float
            Wavelength of operation in microns. Defaults to 1 um.
        pitch_um : (float, float) OR None
            Pixel pitch in microns. Defaults to 8 micron square pixels.
        **kwargs
            See :meth:`.SLM.__init__` for permissible options.
        """
        # Locate the blink wrapper file in the sdk_path. If no sdk_path is provided,
        # search for the most recently installed SDK (if it exists)
        sdk_path = sdk_path or _DEFAULT_MEADOWLARK_PATH
        self.sdk_path: str = self._locate_blink_c_wrapper(sdk_path)
        self.slm_number: ctypes.c_int = ctypes.c_int(int(slm_number))
        self._number_of_boards: ctypes.c_uint(0) = ctypes.c_uint(0)
        self._sdk_mode: _SDK_MODE = _SDK_MODE.NULL

        # Validates the DPI awareness of this context, which is presumably important
        # for scaling.
        if verbose:
            print("Validating DPI awareness...", end="")

        awareness = ctypes.c_int()
        error_get = ctypes.windll.shcore.GetProcessDpiAwareness(
            0, ctypes.byref(awareness)
        )
        error_set = ctypes.windll.shcore.SetProcessDpiAwareness(2)
        success = ctypes.windll.user32.SetProcessDPIAware()

        if not success:
            raise RuntimeError(
                "Meadowlark failed to validate DPI awareness. "
                f"Errors: get={error_get}, set={error_set}, awareness="
                f"{awareness.value}"
            )

        if verbose:
            print("success")

        load_slm = partial(self._load_slm, verbose=verbose)
        # Open & Initialize the SDK.
        if load_slm(self._load_pcie_modern):
            self._sdk_mode = _SDK_MODE.PCIE_MODERN
        elif load_slm(self._load_pcie_legacy):
            self._sdk_mode = _SDK_MODE.PCIE_LEGACY
        elif load_slm(self._load_hdmi):
            self._sdk_mode = _SDK_MODE.HDMI
        else:
            raise RuntimeError("Failed to initialize Meadowlark SDK")
        if verbose:
            print(f"Using {self._sdk_mode.name} SDK")

        # Adjust pre- and post-ramp slopes for accurate voltage setting
        # (otherwise, custom LUT calibration is not properly implemented [this feature
        # is not implemented in slmsuite]). You may need a special version of the SDK
        # sent to you from Meadowlark to have access to these parameters.
        # self.slm_lib.SetPreRampSlope(20) # default is 7
        # self.slm_lib.SetPostRampSlope(24) # default is 24

        if verbose:
            print("success")

        # If using a legacy 512x512, the 'true frames' need to be set
        if self._sdk_mode == _SDK_MODE.PCIE_LEGACY:
            self.slm_lib.Set_true_frames(ctypes.c_int(3))

        # Load LUT.
        if verbose:
            print("Loading LUT file...", end="")
        true_lut_path = self.load_lut(lut_path)
        if verbose and true_lut_path != lut_path:
            print(f"success\n(loaded from '{true_lut_path}')")

        # If using a legacy 512x512, then the SLM needs to be powered on
        if self._sdk_mode == _SDK_MODE.PCIE_LEGACY:
            self.slm_lib.SLM_power(ctypes.c_bool(True))

        # Construct other variables.
        super().__init__(
            (self._get_width(), self._get_height()),
            bitdepth=self._get_bitdepth(),
            name=kwargs.pop("name", "Meadowlark"),
            wav_um=wav_um,
            pitch_um=(pitch_um if pitch_um else self._get_pitch()),
            **kwargs,
        )

        if self.bitdepth > 8:
            warnings.warn(
                f"Bitdepth of {self.bitdepth} > 8 detected; "
                "this has not been tested and might fail.",
                stacklevel=2,
            )

        self.set_phase(None)

    def load_lut(self, lut_path: str | None = None) -> str:
        """
        Loads a voltage 'look-up table' (LUT) to the SLM.
        This converts requested phase values to physical voltage perturbing
        the liquid crystals.

        Parameters
        ----------
        lut_path : str OR None
            Path to look for an LUT file in.

            -   If this is a .lut file, then this file is loaded to the SLM.
            -   If this is a directory, then searches all files inside the
                directory, and loads the most recently created .lut file or if possible
                the .lut file starting with ``"slm"`` which is more likely to
                correspond to the LUT customized to an SLM, as Meadowlark sends such
                files prefixed by ``"slm"`` such as ``"slm5758_at532.lut"``.

        Raises
        ------
        RuntimeError
            If the LUT file fails to load
        FileNotFoundError
            If no .lut files are found in provided path or the specified file does not
            exist.
        Returns
        -------
        str
            The path which was used to load the LUT.
        """
        # If no lut path is provided, we must search for it...

        # REVIEW: To facilitate selection of the correct LUT in the case where
        #  multiple different SLMs are installed, I'm using the dimensions of the SLM
        #  to narrow down the search.
        slm_dims = (self._get_width(), self._get_height())
        _locate_lut_file = partial(self._locate_lut_file, slm_dims=slm_dims)

        # None, default to the SDK path. Use parent since SDK path has
        # 'Blink_C_Wrapper.dll' appended
        if lut_path is None:
            lut_path = Path(self.sdk_path).parent

        # Find the exact LUT file
        if os.path.isdir(lut_path):
            try:
                lut_path = self._locate_lut_file(lut_path)
            except FileNotFoundError:
                # If no LUT file is file, try the parent directory as sometimes the LUTs
                # are in the parent directory of the SDK path in 'LUT'
                lut_path = self._locate_lut_file(Path(lut_path).parent)

        # If the search path doesn't exist, short circuit
        if not os.path.exists(lut_path):
            raise FileNotFoundError(f"Failed to locate LUT file: {lut_path}")

        # Finally, actually load the LUT file
        try:
            if self._sdk_mode == _SDK_MODE.HDMI:
                self.slm_lib.Load_LUT(lut_path)
            elif (
                self._sdk_mode == _SDK_MODE.PCIE_LEGACY
                or self._sdk_mode == _SDK_MODE.PCIE_MODERN
            ):
                EXIT_SUCCESS = self.slm_lib.Load_LUT_file(
                    self.slm_number, lut_path.encode("utf-8")
                )
                assert EXIT_SUCCESS == 0
            else:
                raise AssertionError("Failed to load LUT file due to unknown SDK mode")
        except AssertionError as exc:
            raise RuntimeError(f"Failed to load LUT file: {lut_path}") from exc
        else:
            return lut_path

    def close(self) -> None:
        """
        See :meth:`.SLM.close`.
        """
        # If using a legacy 512x512, the SLM needs to be powered off
        if self._sdk_mode == _SDK_MODE.PCIE_LEGACY:
            self.slm_lib.SLM_power(ctypes.c_bool(False))

        self.slm_lib.Delete_SDK()

    def info(self, verbose: bool = True) -> list[tuple[int, str]]:
        """
        Discover all connected SLMs

        Parameters
        ----------
        verbose : bool
            Whether to print the information.discovered

        Returns
        -------
        list
            The number and a descriptive string for each potential SLM.

        Raises
        ------
        NotImplementedError
            If multiple SLMs are not supported for this SDK
        """
        if (
            self._sdk_mode == _SDK_MODE.PCIE_LEGACY
            or self._sdk_mode == _SDK_MODE.PCIE_MODERN
        ):
            info = [
                (board, f"{self._get_width(board)}x{self._get_height(board)}")
                for board in range(1, self._number_of_boards.value + 1)
            ]
            if verbose:
                for board, dims in info:
                    print(f"SLM {board}: {dims}")
            return info
        else:
            raise NotImplementedError("Multiple SLMs not supported for this SDK")

    def _get_width(self, slm_number: int | None = None) -> int:
        """
        Get the width of the SLM.

        Returns
        -------
        int
            The width of the SLM.

        Raises
        ------
        NotImplementedError
            If the width retrieval is not supported for the SLM.
        """
        slm_number = ctypes.c_int(slm_number) if slm_number else self.slm_number
        if self._sdk_mode == _SDK_MODE.HDMI:
            return self.slm_lib.Get_Width()
        elif (
            self._sdk_mode == _SDK_MODE.PCIE_LEGACY
            or self._sdk_mode == _SDK_MODE.PCIE_MODERN
        ):
            return self.slm_lib.Get_image_width(slm_number)
        else:
            raise NotImplementedError("Width retrieval not supported for this model")

    def _get_height(self, slm_number: int | None = None) -> int:
        """
        Get the height of the SLM

        Returns
        -------
        int
            The height of the SLM.

        Raises
        ------
        NotImplementedError
            If the height retrieval is not supported for the SLM.
        """
        slm_number = ctypes.c_int(slm_number) if slm_number else self.slm_number
        if self._sdk_mode == _SDK_MODE.HDMI:
            return self.slm_lib.Get_Height()
        elif (
            self._sdk_mode == _SDK_MODE.PCIE_LEGACY
            or self._sdk_mode == _SDK_MODE.PCIE_MODERN
        ):
            return self.slm_lib.Get_image_height(slm_number)
        else:
            raise NotImplementedError("Height retrieval not supported for this model")

    def _get_bitdepth(self, slm_number: int | None = None) -> int:
        """
        Get the image depth of the SLM.

        Returns
        -------
        int
            The image depth of the SLM.

        Raises
        ------
        NotImplementedError
            If the image depth retrieval is not supported for the SLM.
        """
        slm_number = ctypes.c_int(slm_number) if slm_number else self.slm_number
        if self._sdk_mode == _SDK_MODE.HDMI:
            return self.slm_lib.Get_Depth()
        elif (
            self._sdk_mode == _SDK_MODE.PCIE_LEGACY
            or self._sdk_mode == _SDK_MODE.PCIE_MODERN
        ):
            return self.slm_lib.Get_image_depth(slm_number)
        else:
            raise NotImplementedError(
                "Image depth retrieval not supported for this model"
            )

    def _get_pitch(self) -> tuple[float, float]:
        """
        Get the pitch of the SLM.

        Returns
        -------
        tuple[float, float]
            The pitch of the SLM. Defaults to (8,8) if this method is not supported.
        """
        try:
            if (
                self._sdk_mode == _SDK_MODE.PCIE_LEGACY
                or self._sdk_mode == _SDK_MODE.PCIE_MODERN
            ):
                self.slm_lib.Get_pitch.restype = ctypes.c_double
                pitch = self.slm_lib.Get_pitch(self.slm_number)
                return pitch, pitch
            else:
                raise NotImplementedError(
                    "Pitch retrieval not supported for this model"
                )
        except (NotImplementedError, AttributeError):
            return 8, 8

    def get_last_error_message(self) -> str:
        """
        Get the last error message from the SLM.

        Returns
        -------
        str
            The last error message.

        Raises
        ------
        NotImplementedError
            If the error message retrieval is not supported for the SLM.
        """
        if (
            self._sdk_mode == _SDK_MODE.PCIE_LEGACY
            or self._sdk_mode == _SDK_MODE.PCIE_MODERN
        ):
            self.slm_lib.Get_last_error_message.restype = ctypes.c_char_p
            return self.slm_lib.Get_last_error_message().decode("utf-8")
        else:
            raise NotImplementedError(
                "Error message retrieval not supported for this model"
            )

    def get_version_info(self) -> str:
        """
        Get the version information of the SLM.

        Returns
        -------
        str
            Version information.
        """
        self.slm_lib.Get_version_info.restype = ctypes.c_char_p
        return self.slm_lib.Get_version_info().decode("utf-8")

    def get_temperature(self) -> float:
        """
        Read the temperature of the SLM.

        Returns
        -------
        float
            Temperature in degrees celcius.

        Raises
        ------
        Not Implemented Error
            If the temperature reading is not supported for the SLM.
        """
        if self._sdk_mode == _SDK_MODE.HDMI:
            return self.slm_lib.Get_SLMTemp()
        elif self._sdk_mode == _SDK_MODE.PCIE_MODERN:
            self.slm_lib.Get_SLMTemp.restype = ctypes.c_double
            return self.slm_lib.Get_SLMTemp(self.slm_number)
        else:
            raise NotImplementedError(
                "Temperature reading not supported for this model"
            )

    def get_coverglass_voltage(self) -> float:
        """
        Read the voltage of the SLM coverglass.

        Returns
        -------
        float
            Voltage of the SLM coverglass.

        Raises
        ------
        Not Implemented Error
            If the coverglass voltage reading is not supported for the SLM.
        """
        if self._sdk_mode == _SDK_MODE.HDMI:
            return self.slm_lib.Get_SLMVCom()
        elif self._sdk_mode == _SDK_MODE.PCIE_MODERN:
            self.slm_lib.Get_SLMVCom.restype = ctypes.c_double
            return self.slm_lib.Get_cover_voltage(self.slm_number)
        else:
            raise NotImplementedError(
                "Coverglass voltage reading not supported for this model"
            )

    def _load_slm(self, method: Callable[..., bool], verbose: bool = True) \
            -> bool:
        self.slm_lib = self._load_lib(verbose)
        try:
            assert method()
            return True
        except AssertionError:
            self._unload_lib()
            return False

    def _load_lib(self, verbose: bool = True) -> ctypes.CDLL:
        """
        Loads the Meadowlark SDK.
        """
        if verbose:
            print("Constructing Blink SDK...", end="")
        try:
            return ctypes.CDLL(self.sdk_path, mode=ctypes.RTLD_LOCAL)
        except OSError as exc:
            print("failure")
            raise ImportError(
                f"Meadowlark .dlls did not did not import correctly. "
                f"Is '{self.sdk_path}' the correct path?"
            ) from exc

    def _unload_lib(self) -> None:
        """
        Unloads the Meadowlark SDK.
        """
        if self.slm_lib is None:
            return

        # Review: This feels hacky because it is. If you try to unload a DLL
        #  by setting to None and calling garbage collector it will not work
        OS = system()
        if OS == "Windows":
            # Warn: Don't call FreeLibrary directly, as it will not work due to
            #  the CDLL handle will be a 64-bit integer (integer overflow)
            unloader = ctypes.WinDLL('kernel32',
                                     use_last_error=True).FreeLibrary
            unloader.argtypes = [ctypes.wintypes.HMODULE] # Handle to memory addr
            unloader.restype = ctypes.wintypes.BOOL
        elif OS == "Linux":
            unloader = ctypes.cdll.LoadLibrary('libc.so.6').dlclose
            unloader.argtypes = [ctypes.c_void_p]
            unloader.restype = None
        else:
            unloader = lambda x: True

        try:
            self.slm_lib.Delete_SDK()
        except OSError as exc:
            warnings.warn(f"Failed to delete SDK: {e}", stacklevel=2)
        finally:
            # noinspection PyProtectedMember
            if not unloader(self.slm_lib._handle):
                raise OSError(f"Failed to unload DLL {self.sdk_path}; "
                              f"error_code: {ctypes.get_last_error()}.")
            self.slm_lib = None

    def _load_hdmi(self) -> bool:
        """
        Initializes the Meadowlark SDK in HDMI mode.

        Returns
        -------
        SDK_MODE
            Whether the SDK was successfully initialized.
        """
        bool_cpp_or_python = ctypes.c_uint(1)
        try:
            self.slm_lib.Create_SDK(bool_cpp_or_python)
        except OSError:
            return False
        else:
            return True

    def _load_pcie_legacy(self) -> bool:
        """
        Initializes the Meadowlark SDK for legacy 512x512 boards.

        Returns
        -------
        bool
            Whether the SDK was successfully initialized.
        """
        # Technically, some of these boards had "OverDrive Plus" technology that
        # allowed increased switching speed. It greatly simplifies the implementation
        # to load all boards without this feature.
        bitdepth = ctypes.c_uint(8)  # all the old HSP512 were 8-bit as far as I know
        constructed_okay = ctypes.c_bool(False)
        # I don't think Meadowlark has any non-nematic SLMs this days? If this is a
        # an argument can be added later
        is_nematic_type = ctypes.c_bool(True)
        ram_write_enable = ctypes.c_bool(True)
        # Only for ODP or for Meadowlark's ImageGen software to make images
        use_gpu = ctypes.c_bool(False)
        max_transients = ctypes.c_uint(10)  # Only for ODP; leaving at default '10'
        regional_lut = ctypes.c_int(0)  # Only for ODP; leaving at default 'NULL'

        try:
            self.slm_lib.Create_SDK(
                bitdepth,
                ctypes.byref(self._number_of_boards),
                ctypes.byref(constructed_okay),
                is_nematic_type,
                ram_write_enable,
                use_gpu,
                max_transients,
                regional_lut,
            )
            assert constructed_okay.value
        except (AssertionError, OSError):
            return False

        # Ensure the board that is found is a 512x512
        try:
            self._sdk_mode = _SDK_MODE.PCIE_LEGACY
            assert self._get_width() == 512
            assert self._get_height() == 512
        except AssertionError:
            self._sdk_mode = _SDK_MODE.NULL
            return False
        else:
            return True

    def _load_pcie_modern(self) -> bool:
        """
        Initializes the Meadowlark SDK for modern PCIe boards.

        Returns
        -------
        bool
            Whether the SDK was successfully initialized.
        """
        bitdepth = ctypes.c_uint(12)  # The new boards I have access to are 12-bit and
        # my copies of the SDK Manuals have 12-bit for all but the 512x512 boards.
        constructed_okay = ctypes.c_bool(False)
        # I don't think Meadowlark has any non-nematic SLMs this days? If this is a
        # an argument can be added later
        is_nematic_type = ctypes.c_bool(False)
        ram_write_enable = ctypes.c_bool(True)
        # Only for ODP or for Meadowlark's ImageGen software to make images
        use_gpu = ctypes.c_bool(False)
        max_transients = ctypes.c_uint(10)  # Only for ODP; leaving at default '10'
        regional_lut = ctypes.c_int(0)  # Only for ODP; leaving at default 'NULL'

        try:
            self.slm_lib.Create_SDK(
                bitdepth,
                ctypes.byref(self._number_of_boards),
                ctypes.byref(constructed_okay),
                is_nematic_type,
                ram_write_enable,
                use_gpu,
                max_transients,
                regional_lut,
            )
            assert constructed_okay == ctypes.c_bool(True)
        except (AssertionError, OSError):
            return False
        else:
            return True

    def _set_phase_hw(self, display: np.ndarray, slm_number: int | None = None) -> None:
        """
        See :meth:`.SLM._set_phase_hw`.
        """
        slm_number = ctypes.c_uint(slm_number) if slm_number else self.slm_number
        if self._sdk_mode == _SDK_MODE.HDMI:
            self.slm_lib.Write_image(
                display.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte)),
                ctypes.c_uint(self.bitdepth == 8),  # Is 8-bit
            )
        elif (
            self._sdk_mode == _SDK_MODE.PCIE_LEGACY
            or self._sdk_mode == _SDK_MODE.PCIE_MODERN
        ):
            # FUTURE: implement triggered mode.
            # Santec also has a triggered mode that is not currently implemented.
            wait_for_trigger = ctypes.c_bool(False)
            # WARN: Do not change this, as doing so will loses the guarantee that
            #  all the pixels are synchronized to the same image (that is, the earliest
            #  pixels can be updated to the next image before the last pixels are
            #  updated to the current image).
            flip_immediate = ctypes.c_bool(False)
            output_pulse_image_flip = ctypes.c_bool(False)
            output_pulse_image_refresh = ctypes.c_bool(False)
            trigger_timeout = ctypes.c_uint(5000)

            self.slm_lib.Write_image(
                slm_number,
                display.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte)),
                ctypes.c_uint(self.shape[0] * self.shape[1] * self.bitdepth),
                wait_for_trigger,
                flip_immediate,
                output_pulse_image_flip,
                output_pulse_image_refresh,
                trigger_timeout,
            )
            # Ensure that partial images are not displayed
            self.slm_lib.ImageWriteComplete(slm_number, trigger_timeout)
        else:
            # We should never end up here, but if for some reason we did... We would
            # really want to know!!!
            raise RuntimeError("Failed to set phase on SLM due to unknown SDK mode")

    @staticmethod
    def _locate_blink_c_wrapper(search_path: str = _DEFAULT_MEADOWLARK_PATH) -> str:
        """
        Infers the location of the Meadowlark SDK's dynamic linked library.

        Parameters
        ----------
        search_path : str
            Path to search for the Meadowlark SDK.

        Returns
        -------
        str
            The path to the Meadowlark SDK folder.

        Raises
        ------
        FileNotFoundError
            If no Blink_C_Wrapper.dll files are found in provided path.

        """
        # Locate the Meadowlark SDK. If there are multiple, default to the
        # most recent one. The search specifies the dynamic link library file because
        # the search will always return multiple files otherwise (.e.g., header), and
        # give false alarm warnings.
        files = {file for file in Path(search_path).rglob("*Blink_C_Wrapper*dll")}
        if len(files) == 1:
            return str(files.pop())
        elif len(files) >= 1:
            # REVIEW: If the user has BOTH HDMI & PCIe SDKs, the one most recently
            #  installed one will be used. Currently, in this case the user ought to
            #  narrow the search path or (ideally) specify it directly. I'm not sure
            #  there's a reliable way around this. To make it more explicit to the user
            #  which SDK is being used, there is a warning that is raised if multiple
            #  SDKs are found.
            sdk_path = Path(max(files, key=os.path.getctime))
            warnings.warn(
                f"Multiple Meadowlark SDKs located. "
                f"Defaulting to the most recent one"
                f" {sdk_path}.",
                stacklevel=2,
            )
            return str(sdk_path)
        else:
            raise FileNotFoundError(
                f"No Blink_C_Wrapper.dll file found in '{search_path}'."
            )

    @staticmethod
    def _locate_lut_file(
        search_path: str | Path, slm_dims: tuple[int, int] | None = None
    ) -> str:
        """
        Locates the LUT file in the given path. If there are multiple, returns the
        most recent file.. If there are none with ".slm" in the
        name, simply returns the most recent ".lut" file.

        Parameters
        ----------
        search_path : str
            Path to search for the LUT file in.

        Returns
        -------
        str
            The path to the LUT file.

        Raises
        ------
        FileNotFoundError
            If no .lut files are found in provided path.
        """
        files = {file for file in Path(search_path).rglob("*.lut")}

        if len(files) == 1:
            return str(files.pop())
        elif len(files) > 1:
            # REVIEW: If there are multiple LUTs, first check if only one matches the
            #  current slm's dimensions (if possible).
            if slm_dims:
                files = {
                    file
                    for file in files
                    if (f"{slm_dims[0]}" in file.stem and f"{slm_dims[1]}" in file.stem)
                }
                if len(files) == 1:
                    return str(files.pop())
            # If there are still multiple LUTs, default to the most recent one.
            lut_path_ = max(files, key=os.path.getctime)
            warnings.warn(
                f"Multiple LUT files located. Defaulting to the most recent one: "
                f"{lut_path_}.",
                stacklevel=3,
            )
            return str(lut_path_)
        else:
            raise FileNotFoundError(f"No .lut file found in '{search_path}'.")
