"""
Hardware control for Texas Instruments Phase Light Modulators (PLMs).

This module provides GPU-accelerated control for TI PLMs via direct implementation
of phase quantization and electrode mapping. Supports both CuPy (GPU) and NumPy (CPU)
for maximum performance and compatibility.

Also includes :class:`DLPC900`, a USB HID interface for configuring the DLPC900
evaluation module (EVM) that drives the PLM. This automates the setup normally done
through TI's DLPC900 GUI software. For further information, refer to the
`DLPC900 Programmer's Guide <https://www.ti.com/lit/ug/dlpu018j/dlpu018j.pdf>`_.

Key Features
------------
- **GPU Acceleration**: Automatic CuPy detection for 5-20x speedup
- **CPU Fallback**: Seamless NumPy fallback when GPU unavailable
- **Optimized Pipeline**: Minimal memory allocations and data transfers
- **USB Configuration**: Automated DLPC900 EVM setup via :class:`DLPC900`

Device Database
---------------
Device specifications are stored in ``texas_instruments.json`` in the same directory.
Available devices can be queried with :meth:`get_device_list()`.

Example
-------
::

    from slmsuite.hardware.slms.texasinstruments import PLM

    # Create PLM instance with automatic USB configuration
    plm = PLM("p47", display_number=1, configure_usb=True)

    # Or without USB (requires prior GUI setup)
    plm = PLM("p47", display_number=1)

    # Set phase pattern
    phase = np.random.rand(540, 960) * 2 * np.pi
    plm.write(phase)

Attributes
----------
device_config : dict
    Device configuration loaded from texas_instruments.json
phase_buckets : ndarray
    Pre-computed phase quantization boundaries
"""

import json
import os
import time
import warnings
from enum import IntEnum
import numpy as np
from slmsuite.hardware._pyglet import _WindowThread
from slmsuite.hardware.slms.screenmirrored import ScreenMirrored

try:
    import cupy as cp
except ImportError:
    cp = np
    warnings.warn(
        "cupy is not installed; using numpy. "
        "Install cupy for GPU-accelerated PLM control.",
    )

# HID availability (for DLPC900 USB control)
try:
    import hid as _hid
    HID_AVAILABLE = True
except ImportError:
    _hid = None
    HID_AVAILABLE = False

# PLM Constants
DEVICE_DB_PATH = os.path.join(os.path.dirname(__file__), "texas_instruments.json")
DLPC900_VENDOR_ID = 0x0451
DLPC900_PRODUCT_ID = 0xC900
DLPC900_EXPOSURE_US = 694

class DisplayMode(IntEnum):
    """
    DLPC900 display modes.
    """
    VIDEO         = 0
    PATTERN       = 1
    VIDEO_PATTERN = 2
    OTF           = 3

class DLPC900Command(IntEnum):
    """
    DLPC900 USB command codes.

    Each value is the two byte ommand code sent over USB HID, as defined in
    the `DLPC900 Programmer's Guide (DLPU018J)
    <https://www.ti.com/lit/ug/dlpu018j/dlpu018j.pdf>`_.
    """
                              # Programmer Guide Sections
    POWER_MODE     = 0x0200   # 2.2.1 — Standby / wakeup / reset
    VERSION        = 0x0206   # 2.1.5 — Firmware version info
    HW_STATUS      = 0x1A0A   # 2.1.1 — Hardware status register
    MAIN_STATUS    = 0x1A0C   # 2.1.3 — Main status register
    INPUT_SOURCE   = 0x1A00   # 2.3.1 — Input source selection
    IT6535_POWER   = 0x1A01   # 2.3.2 — IT6535 receiver power mode
    PORT_CLOCK     = 0x1A03   # 2.3.3 — Port and clock configuration
    DISPLAY_MODE   = 0x1A1B   # 2.4.1 — Display mode selection
    PAT_STARTSTOP  = 0x1A24   # 2.4.4.3.1 — Pattern start / stop / pause
    PAT_LUT_CONFIG = 0x1A31   # 2.4.4.3.3 — Pattern LUT configuration
    PAT_LUT_DEFINE = 0x1A34   # 2.4.4.3.5 — Pattern LUT entry definition


class PLM(ScreenMirrored):
    """
    Interfaces with Texas Instruments' Phase Light Modulators (PLMs).

    This class combines :class:`ScreenMirrored` for display with GPU-accelerated
    phase quantization and electrode mapping. Automatically detects and uses
    CuPy for GPU acceleration, falling back to NumPy if unavailable.

    Optionally configures the DLPC900 EVM via USB, replacing the manual setup
    normally done through TI's GUI software.

    Parameters
    ----------
    device_name : str
        Device identifier from texas_instruments.json (e.g., ``"p47"``, ``"p67"``)
    display_number : int
        Monitor number for display.
    verbose : bool, optional
        Whether to print extra information. Defaults to True.
    configure_usb : bool, optional
        If True, automatically configure the DLPC900 EVM via USB before
        initializing the display. Requires ``hidapi``
        (see :class:`DLPC900`). Defaults to False.
    video_input : str, optional
        Video input source: ``"displayport"`` or ``"hdmi"``.
        Only used when ``configure_usb=True``. Defaults to ``"displayport"``.
    pixel_mode : str or None, optional
        Pixel clock mode: ``"single"`` (30 Hz) or ``"dual"`` (60 Hz).
        If None, defaults to ``"dual"`` for DisplayPort or ``"single"`` for HDMI.
        Only used when ``configure_usb=True``.
    usb_vendor_id : int or None, optional
        Override USB vendor ID for DLPC900.
    usb_product_id : int or None, optional
        Override USB product ID for DLPC900.
    **kwargs
        Additional arguments for :class:`ScreenMirrored`.

    Attributes
    ----------
    device_config : dict
        Device configuration from texas_instruments.json.
    phase_buckets : ndarray
        Pre-computed quantization boundaries.
    dlpc900 : DLPC900 or None
        USB interface to DLPC900 EVM, if configured.
    electrode_layout : ndarray
        Physical electrode layout (CuPy or NumPy).
    memory_lut : ndarray
        Memory lookup table.
    data_flip : tuple
        Axis flip flags for electrode output.
    """

    def __init__(
        self,
        device_name,
        display_number,
        verbose=True,
        configure_usb=False,
        video_input="displayport",
        pixel_mode=None,
        usb_vendor_id=None,
        usb_product_id=None,
        **kwargs
    ):
        self.dlpc900 = None
        self.display_number = display_number

        # Load device configuration from JSON
        self.device_config = self.load_device_config(device_name)

        if verbose:
            backend = "GPU (CuPy)" if cp != np else "CPU (NumPy)"
            print(f"PLM using {backend} backend")

        # Extract device parameters
        self.device_shape = tuple(self.device_config["shape"])  # (rows, cols) - input phase shape
        pitch_um = tuple(np.array(self.device_config["pitch"]) * 1e6)  # Convert m to µm

        # Store electrode layout for later use
        self._electrode_layout_raw = np.array(self.device_config["electrode_layout"])

        # USB pre-config: set up PLM as display 
        if configure_usb:
            self.dlpc900 = DLPC900(vendor_id=usb_vendor_id,
                                   product_id=usb_product_id)
            self._usb_pre_configure(video_input, pixel_mode, display_number, verbose)

        # Compute bitdepth from number of displacement ratios
        n_phases = len(self.device_config["displacement_ratios"])
        bitdepth = int(np.log2(n_phases))

        # Initialize parent ScreenMirrored class with device shape (vs display shape)
        # The SLM.shape should represent the input phase dimensions
        super().__init__(
            display_number,
            verbose=verbose,
            resolution=self.device_shape[::-1],  # ScreenMirrored expects (width, height)
            bitdepth=bitdepth,
            pitch_um=pitch_um,
            **kwargs
        )

        # Calculate display shape after electrode mapping
        elec_shape = self._electrode_layout_raw.shape
        self.display_shape = (self.device_shape[0] * elec_shape[0], self.device_shape[1] * elec_shape[1])

        # Update window shape and recreate buffers for electrode-mapped output.
        # Must run on the window thread to satisfy OpenGL context thread affinity.
        def _reconfigure(window, shape):
            window.shape = shape
            window._setup_context()

        future = self._window_thread.submit(_reconfigure, self.window, self.display_shape)
        _WindowThread.wait(future)

        # USB post-config: wait for source lock and switch to video-pattern mode.
        if configure_usb:
            self._usb_post_configure(video_input, pixel_mode, verbose)

        # Pre-compute phase buckets for quantization
        self._init_phase_buckets()

        # Convert device arrays to GPU if needed
        self.memory_lut = cp.array(self.device_config["memory_lut"], dtype=cp.uint8)
        self.electrode_layout = cp.array(self._electrode_layout_raw, dtype=cp.uint8)
        self.data_flip = tuple(self.device_config["data_flip"])

    @staticmethod
    def load_device_config(device_name):
        """
        Load device configuration from texas_instruments.json.

        Parameters
        ----------
        device_name : str
            Device identifier (e.g., "p47", "p67")

        Returns
        -------
        dict
            Device configuration

        Raises
        ------
        ValueError
            If device not found in database
        """
        with open(DEVICE_DB_PATH, 'r') as f:
            device_db = json.load(f)

        if device_name not in device_db:
            available = list(device_db.keys())
            raise ValueError(
                f"Device '{device_name}' not found. "
                f"Available devices: {available}"
            )

        return device_db[device_name]

    def _usb_pre_configure(self, video_input, pixel_mode, display_number, verbose=True):
        """
        USB setup steps that must happen before the pyglet window is created.

        Sets input source, port clock configuration, and switches to video mode
        so the EVM is ready to accept video signal from the display. Polls pyglet
        to confirm the target display is available before proceeding.
        """
        from slmsuite.hardware.slms.screenmirrored import ScreenMirrored

        dlpc = self.dlpc900

        if verbose:
            fw = dlpc.get_firmware_version()
            print(f"DLPC900 connected: firmware {fw}")

        # Resolve pixel mode default
        if pixel_mode is None:
            pixel_mode = "dual" if video_input == "displayport" else "single"

        # Configure port clock for single or dual pixel mode
        if pixel_mode == "dual":
            dlpc.set_port_clock(data_port=2)
        else:
            dlpc.set_port_clock(data_port=0)

        # Power up IT6535 receiver for the correct input before any display config
        dlpc.set_it6535_power(video_input)

        # Switch to video mode (required before video-pattern)
        dlpc.set_display_mode("video")

        # Wait for the target display to become available
        DLPC900._poll_until(
            lambda: display_number in [s[0] for s in ScreenMirrored.info(verbose=False)],
            error_msg=f"Display {display_number} not detected.",
        )

        if verbose:
            print("DLPC900 pre-configured (video mode, display detected)")

    def _usb_post_configure(self, video_input, pixel_mode, verbose):
        """
        USB setup steps that happen after the pyglet window is created.

        Waits for external source lock (video signal detected), then switches
        to video-pattern mode, configures the pattern LUT, and starts the
        pattern sequence.
        """
        dlpc = self.dlpc900

        # Resolve pixel mode default
        if pixel_mode is None:
            pixel_mode = "dual" if video_input == "displayport" else "single"

        # Wait for external source lock
        DLPC900._poll_until(
            lambda: dlpc.get_main_status()["source_locked"],
            error_msg=("DLPC900: Video source failed to lock. "))

        if verbose:
            print("DLPC900 source locked, switching to video-pattern mode...")

        # Switch to video-pattern mode and wait for confirmation
        dlpc.set_display_mode("video-pattern")
        DLPC900._poll_until(
            lambda: dlpc.get_display_mode() == DisplayMode.VIDEO_PATTERN,
            error_msg="DLPC900: Failed to switch to video-pattern mode.",
        )

        # Stop any existing sequence
        dlpc.stop_pattern()

        # Define a single 1-bit pattern entry (copied to all bits by PLM class):
        # - No clear 
        # - Trigger out 2 enabled (per GUI instructions)
        # - Frame change on first bit (bit_position=0)
        dlpc.define_pattern(
            index=0,
            bitdepth=1,
            color=7, #white
            clear_after_exposure=False,
            wait_for_trigger=False,
            dark_time_us=0,
            trigger_out2=True,
            image_index=0,
            bit_position=0,
        )

        # Configure LUT: 1 entry, repeat indefinitely
        dlpc.configure_pattern_lut(num_entries=1, num_repeats=0)
        time.sleep(1) # Wait for small unresponsive time window

        # Start the pattern sequence and wait for confirmation
        dlpc.start_pattern()
        DLPC900._poll_until(
            lambda: dlpc.get_main_status()["sequencer_running"],
            timeout_s=2,
            error_msg=(
                "DLPC900: Pattern sequence failed to start after 2 seconds. "
                "Check hardware status with dlpc.get_hardware_status()."
            ),
        )

        if verbose:
            print("DLPC900 configured successfully - pattern sequence running")

    def close(self):
        """Close the PLM, stopping the pattern sequence and releasing USB."""
        if self.dlpc900 is not None:
            try:
                self.dlpc900.stop_pattern()
                self.dlpc900.standby()
                self.dlpc900.close()
            except Exception:
                pass
            self.dlpc900 = None
        super().close()

    def _init_phase_buckets(self):
        """
        Pre-compute phase quantization boundaries.

        Creates bucket boundaries for digitizing continuous phase values
        into discrete phase states based on device displacement ratios.
        """
        displacement_ratios = np.array(self.device_config["displacement_ratios"])
        phase_range = (0, 2 * np.pi)

        # Scale displacement ratios to (bitresolution - 1) / bitresolution
        # This maps the full range to one less than available states
        ratio_scale = (self.bitresolution - 1) / self.bitresolution

        # Map displacement ratios to phase values
        phase_disp = phase_range[0] + displacement_ratios * ratio_scale * (phase_range[1] - phase_range[0])
        phase_disp = np.concatenate([phase_disp, [phase_range[1]]])

        # Create bucket boundaries (average of adjacent phase levels)
        self.phase_buckets = (phase_disp[:-1] + phase_disp[1:]) / 2

        # Convert to GPU array if using CuPy
        self.phase_buckets = cp.array(self.phase_buckets)

    def _quantize(self, phase_map):
        """
        Quantize continuous phase values to discrete phase states.

        Parameters
        ----------
        phase_map : ndarray
            Phase data

        Returns
        -------
        ndarray (uint8)
            Quantized phase state indices [0, bitresolution)
        """
        # Ensure array is on the correct device
        phase_map = cp.asarray(phase_map)

        # Optimized quantization using searchsorted (faster than digitize)
        # searchsorted is optimized for sorted arrays and can be faster on GPU
        phase_state_idx = cp.searchsorted(self.phase_buckets, phase_map, side='right')
        phase_state_idx = phase_state_idx % self.bitresolution

        return phase_state_idx.astype(cp.uint8)

    def _electrode_map(self, phase_state_idx):
        """
        Map phase state indices to electrode bit patterns.

        Converts quantized phase states to the physical electrode layout
        pattern required by the PLM hardware.

        Parameters
        ----------
        phase_state_idx : ndarray (uint8)
            Phase state indices, must be at least 2D
            Last 2 dimensions represent (rows, cols)

        Returns
        -------
        ndarray (uint8)
            Binary electrode pattern with expanded dimensions based on
            electrode_layout shape
        """
        # Look up memory values for each phase state
        memory = self.memory_lut[phase_state_idx]

        # Broadcast and apply bitwise operations for electrode mapping
        # memory[..., None, None] adds 2 dims: (..., rows, cols, 1, 1)
        # electrode_layout has shape (elec_rows, elec_cols)
        # Result has shape (..., rows, cols, elec_rows, elec_cols)
        out = cp.right_shift(
            memory[..., None, None],
            self.electrode_layout
        ).astype(cp.uint8) & 1

        # Rearrange axes and reshape to interleave electrode bits
        elec_h, elec_w = self.electrode_layout.shape
        new_shape = memory.shape[:-2] + (memory.shape[-2] * elec_h, memory.shape[-1] * elec_w)
        out = cp.swapaxes(out, -2, -3).reshape(new_shape)

        # Apply data flip if specified
        flip_axes = tuple(-2 + idx for idx, flip in enumerate(self.data_flip) if flip)
        if flip_axes:
            out = cp.flip(out, flip_axes)

        return out

    def _format_phase_hw(self, phase, replicate_bits=True, enforce_shape=True):
        """
        Process phase array into PLM electrode bitmap.

        Combines quantization and electrode mapping into optimized pipeline.
        Data stays on GPU if available for maximum performance - ScreenMirrored
        will handle GPU→CPU transfer only when needed for display.

        Parameters
        ----------
        phase : numpy.ndarray or cupy.ndarray
            Phase data in range [0, 2π]
        replicate_bits : bool, optional
            Multiply final bitplane by 255 to display same CGH for full frame.
            Defaults to True.
        enforce_shape : bool, optional
            Check that input shape matches device shape. Defaults to True.

        Returns
        -------
        numpy.ndarray or cupy.ndarray (uint8)
            Electrode-mapped bitmap ready for display.
            Returns GPU array if GPU is available, otherwise CPU array.

        Raises
        ------
        ValueError
            If enforce_shape=True and phase shape doesn't match device
        """
        # Shape validation
        if enforce_shape:
            expected_shape = tuple(self.device_config["shape"])
            if len(phase.shape) < 2 or phase.shape[-2:] != expected_shape:
                raise ValueError(
                    f"Phase map shape {phase.shape} does not match "
                    f"device shape {expected_shape}"
                )

        # Convert phase from slmsuite convention to [0, 2π]
        # TODO: can wrap into phase2gray
        # slmsuite uses bitresolution-scaled values, we need radians
        # phase = self._phase2gray(phase) * (2 * np.pi / self.bitresolution)
        phase -= phase.min()
        phase %= 2 * np.pi

        # Quantize phase to discrete states (xp.asarray handles device transfer)
        phase_state_idx = self._quantize(phase)

        # Map to electrode pattern
        out = self._electrode_map(phase_state_idx)

        # Replicate bits for full frame display
        if replicate_bits:
            out *= 255

        # Keep data on GPU - ScreenMirrored._set_phase_hw() will transfer to CPU
        # only when needed for display. This avoids unnecessary GPU→CPU→GPU transfers
        # and allows GPU processing pipelines to stay on GPU.
        return out

    @staticmethod
    def bitpack(bitmaps):
        """
        Combine multiple binary CGHs into single 8-bit or 24-bit image.

        Stacks the MSB of 8 or 24 bitmaps into a single multi-bit image.
        Supports GPU acceleration if CuPy is available and input is on GPU.

        Parameters
        ----------
        bitmaps : list or tuple of ndarray
            List of 8 or 24 binary bitmaps (uint8) of same shape

        Returns
        -------
        ndarray (uint8)
            Packed image with shape (1, rows, cols) for 8 bitmaps
            or (3, rows, cols) for 24 bitmaps (RGB channels)

        Raises
        ------
        ValueError
            If number of bitmaps is not 8 or 24
        """
        # Determine if using GPU (check if any input is a CuPy array)
        use_gpu = cp != np and any(
            isinstance(bm, cp.ndarray) for bm in bitmaps
        )
        xp = cp if use_gpu else np

        # Ensure all bitmaps are on same device
        bitmaps = [xp.asarray(bm) for bm in bitmaps]

        if len(bitmaps) == 8:
            # Single channel output
            stacked = xp.stack(bitmaps) & 1  # Isolate LSB
            shifts = xp.arange(8)[:, None, None]  # Shape (8, 1, 1) for broadcasting
            shifted = xp.left_shift(stacked.astype(xp.uint8), shifts.astype(xp.uint8))
            result = xp.sum(shifted, axis=0)[None, ...]  # Add channel dimension

        elif len(bitmaps) == 24:
            # RGB output (3 channels, 8 bits each)
            rgb = []
            for n in range(3):
                channel_bitmaps = bitmaps[n*8:(n+1)*8]
                stacked = xp.stack(channel_bitmaps) & 1
                shifts = xp.arange(8)[:, None, None]
                shifted = xp.left_shift(stacked.astype(xp.uint8), shifts.astype(xp.uint8))
                rgb.append(xp.sum(shifted, axis=0))
            result = xp.stack(rgb)

        else:
            raise ValueError(
                f"Bitpack requires 8 or 24 bitmaps, got {len(bitmaps)}"
            )

        # Convert back to NumPy if needed
        if use_gpu:
            result = np.asarray(result)

        return result

    @staticmethod
    def get_device_list():
        """
        Get list of available PLM devices from database.

        Returns
        -------
        list of str
            Device identifiers available in texas_instruments.json
        """
        with open(DEVICE_DB_PATH, 'r') as f:
            device_db = json.load(f)

        return list(device_db.keys())


class DLPC900:
    """
    USB HID interface for the DLPC900 evaluation module.

    Implements the DLPC900 USB commands needed to configure the EVM for video
    pattern mode, eliminating the need for TI's GUI software. Uses the native
    OS HID driver via ``hidapi`` — no driver replacement (Zadig) required.

    The DLPC900 communicates via 64-byte HID reports with a 6-byte header::

        [flag, seq, len_lo, len_hi, cmd_lo, cmd_hi, ...payload...]

    See :class:`DLPC900Command` for the implemented command codes and their
    DLPU018J section references.

    Parameters
    ----------
    vendor_id : int or None
        USB vendor ID. Defaults to ``0x0451`` (Texas Instruments).
    product_id : int or None
        USB product ID. Defaults to ``0xC900`` (DLPC900 EVM).

    Raises
    ------
    ImportError
        If the ``hidapi`` package is not installed.
    RuntimeError
        If the DLPC900 USB device is not found.

    Example
    -------
    ::

        from slmsuite.hardware.slms.texasinstruments import DLPC900

        dlpc = DLPC900()
        print(dlpc.get_main_status())
        dlpc.close()
    """

    def __init__(self, vendor_id=None, product_id=None):
        if not HID_AVAILABLE:
            raise ImportError(
                "hidapi is required for DLPC900 USB control. "
                "Install with: pip install hidapi"
            )

        vid = vendor_id if vendor_id is not None else DLPC900_VENDOR_ID
        pid = product_id if product_id is not None else DLPC900_PRODUCT_ID

        self._dev = _hid.device()
        try:
            self._dev.open(vid, pid)
        except OSError as e:
            raise RuntimeError(
                f"DLPC900 USB device not found (VID=0x{vid:04X}, PID=0x{pid:04X}). "
                "Check that the EVM is powered on and connected via USB."
            ) from e

        self._seq = 0

    def _send(self, mode, cmd, payload=None):
        """
        Send a command and optionally read the response.

        Parameters
        ----------
        mode : str
            ``'r'`` for read, ``'w'`` for write.
        cmd : DLPC900Command or int
            16-bit command code.
        payload : list of int or None
            Command data bytes.

        Returns
        -------
        list of int or None
            64-byte response for reads, None for writes.
        """
        if payload is None:
            payload = []

        self._seq = (self._seq + 1) & 0xFF
        cmd = int(cmd)
        length = len(payload) + 2

        # Build 64-byte packet: [flag, seq, len_lo, len_hi, cmd_lo, cmd_hi, ...data...]
        flag = 0xC0 if mode == 'r' else 0x00
        header = bytes([flag, self._seq]) + length.to_bytes(2, 'little') + cmd.to_bytes(2, 'little')
        buf = list(header) + payload[:58] + [0] * (58 - len(payload[:58]))

        # hidapi write: prepend report ID 0x00
        # print(" ".join(f"{b:02X}" for b in buf))
        self._dev.write([0x00] + buf)

        # Multi-packet payload (>58 bytes)
        remaining = payload[58:]
        while remaining:
            chunk = remaining[:64]
            remaining = remaining[64:]
            padded = chunk + [0x00] * (64 - len(chunk))
            self._dev.write([0x00] + padded)

        if mode == 'r':
            try:
                ret = self._dev.read(64, timeout_ms=1000)
                # print(" ".join(f"{b:02X}" for b in ret))
                return ret
            except Exception:
                print("Read command failed; ensure PLM GUI is closed.")

        return None

    def _read_byte(self, cmd):
        """Read a single status byte (response byte 5) for a command."""
        ans = self._send('r', cmd)
        return ans[4] if ans else None

    @staticmethod
    def _poll_until(check_fn, timeout_s=10, interval_s=0.5, error_msg=""):
        """
        Poll ``check_fn`` until it returns truthy, or raise on timeout.

        Parameters
        ----------
        check_fn : callable
            Zero-argument callable that returns a truthy value on success.
        timeout_s : float
            Maximum time to wait in seconds.
        interval_s : float
            Sleep interval between polls in seconds.
        error_msg : str
            Message for the :class:`RuntimeError` raised on timeout.
        """
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            time.sleep(interval_s)
            if check_fn():
                return
        raise RuntimeError(error_msg)

    def close(self):
        """Release the USB HID device."""
        if self._dev is not None:
            self._dev.close()
            self._dev = None

    def get_hardware_status(self):
        """
        Read hardware status register.

        Returns
        -------
        dict
            Bool flags: ``init_done``, ``drc_error``, ``forced_swap``,
            ``sequencer_abort``, ``sequencer_error``.
        """
        b = self._read_byte(DLPC900Command.HW_STATUS)
        return {
            "init_done":       bool(b & 0x01),
            "drc_error":       bool(b & 0x04),
            "forced_swap":     bool(b & 0x08),
            "sequencer_abort": bool(b & 0x40),
            "sequencer_error": bool(b & 0x80),
        }

    def get_main_status(self):
        """
        Read main status register.

        Returns
        -------
        dict
            Bool flags: ``mirrors_parked``, ``sequencer_running``,
            ``video_frozen``, ``source_locked``, ``port1_syncs_valid``,
            ``port2_syncs_valid``.
        """
        b = self._read_byte(DLPC900Command.MAIN_STATUS)
        return {
            "mirrors_parked":    bool(b & 0x01),
            "sequencer_running": bool(b & 0x02),
            "video_frozen":      bool(b & 0x04),
            "source_locked":     bool(b & 0x08),
            "port1_syncs_valid": bool(b & 0x10),
            "port2_syncs_valid": bool(b & 0x20),
        }

    def get_firmware_version(self):
        """
        Read firmware version info.

        Returns
        -------
        dict
            Keys: ``app_version``, ``api_version``, ``sw_patch``,
            ``sw_minor``, ``sw_major``.
        """
        ans = self._send('r', DLPC900Command.VERSION)
        if not ans or len(ans) < 10:
            return {}
        return {
            "app_version": ans[6],
            "api_version": ans[7],
            "sw_patch":    ans[8],
            "sw_minor":    ans[9],
            "sw_major":    ans[10] if len(ans) > 10 else 0,
        }

    def set_input_source(self, source=0, bitdepth=0):
        """
        Set input source.

        Parameters
        ----------
        source : int
            0 = parallel (HDMI/DP), 1 = test, 2 = flash, 3 = curtain.
        bitdepth : int
            0 = 30-bit, 1 = 24-bit, 2 = 20-bit, 3 = 16-bit.
        """
        self._send('w', DLPC900Command.INPUT_SOURCE,
                   [source & 0x07 | (bitdepth & 0x03) << 3])

    def set_port_clock(self, data_port, px_clock=0, data_enable=0, vhsync=0):
        """
        Configure data port and clock routing.

        Parameters
        ----------
        data_port : int
            0 = port 1, 1 = port 2, 2 = dual (1-2), 3 = dual (2-1).
        px_clock : int
            0 = clock 1, 1 = clock 2, 2 = clock 3.
        data_enable : int
            0 = enable 1, 1 = enable 2.
        vhsync : int
            0 = P1 sync, 1 = P2 sync.
        """
        self._send('w', DLPC900Command.PORT_CLOCK, [
            data_port & 0x03
            | (px_clock & 0x03) << 2
            | (data_enable & 0x01) << 4
            | (vhsync & 0x01) << 5
        ])

    def set_display_mode(self, mode):
        """
        Set display mode.

        Parameters
        ----------
        mode : str or DisplayMode
            ``"video"``, ``"pattern"``, ``"video-pattern"``, or ``"otf"``.
            Must be in ``"video"`` mode with source locked before switching
            to ``"video-pattern"``.
        """
        if isinstance(mode, DisplayMode):
            self._send('w', DLPC900Command.DISPLAY_MODE, [int(mode)])
            return

        # Accept string with underscore or hyphen
        name = mode.upper().replace("-", "_")
        try:
            val = DisplayMode[name]
        except KeyError:
            valid = [m.name.lower().replace("_", "-") for m in DisplayMode]
            raise ValueError(
                f"Unknown mode '{mode}'. Valid: {valid}"
            )
        self._send('w', DLPC900Command.DISPLAY_MODE, [int(val)])

    def get_display_mode(self):
        """
        Read current display mode.

        Returns
        -------
        DisplayMode
            The current display mode.
        """
        b = self._read_byte(DLPC900Command.DISPLAY_MODE)
        try:
            return DisplayMode(b)
        except ValueError:
            raise ValueError(f"Unknown display mode byte: {b}")

    def start_pattern(self):
        """Start the pattern display sequence."""
        self._send('w', DLPC900Command.PAT_STARTSTOP, [2])

    def stop_pattern(self):
        """Stop the pattern display sequence."""
        self._send('w', DLPC900Command.PAT_STARTSTOP, [0])

    def configure_pattern_lut(self, num_entries, num_repeats=0):
        """
        Configure the pattern LUT.

        Parameters
        ----------
        num_entries : int
            Number of LUT entries to display.
        num_repeats : int
            Repeat count (0 = infinite).
        """
        self._send('w', DLPC900Command.PAT_LUT_CONFIG,
                   list(num_entries.to_bytes(2, 'little'))
                   + list(num_repeats.to_bytes(4, 'little')))

    def define_pattern(
        self, index, bitdepth=1, color=7,
        clear_after_exposure=False, wait_for_trigger=False,
        dark_time_us=0, trigger_out2=False,
        image_index=0, bit_position=0,
    ):
        """
        Define a single pattern LUT entry.

        Uses the fixed exposure time :data:`DLPC900_EXPOSURE_US`.

        Parameters
        ----------
        index : int
            LUT index (0-399).
        bitdepth : int
            Bit depth (1-8).
        color : int
            Color channel (0-7; 7 = all RGB).
        clear_after_exposure : bool
            Clear pattern after exposure.
        wait_for_trigger : bool
            Wait for external trigger.
        dark_time_us : int
            Dark time after exposure (microseconds).
        trigger_out2 : bool
            Assert trigger output 2.
        image_index : int
            Source image/frame index.
        bit_position : int
            Bit position in image (0-23).
        """
        # Byte 5: [trigger_wait(7)][color(6:4)][depth-1(3:1)][clear(0)]
        options = (
            int(clear_after_exposure) & 0x01
            | ((bitdepth - 1) & 0x07) << 1
            | (color & 0x07) << 4
            | (int(wait_for_trigger) & 0x01) << 7
        )

        self._send('w', DLPC900Command.PAT_LUT_DEFINE,
                   list(index.to_bytes(2, 'little'))
                   + list(DLPC900_EXPOSURE_US.to_bytes(3, 'little'))
                   + [options]
                   + list(dark_time_us.to_bytes(3, 'little'))
                   + [int(trigger_out2) & 0x01,
                      image_index & 0xFF,
                      (image_index >> 8) & 0x07 | (bit_position & 0x1F) << 3])

    def set_it6535_power(self, mode):
        """
        Set IT6535 receiver power mode (0x1A01).

        Must be called before setting video mode.

        Parameters
        ----------
        mode : int or str
            0 or ``"off"`` = power-down (outputs tri-stated),
            1 or ``"hdmi"`` = power-up for HDMI input,
            2 or ``"displayport"`` = power-up for DisplayPort input.
        """
        modes = {"off": 0, "hdmi": 1, "displayport": 2}
        if isinstance(mode, str):
            mode = modes[mode.lower()]
        if mode not in modes.values():
            raise ValueError(f"Invalid IT6535 power mode: {mode}")
        else:
            self._send('w', DLPC900Command.IT6535_POWER, [mode & 0x03])

    def standby(self):
        """Put the IT6535 receiver into power-down mode."""
        self.set_it6535_power(0)

    def reset(self):
        """Reset the DLPC900."""
        self._send('w', DLPC900Command.POWER_MODE, [0x02])