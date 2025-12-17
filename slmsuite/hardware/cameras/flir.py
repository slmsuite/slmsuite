"""
Hardware control for FLIR cameras via the :mod:`PySpin` interface to the Spinnaker SDK.
Install Spinnaker by following the
`provided instructions <https://www.flir.com/products/spinnaker-sdk/>`_.
Tested with FLIR Blackfly S (BFS-PGE-16S2M).
"""

from __future__ import annotations

import warnings
from typing import List, Tuple, Optional
import threading
import queue

import numpy as np  # noqa: F401 (used by PySpin.GetNDArray())

try:
    import PySpin
except ImportError:
    PySpin = None
    warnings.warn("PySpin not installed. Install to use FLIR cameras.")

from .camera import Camera


class FLIR(Camera):
    """
    FLIR camera subclass for interfacing with :mod:`PySpin` and the Spinnaker SDK.

    Each instance runs its own background acquisition thread. The thread owns
    ``BeginAcquisition()``/``GetNextImage()``/``EndAcquisition()``. The ``.get_image()``
    API pulls frames from an internal queue, so you can safely use multiple
    FLIR objects at once from a single Python thread.
    """

    # Class variable pointing to a singleton Spinnaker System object.
    sdk: Optional["PySpin.System"] = None

    # Construction / initialization

    def __init__(
        self,
        serial: str = "",
        pitch_um: Optional[Tuple[float, float]] = None,
        verbose: bool = True,
        **kwargs,
    ) -> None:
        if PySpin is None:
            raise ImportError(
                "PySpin not installed. Install FLIR Spinnaker SDK and its Python "
                "bindings to use FLIR cameras."
            )

        # Default pixel pitch for BFS-PGE-16S2M if not specified.
        if pitch_um is None:
            pitch_um = (3.45, 3.45)

        # ----- Initialize the Spinnaker system (singleton) -----
        if FLIR.sdk is None:
            if verbose:
                print("Initializing PySpin system... ", end="")
            FLIR.sdk = PySpin.System.GetInstance()
            if verbose:
                print("success")

        # ----- Discover and attach to the desired camera -----
        cam_list = FLIR.sdk.GetCameras()
        self.cam = None  # type: ignore
        try:
            num_cams = cam_list.GetSize()
            if num_cams == 0:
                cam_list.Clear()
                raise RuntimeError("No FLIR / Spinnaker cameras detected.")

            if serial:
                selected = None
                for i in range(num_cams):
                    cam = cam_list[i]
                    tl_nodemap = cam.GetTLDeviceNodeMap()
                    node_serial = PySpin.CStringPtr(
                        tl_nodemap.GetNode("DeviceSerialNumber")
                    )
                    if PySpin.IsAvailable(node_serial) and PySpin.IsReadable(
                        node_serial
                    ):
                        if node_serial.GetValue() == serial:
                            selected = cam
                            break
                if selected is None:
                    cam_list.Clear()
                    raise RuntimeError(
                        f"Could not find FLIR camera with serial '{serial}'. "
                        "Use FLIR.info() to list available devices."
                    )
                self.cam = selected
            else:
                # Default to first camera
                self.cam = cam_list[0]
                tl_nodemap = self.cam.GetTLDeviceNodeMap()
                node_serial = PySpin.CStringPtr(
                    tl_nodemap.GetNode("DeviceSerialNumber")
                )
                if PySpin.IsAvailable(node_serial) and PySpin.IsReadable(node_serial):
                    serial = node_serial.GetValue()

            if verbose:
                print(f"Opening FLIR camera '{serial}'... ", end="")

            # Once we have the camera handle, we can clear the list.
            cam_list.Clear()

            # ----- Initialize the camera -----
            self.cam.Init()
            self._nodemap = self.cam.GetNodeMap()
            self._tl_nodemap = self.cam.GetTLDeviceNodeMap()
            self._stream_nodemap = self.cam.GetTLStreamNodeMap()

            # Configure stream & acquisition mode for performance
            self._configure_stream(verbose=verbose)
            self._configure_acquisition_mode(verbose=verbose)

            # IMPORTANT: limit per-camera bandwidth when multiple cameras share a link
            self._configure_bandwidth(num_cams=num_cams, verbose=verbose)
            self._configure_frame_rate(fps=15.0, verbose=verbose)


            # ----- Configure basic image format -----
            bitdepth = self._configure_pixel_format(verbose=verbose)
            # Datatype is later set in Camera.__init__.

            # Resolution (Width, Height) – note Camera.__init__ expects (width, height).
            width_node = PySpin.CIntegerPtr(self._nodemap.GetNode("Width"))
            height_node = PySpin.CIntegerPtr(self._nodemap.GetNode("Height"))
            width = int(width_node.GetValue())
            height = int(height_node.GetValue())

            # Exposure node & bounds (seconds)
            self._exposure_node = PySpin.CFloatPtr(self._nodemap.GetNode("ExposureTime"))
            if PySpin.IsAvailable(self._exposure_node) and PySpin.IsReadable(
                self._exposure_node
            ):
                # Disable auto exposure so slmsuite's autoexposure has full control
                exposure_auto = PySpin.CEnumerationPtr(
                    self._nodemap.GetNode("ExposureAuto")
                )
                if (
                    PySpin.IsAvailable(exposure_auto)
                    and PySpin.IsWritable(exposure_auto)
                ):
                    off_entry = exposure_auto.GetEntryByName("Off")
                    if (
                        off_entry is not None
                        and PySpin.IsAvailable(off_entry)
                        and PySpin.IsReadable(off_entry)
                    ):
                        exposure_auto.SetIntValue(off_entry.GetValue())

                # ExposureTime is in microseconds; convert to seconds.
                try:
                    t_min = float(self._exposure_node.GetMin()) * 1e-6
                    t_max = float(self._exposure_node.GetMax()) * 1e-6
                    self.exposure_bounds_s = (t_min, t_max)
                except Exception:
                    # If bounds are not available, leave as default (None)
                    pass

            # Track whether we have an active acquisition (used by thread & helpers).
            self._acquiring = False

            # ----- Initialize Camera superclass -----
            super().__init__(
                (width, height),
                bitdepth=bitdepth,
                pitch_um=pitch_um,
                name=serial or "FLIR",
                **kwargs,
            )

            # Store default full-frame WOI and cached shape
            self.woi = (0, width, 0, height)
            self.default_shape = (height, width)
            self.shape = (height, width)
            self.resolution = (width, height)  # Store (width, height) for compatibility

            # Make sure we're not in some weird leftover streaming state
            try:
                self.cam.EndAcquisition()
            except Exception:
                pass
            self._acquiring = False

            # ----- Background acquisition thread -----
            # One thread per camera that owns BeginAcquisition/GetNextImage.
            self._frame_queue: "queue.Queue[Optional[np.ndarray]]" = queue.Queue(
                maxsize=1
            )
            self._acq_stop_event = threading.Event()
            self._acq_thread_error: Optional[BaseException] = None
            self._acq_started_event = threading.Event()

            self._acq_thread = threading.Thread(
                target=self._acquisition_worker,
                name=f"FLIR-{serial or 'camera'}",
                daemon=True,
            )
            self._acq_thread.start()

            if verbose:
                print(f"success ({width} x {height}, {bitdepth}-bit)")
        except Exception:
            # Make sure to clear cam_list if an exception happens before cam_list.Clear()
            try:
                cam_list.Clear()
            except Exception:
                pass
            raise

    # SDK / device discovery helpers

    @staticmethod
    def info(verbose: bool = True) -> List[str]:
        """
        Discover all FLIR / Spinnaker cameras reachable via PySpin.
        """
        if PySpin is None:
            if verbose:
                print("PySpin is not available; no FLIR cameras can be detected.")
            return []

        if FLIR.sdk is None:
            FLIR.sdk = PySpin.System.GetInstance()

        cam_list = FLIR.sdk.GetCameras()
        serials: List[str] = []
        try:
            num_cams = cam_list.GetSize()
            for i in range(num_cams):
                cam = cam_list[i]
                tl_nodemap = cam.GetTLDeviceNodeMap()

                node_model = PySpin.CStringPtr(
                    tl_nodemap.GetNode("DeviceModelName")
                )
                model = (
                    node_model.GetValue()
                    if PySpin.IsAvailable(node_model)
                    and PySpin.IsReadable(node_model)
                    else "unknown"
                )

                node_serial = PySpin.CStringPtr(
                    tl_nodemap.GetNode("DeviceSerialNumber")
                )
                if PySpin.IsAvailable(node_serial) and PySpin.IsReadable(node_serial):
                    serial = node_serial.GetValue()
                else:
                    serial = f"cam_{i}"

                serials.append(serial)

                if verbose:
                    print(f"{i}: {serial} ({model})")
        finally:
            cam_list.Clear()

        return serials

    # Camera lifecycle

    def close(self) -> None:
        """See :meth:`.Camera.close`."""
        if PySpin is None or self.cam is None:
            return

        # Stop the background acquisition thread first.
        try:
            if hasattr(self, "_acq_stop_event"):
                self._acq_stop_event.set()
            if hasattr(self, "_acq_thread") and self._acq_thread.is_alive():
                self._acq_thread.join(timeout=1.0)
        except Exception:
            pass

        try:
            if getattr(self, "_acquiring", False):
                try:
                    self.cam.EndAcquisition()
                except Exception:
                    pass
                self._acquiring = False

            try:
                self.cam.DeInit()
            except Exception:
                pass
        finally:
            # Do not release FLIR.sdk here; other FLIR instances may be using it.
            self.cam = None  # type: ignore

    # Exposure control

    def _get_exposure_hw(self) -> float:
        """See :meth:`.Camera._get_exposure_hw`."""
        if PySpin is None:
            raise RuntimeError("PySpin is not available.")
        if self.cam is None:
            raise RuntimeError("Camera has been closed. Cannot get exposure.")
        if not PySpin.IsAvailable(self._exposure_node) or not PySpin.IsReadable(
            self._exposure_node
        ):
            raise RuntimeError("ExposureTime node not readable on FLIR camera.")
        exposure_us = float(self._exposure_node.GetValue())
        return exposure_us * 1e-6

    def _set_exposure_hw(self, exposure_s: float) -> None:
        """See :meth:`.Camera._set_exposure_hw`."""
        if PySpin is None:
            raise RuntimeError("PySpin is not available.")
        if self.cam is None:
            raise RuntimeError("Camera has been closed. Cannot set exposure.")
        if not PySpin.IsAvailable(self._exposure_node) or not PySpin.IsWritable(
            self._exposure_node
        ):
            raise RuntimeError("ExposureTime node not writable on FLIR camera.")

        # Disable auto exposure just in case.
        exposure_auto = PySpin.CEnumerationPtr(self._nodemap.GetNode("ExposureAuto"))
        if PySpin.IsAvailable(exposure_auto) and PySpin.IsWritable(exposure_auto):
            off_entry = exposure_auto.GetEntryByName("Off")
            if (
                off_entry is not None
                and PySpin.IsAvailable(off_entry)
                and PySpin.IsReadable(off_entry)
            ):
                exposure_auto.SetIntValue(off_entry.GetValue())

        exposure_us = float(exposure_s) * 1e6
        try:
            min_us = self._exposure_node.GetMin()
            max_us = self._exposure_node.GetMax()
            exposure_us = max(min_us, min(max_us, exposure_us))
        except Exception:
            # If bounds are not available, just set the requested value.
            pass

        self._exposure_node.SetValue(exposure_us)

    # Window of interest (hardware ROI)

    def set_woi(self, woi=None):
        """See :meth:`.Camera.set_woi`.

        If `woi` is None, this tries to set a full-frame WOI. If the hardware ROI
        nodes are not writable (e.g. locked by TLParamsLocked), we fall back to
        simply *reading* the current ROI and updating the cached attributes.

        Note
        ----
        To avoid race conditions with the background acquisition thread,
        changing WOI after the camera has started streaming is not supported.
        Create a new FLIR instance with the desired WOI instead.
        """
        if PySpin is None:
            raise RuntimeError("PySpin is not available.")

        if self.cam is None:
            raise RuntimeError("Camera has been closed. Cannot set WOI.")

        # Camera.__init__ calls set_woi() before we start the acquisition thread,
        # so this guard only affects user calls *after* initialization.
        if hasattr(self, "_acq_thread") and self._acq_thread.is_alive():
            raise RuntimeError(
                "Changing WOI after acquisition has started is not supported "
                "for FLIR cameras with threaded acquisition. "
                "Please create a new FLIR instance with the desired WOI."
            )

        nm = self._nodemap

        width_node = PySpin.CIntegerPtr(nm.GetNode("Width"))
        height_node = PySpin.CIntegerPtr(nm.GetNode("Height"))
        offsetx_node = PySpin.CIntegerPtr(nm.GetNode("OffsetX"))
        offsety_node = PySpin.CIntegerPtr(nm.GetNode("OffsetY"))

        width_max_node = PySpin.CIntegerPtr(nm.GetNode("WidthMax"))
        height_max_node = PySpin.CIntegerPtr(nm.GetNode("HeightMax"))

        def readable(node):
            return PySpin.IsAvailable(node) and PySpin.IsReadable(node)

        def writable(node):
            return PySpin.IsAvailable(node) and PySpin.IsWritable(node)

        if woi is None:
            # --- Initial call from Camera.__init__ ---
            # If ROI nodes are not writable, just read current ROI and return.
            if not (
                writable(width_node)
                and writable(height_node)
                and writable(offsetx_node)
                and writable(offsety_node)
            ):
                # Best-effort: read existing ROI, or fall back to max size.
                x = int(offsetx_node.GetValue()) if readable(offsetx_node) else 0
                y = int(offsety_node.GetValue()) if readable(offsety_node) else 0

                if readable(width_node):
                    w = int(width_node.GetValue())
                elif readable(width_max_node):
                    w = int(width_max_node.GetValue())
                else:
                    w = self.resolution[0]

                if readable(height_node):
                    h = int(height_node.GetValue())
                elif readable(height_max_node):
                    h = int(height_max_node.GetValue())
                else:
                    h = self.resolution[1]

                self.woi = (x, w, y, h)
                self.shape = (h, w)
                return list(self.woi)

            # Otherwise, set full-frame hardware ROI.
            w_max = int(width_max_node.GetValue())
            h_max = int(height_max_node.GetValue())
            offsetx_node.SetValue(0)
            offsety_node.SetValue(0)
            width_node.SetValue(w_max)
            height_node.SetValue(h_max)
            woi = (0, w_max, 0, h_max)

        else:
            # --- User-requested WOI before acquisition starts ---
            x, w, y, h = [int(v) for v in woi]

            if not (
                writable(width_node)
                and writable(height_node)
                and writable(offsetx_node)
                and writable(offsety_node)
            ):
                raise RuntimeError(
                    "Camera ROI nodes are not writable; cannot set WOI."
                )

            # Clamp ROI to legal ranges
            w_max = int(width_max_node.GetValue())
            h_max = int(height_max_node.GetValue())

            if x < 0 or y < 0:
                raise ValueError("WOI offsets must be non-negative.")
            if w <= 0 or h <= 0:
                raise ValueError("WOI width and height must be positive.")

            if x + w > w_max:
                w = w_max - x
                warnings.warn(
                    "Requested WOI exceeds sensor width; clamping to max."
                )
            if y + h > h_max:
                h = h_max - y
                warnings.warn(
                    "Requested WOI exceeds sensor height; clamping to max."
                )

            offsetx_node.SetValue(x)
            offsety_node.SetValue(y)
            width_node.SetValue(w)
            height_node.SetValue(h)

            woi = (x, w, y, h)

        # Update cached attributes used by Camera.get_image()
        self.woi = woi
        _, w, _, h = woi
        self.shape = (h, w)

        return list(woi)

    # Background acquisition worker

    def _acquisition_worker(self) -> None:
        """Background thread: owns BeginAcquisition / GetNextImage loop."""
        if PySpin is None:
            self._acq_thread_error = RuntimeError("PySpin is not available.")
            self._acq_started_event.set()
            return

        try:
            self.cam.BeginAcquisition()
            self._acquiring = True
            self._acq_started_event.set()
        except Exception as e:
            self._acq_thread_error = e
            self._acq_started_event.set()
            return

        try:
            while not self._acq_stop_event.is_set():
                try:
                    image_result = self.cam.GetNextImage(
                        PySpin.EVENT_TIMEOUT_INFINITE
                    )
                except Exception as e:
                    self._acq_thread_error = e
                    # Push sentinel to wake any waiting get_image() calls.
                    try:
                        if self._frame_queue.full():
                            try:
                                self._frame_queue.get_nowait()
                            except queue.Empty:
                                pass
                        self._frame_queue.put_nowait(None)
                    except queue.Full:
                        pass
                    break

                if image_result.IsIncomplete():
                    # Skip incomplete frames, but keep streaming
                    status = image_result.GetImageStatus()
                    image_result.Release()
                    warnings.warn(
                        f"Incomplete image from FLIR camera (status {int(status)})"
                    )
                    continue

                frame = image_result.GetNDArray()
                image_result.Release()

                # Keep only the most recent frame in the queue.
                try:
                    if self._frame_queue.full():
                        try:
                            self._frame_queue.get_nowait()
                        except queue.Empty:
                            pass
                    self._frame_queue.put_nowait(frame)
                except queue.Full:
                    # Should be rare; just drop frame.
                    pass
        finally:
            try:
                if self._acquiring:
                    self.cam.EndAcquisition()
            except Exception:
                pass
            self._acquiring = False

    # Core image acquisition (used by Camera.get_image / get_images)

    def _get_image_hw(self, timeout_s: float = 0.1):
        """See :meth:`.Camera._get_image_hw`.

        Now implemented by reading from the background thread's frame queue.
        """
        if PySpin is None:
            raise RuntimeError("PySpin is not available.")

        # Check if camera has been closed
        if self.cam is None:
            raise RuntimeError("Camera has been closed. Cannot get image.")

        # If the acquisition thread hit a fatal error, surface it.
        if self._acq_thread_error is not None:
            raise RuntimeError(
                f"FLIR acquisition thread failed: {self._acq_thread_error}"
            )

        # Wait for a frame from the queue.
        try:
            if timeout_s is None or timeout_s <= 0:
                item = self._frame_queue.get()
            else:
                item = self._frame_queue.get(timeout=timeout_s)
        except queue.Empty:
            raise RuntimeError("Timeout waiting for image from FLIR camera.")

        if item is None:
            # Sentinel from worker meaning "I died".
            err = self._acq_thread_error
            if err is None:
                raise RuntimeError("FLIR acquisition thread stopped unexpectedly.")
            raise RuntimeError(f"FLIR acquisition thread failed: {err}")

        return item

    # Internal helpers (unchanged)

    def _configure_stream(self, verbose: bool = True) -> None:
        """Configure streaming buffers for performance (NewestOnly, manual buffer count)."""
        if PySpin is None:
            return

        s_nm = self._stream_nodemap

        # Buffer handling mode: NewestOnly to avoid backlog and large latency.
        handling_mode = PySpin.CEnumerationPtr(
            s_nm.GetNode("StreamBufferHandlingMode")
        )
        if PySpin.IsAvailable(handling_mode) and PySpin.IsWritable(handling_mode):
            entry_newest = handling_mode.GetEntryByName("NewestOnly")
            if (
                entry_newest is not None
                and PySpin.IsAvailable(entry_newest)
                and PySpin.IsReadable(entry_newest)
            ):
                handling_mode.SetIntValue(entry_newest.GetValue())
                if verbose:
                    print("StreamBufferHandlingMode set to NewestOnly")

        # Use manual buffer count with a moderately high value (e.g. 16).
        buffer_mode = PySpin.CEnumerationPtr(s_nm.GetNode("StreamBufferCountMode"))
        buffer_count_node = PySpin.CIntegerPtr(
            s_nm.GetNode("StreamBufferCountManual")
        )

        if (
            PySpin.IsAvailable(buffer_mode)
            and PySpin.IsWritable(buffer_mode)
            and PySpin.IsAvailable(buffer_count_node)
            and PySpin.IsWritable(buffer_count_node)
        ):
            entry_manual = buffer_mode.GetEntryByName("Manual")
            if (
                entry_manual is not None
                and PySpin.IsAvailable(entry_manual)
                and PySpin.IsReadable(entry_manual)
            ):
                buffer_mode.SetIntValue(entry_manual.GetValue())
                desired = 16
                try:
                    min_v = buffer_count_node.GetMin()
                    max_v = buffer_count_node.GetMax()
                    val = max(min_v, min(desired, max_v))
                except Exception:
                    val = desired
                buffer_count_node.SetValue(val)
                if verbose:
                    print(f"Stream buffer count set to {val}")

    def _configure_acquisition_mode(self, verbose: bool = True) -> None:
        """Set continuous freerun acquisition (Continuous + TriggerMode Off)."""
        if PySpin is None:
            return

        # AcquisitionMode = Continuous
        acq_mode = PySpin.CEnumerationPtr(self._nodemap.GetNode("AcquisitionMode"))
        if PySpin.IsAvailable(acq_mode) and PySpin.IsWritable(acq_mode):
            entry_continuous = acq_mode.GetEntryByName("Continuous")
            if (
                entry_continuous is not None
                and PySpin.IsAvailable(entry_continuous)
                and PySpin.IsReadable(entry_continuous)
            ):
                acq_mode.SetIntValue(entry_continuous.GetValue())
                if verbose:
                    print("AcquisitionMode set to Continuous")

        # TriggerMode = Off (freerun)
        trig_mode = PySpin.CEnumerationPtr(self._nodemap.GetNode("TriggerMode"))
        if PySpin.IsAvailable(trig_mode) and PySpin.IsWritable(trig_mode):
            entry_off = trig_mode.GetEntryByName("Off")
            if (
                entry_off is not None
                and PySpin.IsAvailable(entry_off)
                and PySpin.IsReadable(entry_off)
            ):
                trig_mode.SetIntValue(entry_off.GetValue())
                if verbose:
                    print("TriggerMode set to Off (freerun)")

    def _configure_bandwidth(self, num_cams: int, verbose: bool = True) -> None:
        """
        Limit per-camera bandwidth when multiple cameras share a link.

        Uses DeviceLinkSpeed (link speed) and DeviceLinkThroughputLimit* nodes
        if present. This is important for multi-camera setups; without it,
        two full-rate cameras can easily saturate a 1 Gbit/s link and cause
        'Stream has been aborted' errors.
        """
        if PySpin is None:
            return

        nm = self._nodemap

        # Nodes may not exist on all cameras (USB vs GigE, older firmware, etc).
        try:
            speed_node = PySpin.CFloatPtr(nm.GetNode("DeviceLinkSpeed"))
        except Exception:
            speed_node = None
        try:
            limit_mode = PySpin.CEnumerationPtr(
                nm.GetNode("DeviceLinkThroughputLimitMode")
            )
        except Exception:
            limit_mode = None
        try:
            limit_node = PySpin.CIntegerPtr(
                nm.GetNode("DeviceLinkThroughputLimit")
            )
        except Exception:
            limit_node = None

        if (
            speed_node is None
            or not PySpin.IsAvailable(speed_node)
            or not PySpin.IsReadable(speed_node)
            or limit_node is None
            or not PySpin.IsAvailable(limit_node)
            or not PySpin.IsWritable(limit_node)
        ):
            # Nothing we can do; silently skip.
            return

        try:
            link_speed = float(speed_node.GetValue())
        except Exception:
            return

        if link_speed <= 0:
            return

        # Use ~80% of the link, divided across all detected cameras.
        usable_fraction = 0.8
        per_cam = int(usable_fraction * link_speed / max(num_cams, 1))

        # Clamp to allowed range for the node.
        try:
            min_v = limit_node.GetMin()
            max_v = limit_node.GetMax()
            per_cam = max(min_v, min(max_v, per_cam))
        except Exception:
            pass

        # Turn the limit mode on if possible.
        if (
            limit_mode is not None
            and PySpin.IsAvailable(limit_mode)
            and PySpin.IsWritable(limit_mode)
        ):
            try:
                on_entry = limit_mode.GetEntryByName("On")
            except Exception:
                on_entry = None
            if (
                on_entry is not None
                and PySpin.IsAvailable(on_entry)
                and PySpin.IsReadable(on_entry)
            ):
                limit_mode.SetIntValue(on_entry.GetValue())

        try:
            limit_node.SetValue(per_cam)
            if verbose:
                print(
                    f"DeviceLinkThroughputLimit set to ~{per_cam/1e6:.1f} Mbps per camera "
                    f"(num_cams={num_cams})"
                )
        except Exception:
            # Don't hard-fail if the limit can't be set.
            if verbose:
                warnings.warn("Could not set DeviceLinkThroughputLimit; "
                            "multi-camera bandwidth may be too high.")

    def _configure_frame_rate(self, fps: float = 15.0, verbose: bool = True) -> None:
        """
        Limit the camera acquisition frame rate to `fps` (Hz).

        Uses AcquisitionFrameRateEnable + AcquisitionFrameRate nodes if present.
        If those nodes aren't available on this model/firmware, this silently
        does nothing.
        """
        if PySpin is None:
            return

        nm = self._nodemap

        # Enable frame-rate control if such a node exists.
        enable_bool = None
        enable_enum = None

        try:
            enable_bool = PySpin.CBooleanPtr(
                nm.GetNode("AcquisitionFrameRateEnable")
            )
        except Exception:
            enable_bool = None

        try:
            enable_enum = PySpin.CEnumerationPtr(
                nm.GetNode("AcquisitionFrameRateEnable")
            )
        except Exception:
            enable_enum = None

        # Boolean-style enable
        if (
            enable_bool is not None
            and PySpin.IsAvailable(enable_bool)
            and PySpin.IsWritable(enable_bool)
        ):
            try:
                enable_bool.SetValue(True)
            except Exception:
                pass
        # Enumeration-style enable
        elif (
            enable_enum is not None
            and PySpin.IsAvailable(enable_enum)
            and PySpin.IsWritable(enable_enum)
        ):
            try:
                on_entry = enable_enum.GetEntryByName("On")
            except Exception:
                on_entry = None
            if (
                on_entry is not None
                and PySpin.IsAvailable(on_entry)
                and PySpin.IsReadable(on_entry)
            ):
                enable_enum.SetIntValue(on_entry.GetValue())

        # Now set the desired frame rate.
        try:
            fr_node = PySpin.CFloatPtr(nm.GetNode("AcquisitionFrameRate"))
        except Exception:
            fr_node = None

        if (
            fr_node is None
            or not PySpin.IsAvailable(fr_node)
            or not PySpin.IsWritable(fr_node)
        ):
            # Node not present; nothing more to do.
            if verbose:
                warnings.warn(
                    "AcquisitionFrameRate node not writable; cannot limit FPS."
                )
            return

        # Clamp fps into legal range.
        try:
            min_fps = fr_node.GetMin()
            max_fps = fr_node.GetMax()
            fps_clamped = max(min_fps, min(max_fps, fps))
        except Exception:
            fps_clamped = fps

        try:
            fr_node.SetValue(fps_clamped)
            if verbose:
                print(f"AcquisitionFrameRate set to {fps_clamped:.2f} Hz")
        except Exception:
            if verbose:
                warnings.warn("Failed to set AcquisitionFrameRate.")

    def _configure_pixel_format(self, verbose: bool = True) -> int:
        """
        Choose a sensible monochrome pixel format and return its bit depth.

        Tries Mono16 -> Mono12p -> Mono10p -> Mono8 in that order.
        """
        if PySpin is None:
            raise RuntimeError("PySpin is not available.")

        pixel_format_enum = PySpin.CEnumerationPtr(self._nodemap.GetNode("PixelFormat"))
        bitdepth = 8  # reasonable default

        if not PySpin.IsAvailable(pixel_format_enum) or not PySpin.IsWritable(
            pixel_format_enum
        ):
            # Try to infer bitdepth from PixelSize if possible
            pixel_size_node = PySpin.CIntegerPtr(self._nodemap.GetNode("PixelSize"))
            if PySpin.IsAvailable(pixel_size_node) and PySpin.IsReadable(
                pixel_size_node
            ):
                bitdepth = int(pixel_size_node.GetValue())
            if verbose:
                warnings.warn(
                    "PixelFormat node not writable; leaving current format unchanged."
                )
            return bitdepth

        # Preferred formats for BFS-PGE-16S2M (monochrome)
        candidates = [
            ("Mono16", 16),
            ("Mono12p", 12),
            ("Mono10p", 10),
            ("Mono8", 8),
        ]

        for name, bits in candidates:
            try:
                entry = pixel_format_enum.GetEntryByName(name)
            except Exception:
                entry = None
            if (
                entry is not None
                and PySpin.IsAvailable(entry)
                and PySpin.IsReadable(entry)
            ):
                pixel_format_enum.SetIntValue(entry.GetValue())
                bitdepth = bits
                if verbose:
                    print(f"PixelFormat set to {name} ({bits}-bit)")
                break
        else:
            # Fallback: try to infer from PixelSize
            pixel_size_node = PySpin.CIntegerPtr(self._nodemap.GetNode("PixelSize"))
            if PySpin.IsAvailable(pixel_size_node) and PySpin.IsReadable(
                pixel_size_node
            ):
                bitdepth = int(pixel_size_node.GetValue())
            if verbose:
                warnings.warn(
                    "Could not change PixelFormat; using existing format "
                    f"with assumed bitdepth={bitdepth}."
                )

        return bitdepth
