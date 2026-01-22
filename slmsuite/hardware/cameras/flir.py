"""
Hardware control for FLIR cameras via the :mod:`PySpin` interface to the Spinnaker SDK.
Install PySpin using the `provided instructions
<https://www.teledynevisionsolutions.com/support/support-center/technical-guidance/iis/installing-pyspin-for-the-spinnaker-sdk/>`_.

This implementation uses the QuickSpin API for simplified property access.
Refer to the PySpin documentation (included with installation) for 
details on alternative approaches using the full Spinnaker API.

"""
import warnings

from .camera import Camera

try:
    import PySpin
except ImportError:
    PySpin = None
    warnings.warn("PySpin not installed. Install to use FLIR cameras.")

class FLIR(Camera):
    """
    FLIR camera.

    Attributes
    ----------
    sdk : PySpin.System
        Spinnaker SDK singleton. Shared among instances of :class:`FLIR`.
    cam : PySpin.Camera
        Object to talk with the desired camera.
    camera_list : PySpin.CameraList
        List of available cameras for cleanup.
    """

    sdk = None

    ### Initialization and termination ###
    def __init__(self, serial="", pitch_um=None, verbose=True, **kwargs):
        """
        Initialize camera and attributes.

        Parameters
        ----------
        serial : str
            Serial number of the camera to open.
            Use :meth:`.info()` to see detected options.
            If empty, defaults to the first camera in the list
            returned by :meth:`PySpin.System.GetCameras()`.
        pitch_um : (float, float) OR None
            Fill in extra information about the pixel pitch in ``(dx_um, dy_um)`` form
            to use additional calibrations.
        verbose : bool
            Whether or not to print extra information.
        **kwargs
            See :meth:`.Camera.__init__` for permissible options.
        """
        if PySpin is None:
            raise ImportError("PySpin not installed. Install to use FLIR cameras.")

        # Initialize SDK singleton if needed
        if FLIR.sdk is None:
            if verbose:
                print("PySpin initializing... ", end="")
            FLIR.sdk = PySpin.System.GetInstance()

        # Get camera list
        if verbose:
            print("Looking for cameras... ", end="")
        self.camera_list = FLIR.sdk.GetCameras()

        # Build serial list and validate camera selection
        num_cameras = self.camera_list.GetSize()
        serial_list = []
        for i in range(num_cameras):
            cam_temp = self.camera_list.GetByIndex(i)
            nodemap_tldevice = cam_temp.GetTLDeviceNodeMap()
            node_serial = PySpin.CStringPtr(nodemap_tldevice.GetNode('DeviceSerialNumber'))
            if PySpin.IsReadable(node_serial):
                serial_list.append(node_serial.GetValue())

        if serial == "":
            if num_cameras == 0:
                raise RuntimeError("No cameras found by PySpin.")
            if num_cameras > 1 and verbose:
                print(f"No serial given... Choosing first of {serial_list}")
            self.cam = self.camera_list.GetByIndex(0)
            # Get actual serial for naming
            nodemap_tldevice = self.cam.GetTLDeviceNodeMap()
            node_serial = PySpin.CStringPtr(nodemap_tldevice.GetNode('DeviceSerialNumber'))
            if PySpin.IsReadable(node_serial):
                serial = node_serial.GetValue()
        else:
            if serial in serial_list:
                self.cam = self.camera_list.GetBySerial(serial)
            else:
                raise RuntimeError(
                    f"Serial {serial} not found by PySpin. Available: {serial_list}"
                )

        # Initialize camera
        if verbose:
            print(f"PySpin sn '{serial}' initializing... ", end="")

        try:
            self.cam.Init()
        except PySpin.SpinnakerException as ex:
            raise RuntimeError(f"Failed to initialize camera: {ex}")

        # Configure camera properties
        try:
            # Turn off automatic modes for manual control
            if self.cam.GainAuto.GetAccessMode() == PySpin.RW:
                self.cam.GainAuto.SetValue(PySpin.GainAuto_Off)
            if self.cam.ExposureAuto.GetAccessMode() == PySpin.RW:
                self.cam.ExposureAuto.SetValue(PySpin.ExposureAuto_Off)
            if self.cam.AdcBitDepth.GetAccessMode() == PySpin.RW:
                self.cam.AdcBitDepth.SetValue(PySpin.AdcBitDepth_Bit10)
            if self.cam.PixelFormat.GetAccessMode() == PySpin.RW:
                self.cam.PixelFormat.SetValue(PySpin.PixelFormat_Mono8)

            # Configure software trigger
            if self.cam.TriggerMode.GetAccessMode() == PySpin.RW:
                self.cam.TriggerMode.SetValue(PySpin.TriggerMode_Off)
            if self.cam.TriggerSource.GetAccessMode() == PySpin.RW:
                self.cam.TriggerSource.SetValue(PySpin.TriggerSource_Software)
            if self.cam.TriggerMode.GetAccessMode() == PySpin.RW:
                self.cam.TriggerMode.SetValue(PySpin.TriggerMode_On)

        except PySpin.SpinnakerException as ex:
            raise RuntimeError(f"Failed to configure camera: {ex}")

        # Begin acquisition 
        try:
            self.cam.BeginAcquisition()
        except PySpin.SpinnakerException as ex:
            raise RuntimeError(f"Failed to begin acquisition: {ex}")

        super().__init__(
            (self.cam.WidthMax.GetValue(), self.cam.HeightMax.GetValue()),
            bitdepth=int(self.cam.PixelSize.ToString()[3:]),
            pitch_um=pitch_um,
            name=serial,
            **kwargs
        )

        if verbose:
            print(f"Successfully initialized FLIR cam {serial}.")

    def close(self, close_sdk=True):
        """
        See :meth:`.Camera.close`.

        Parameters
        ----------
        close_sdk : bool
            Whether or not to close the PySpin System instance.
        """
        try:
            self.cam.EndAcquisition()
        except:
            pass

        try:
            self.cam.DeInit()
        except:
            pass

        # Clean up camera list
        if hasattr(self, 'camera_list'):
            try:
                self.camera_list.Clear()
            except:
                pass
            del self.camera_list

        del self.cam

        if close_sdk:
            self.close_sdk()

    @staticmethod
    def info(verbose=True):
        """
        Discovers all FLIR cameras.

        Parameters
        ----------
        verbose : bool
            Whether to print the discovered information.

        Returns
        --------
        list of str
            List of FLIR serial numbers.
        """
        if PySpin is None:
            raise ImportError("PySpin not installed. Install to use FLIR cameras.")

        # Note: We don't close the SDK in info() because PySpin holds references
        # to cameras that prevent clean shutdown. The SDK will be cleaned up when
        # the last camera instance calls close_sdk() or at program exit.
        if FLIR.sdk is None:
            FLIR.sdk = PySpin.System.GetInstance()

        try:
            camera_list = FLIR.sdk.GetCameras()
            num_cameras = camera_list.GetSize()
            serial_list = []

            for i in range(num_cameras):
                cam = camera_list.GetByIndex(i)
                nodemap_tldevice = cam.GetTLDeviceNodeMap()
                node_serial = PySpin.CStringPtr(nodemap_tldevice.GetNode('DeviceSerialNumber'))
                if PySpin.IsReadable(node_serial):
                    serial_list.append(node_serial.GetValue())
                # Don't hold references to individual cameras
                del cam

            if verbose:
                print("PySpin serials:")
                for serial in serial_list:
                    print(f"'{serial}'")

            # Clear camera list
            camera_list.Clear()
            del camera_list

        except PySpin.SpinnakerException as ex:
            raise RuntimeError(f"Failed to enumerate cameras: {ex}")

        return serial_list

    @classmethod
    def close_sdk(cls):
        """
        Close the PySpin System instance.
        """
        if cls.sdk is not None:
            cls.sdk.ReleaseInstance()
            cls.sdk = None

    ### Property Configuration ###

    def get_properties(self, verbose=True):
        """
        Print the list of camera properties using GenICam nodemap traversal.

        Parameters
        ----------
        verbose : bool
            Whether to print properties to console. If False, returns property dict.

        Returns
        -------
        dict or None
            Dictionary of {property_name: property_value} if verbose=False, else None.
        """
        properties = {}

        def traverse_category(category_node):
            """Recursively traverse category nodes to find actual properties."""
            try:
                features = category_node.GetFeatures()

                for feature in features:
                    # Skip if not readable
                    if not PySpin.IsReadable(feature):
                        continue

                    try:
                        # Check node type
                        node_type = feature.GetPrincipalInterfaceType()

                        # If it's a category, recurse into it
                        if node_type == PySpin.intfICategory:
                            category = PySpin.CCategoryPtr(feature)
                            print(f"\nCategory: {category.ToString()}\n")
                            traverse_category(category)
                        else:
                            # This is an actual property node - extract info
                            node = PySpin.CValuePtr(feature)
                            name = node.GetName()

                            # Try to get value as string
                            try:
                                value = node.ToString()
                            except:
                                value = "N/A"

                            # Try to get unit
                            try:
                                unit = node.GetUnit()
                            except:
                                unit = ""

                            # Try to get description
                            try:
                                description = node.GetToolTip()
                            except:
                                description = ""

                            properties[name] = value

                            if verbose:
                                output = f"{name}\t{value}"
                                if unit:
                                    output += f"\t{unit}"
                                if description:
                                    output += f"\t{description}"
                                print(output)

                    except Exception:
                        continue

            except Exception:
                pass

        try:
            nodemap = self.cam.GetNodeMap()

            # Get root category
            root = PySpin.CCategoryPtr(nodemap.GetNode("Root"))
            if not PySpin.IsReadable(root):
                if verbose:
                    print("Unable to access camera properties")
                return properties if not verbose else None

            # Recursively traverse all categories to find properties
            traverse_category(root)

        except PySpin.SpinnakerException as ex:
            if verbose:
                print(f"Error accessing properties: {ex}")

        return properties if not verbose else None 

    def _get_exposure_hw(self):
        """See :meth:`.Camera._get_exposure_hw`."""
        return float(self.cam.ExposureTime.GetValue()) / 1e6

    def _set_exposure_hw(self, exposure_s):
        """See :meth:`.Camera._set_exposure_hw`."""
        self.cam.ExposureTime.SetValue(float(exposure_s * 1e6))

    def set_woi(self, woi=None):
        """
        See :meth:`.Camera.set_woi`.

        Note: WOI changes require stopping and restarting acquisition.
        """
        # Get max WOI from nodemap (use WidthMax/HeightMax, not sensor size)
        maxwoi = (0, self.cam.WidthMax.GetValue(), 0, self.cam.HeightMax.GetValue())

        if woi is None:
            woi = maxwoi

        x, w, y, h = woi

        # WOI changes require stopping acquisition
        acquisition_active = False
        try:
            # Check if acquisition is running
            if self.cam.IsStreaming():
                self.cam.EndAcquisition()
                acquisition_active = True
        except PySpin.SpinnakerException:
            pass

        try:
            # Get nodemap and nodes
            nodemap = self.cam.GetNodeMap()
            node_height = PySpin.CIntegerPtr(nodemap.GetNode('Height'))
            node_width = PySpin.CIntegerPtr(nodemap.GetNode('Width'))
            node_offset_x = PySpin.CIntegerPtr(nodemap.GetNode('OffsetX'))
            node_offset_y = PySpin.CIntegerPtr(nodemap.GetNode('OffsetY'))

            # Set to minimum dimensions first to avoid offset constraint violations
            if PySpin.IsWritable(node_height):
                node_height.SetValue(node_height.GetMin())
            if PySpin.IsWritable(node_width):
                node_width.SetValue(node_width.GetMin())

            # Set offsets
            if PySpin.IsWritable(node_offset_x):
                node_offset_x.SetValue(int(x))
            if PySpin.IsWritable(node_offset_y):
                node_offset_y.SetValue(int(y))

            # Set desired dimensions
            if PySpin.IsWritable(node_height):
                node_height.SetValue(int(h))
            if PySpin.IsWritable(node_width):
                node_width.SetValue(int(w))

            # Update stored WOI on success
            self.woi = woi

        except PySpin.SpinnakerException as ex:
            # Restart acquisition before re-raising
            if acquisition_active:
                try:
                    self.cam.BeginAcquisition()
                except:
                    pass
            raise RuntimeError(f"Failed to set WOI: {ex}")

        finally:
            # Restart acquisition if it was active
            if acquisition_active:
                try:
                    self.cam.BeginAcquisition()
                except PySpin.SpinnakerException as ex:
                    raise RuntimeError(f"Failed to restart acquisition after WOI change: {ex}")

    def _get_image_hw(self, timeout_s):
        """
        See :meth:`.Camera._get_image_hw`.

        Uses software trigger to capture each frame.

        Parameters
        ----------
        timeout_s : float
            Timeout in seconds.
        """
        try:
            # Execute software trigger
            self.cam.TriggerSoftware.Execute()

            # Get triggered image
            frame = self.cam.GetNextImage(int(timeout_s * 1e3))

            # Check if image is incomplete
            if frame.IsIncomplete():
                status = frame.GetImageStatus()
                frame.Release()
                raise RuntimeError(f"Image incomplete with status {status}")

            # Get numpy array from image
            image_data = frame.GetNDArray()

            # Release frame to free buffer
            frame.Release()

            return image_data

        except PySpin.SpinnakerException as ex:
            raise RuntimeError(f"Camera acquisition failed: {ex}")