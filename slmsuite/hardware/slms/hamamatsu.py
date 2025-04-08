"""
Hardware control for Hamamatsu SLMs.
Tested with Hamamatsu LCOS-SLM X15213-02.

Note
~~~~
:class:`.Hamamatsu` requires dynamically linked libraries from Hamamatsu to be present in the
runtime directory:

- hpkSLMdaLV.dll
- hpkSLMda.dll

These two files must be in the same directory.

Note
~~~~
Hamamatsu provides base wavefront correction accounting for the curvature of the SLM surface.
Consider loading these files via :meth:`.SLM.load_vendor_phase_correction()`

Note
~~~~
In some methods there is the argument 'slot_number'. This is always set to zero 
and is referred to the number of the frame memory slot of the memory device.
"""


import numpy as np
from PIL import Image
from slmsuite.holography import toolbox
from slmsuite.holography import analysis
from ctypes import *
import copy
import time as tm
import matplotlib.pyplot as plt
from slmsuite.misc.math import INTEGER_TYPES
from slmsuite.hardware.slms.slm import SLM
import sys



# Insert the full path of "hpkSLMdaLV.dll". 
file_name = "path\\to\\dll\\hpkSLMdaLV.dll"

try:
    Lcoslib = cdll.LoadLibrary(file_name) #for cdll
    """
    Lcoslib = windll.LoadLibrary(file_name) #for windll
    """
    
except FileNotFoundError:
    raise RuntimeError('File not found !!!')



class Hamamatsu(SLM):
    r"""
        Initializes an instance of a Hamamatsu SLM.

        Arguments
        ------
        head_serial_numb
            Serial number of the connected device
        others
            See :meth:`.SLM.__init__` for permissible options.

        Note
        ----
        The default values of the parameters are relative to the model 
        LCOS-SLM X15213-02.
    """
    def __init__(
            self,
            resolution = (1272,1024),
            bitdepth = 8,
            name = 'SLM',
            wav_um =0.759,
            wav_design_um = None,
            pitch_um = (12.5,12.5),
            settle_time_s= 0.1,
            head_serial_numb = None
            ):
        
        
        check = False
        bID_size = 1    # Number of connected device
        n_dev,IDs = self._Open_Device(bID_size = bID_size)
        self.n_dev = n_dev
        self.IDs = IDs
        
        if n_dev !=0:
            self.ID =  IDs[0]
            h_ser = self._Check_HeadSerial(bID = self.ID)
            
            if (h_ser == head_serial_numb) and (head_serial_numb != None):
                print('The connected device is the right one.')
                self.head_serial_numb = head_serial_numb
                check = True
                
            else:
                raise RuntimeError('The connected device is NOT the right one !!!')      
        
        else: 
            raise RuntimeError('Device NOT connected !!! Try again.')   
        
        if check:
            
            self.resolution = resolution
            self.phase_corr_gray = None
            self.wav_um = wav_um
            self.dx = pitch_um[0] / self.wav_um
            self.dy = pitch_um[1] / self.wav_um
            self.phase_correction = None

            # Make normalized coordinate grids.
            xpix = (self.resolution[0] - 1) *  np.linspace(-.5, .5, self.resolution[0])
            ypix = (self.resolution[1] - 1) * np.linspace(-.5, .5, self.resolution[1])
            self.x_grid, self.y_grid = np.meshgrid(self.dx * xpix, self.dy * ypix)
            
        
            super().__init__(
                resolution = resolution,
                bitdepth = bitdepth,
                name = name ,
                wav_um = wav_um,
                wav_design_um = wav_design_um,
                pitch_um = pitch_um,
                settle_time_s = settle_time_s
                )




    def _write_hw(self, phase):
        r"""    
            Method called inside the method :meth:'write()' of the SLM class. 
            The array must contains integer values.
        """
        arr = phase.reshape(self.resolution[1]*self.resolution[0],)
        self._Write_FMemArray(bID = self.ID,array = arr,x_pixel= self.resolution[0],y_pixel= self.resolution[1], slot_number = 0)
        self._Change_DispSlot(bID = self.ID,slot_number = 0,x_pixel = self.resolution[0],y_pixel =self.resolution[1],plot=False)

  

    def write(
        self,
        phase,
        phase_correct=True,
        settle=True
    ):
        r"""
        Checks, cleans, and adds to data, then sends the data to the SLM and
        potentially waits for settle. This method calls the SLM-specific private method
        :meth:`_write_hw()` which transfers the data to the SLM.

        Caution
        ~~~~~~~
        The sign on ``phase`` is flipped before converting to integer data. This is to
        convert between
        the 'increasing value ==> increasing voltage (= decreasing phase delay)' convention in most SLMs and
        :mod:`slmsuite`'s 'increasing value ==> increasing phase delay' convention.
        As a result, zero phase will appear entirely white (255 for an 8-bit SLM), and increasing phase
        will darken the displayed pattern.
        If integer data is passed, this data is displayed directly and the sign is *not* flipped.

        Important
        ~~~~~~~~~
        The user does not need to wrap (e.g. :mod:`numpy.mod(data, 2*numpy.pi)`) the passed phase data,
        unless they are pre-caching data for speed (see below).
        :meth:`.write()` uses optimized routines to wrap the phase (see the
        private method :meth:`_phase2gray()`).
        Which routine is used depends on :attr:`phase_scaling`:

         - :attr:`phase_scaling` is one.

            Fast bitwise integer modulo is used. Much faster than the other routines which
            depend on :meth:`numpy.mod()`.

         - :attr:`phase_scaling` is less than one.

            In this case, the SLM has **more phase tuning range** than necessary.
            If the data is within the SLM range ``[0, 2*pi/phase_scaling]``, then the data is passed directly.
            Otherwise, the data is wrapped by :math:`2\pi` using the very slow :meth:`numpy.mod()`.
            Try to avoid this in applications where speed is important.

         - :attr:`phase_scaling` is more than one.

            In this case, the SLM has **less phase tuning range** than necessary.
            Processed the same way as the :attr:`phase_scaling` is less than one case, with the
            important exception that phases (after wrapping) between ``2*pi/phase_scaling`` and
            ``2*pi`` are set to zero. For instance, a sawtooth blaze would be truncated at the tips.

        Caution
        ~~~~~~~
        After scale conversion, data is ``floor()`` ed to integers with ``np.copyto``, rather than
        rounded to the nearest integer (``np.around()`` equivalent). While this is
        irrelevant for the average user, it may be significant in some cases.
        If this behavior is undesired consider either: :meth:`write()` integer data
        directly or modifying the behavior of the private method :meth:`_phase2gray()` in
        a pull request. We have not been able to find an example of ``np.copyto``
        producing undesired behavior, but will change this if such behavior is found.

        Parameters
        ----------
        phase : numpy.ndarray or None
            Phase data to display in units of :math:`2\pi`,
            unless the passed data is of integer type and the data is applied directly.

             - If ``None`` is passed to :meth:`.write()`, data is zeroed.
             - If the array has a larger shape than the SLM shape, then the data is
               cropped to size in a centered manner
               (:attr:`~slmsuite.holography.toolbox.unpad`).
             - If integer data is passed with the same type as :attr:`display`
               (``np.uint8`` for <=8-bit SLMs, ``np.uint16`` otherwise),
               then this data is **directly** passed to the
               SLM, without going through the "phase delay to grayscale" conversion
               defined in the private method :meth:`_phase2gray`. In this situation,
               ``phase_correct`` is **ignored**.
               This is error-checked such that bits with greater significance than the
               bitdepth of the SLM are zero (e.g. the final 6 bits of 16 bit data for a
               10-bit SLM). Integer data with type different from :attr:`display` leads
               to a TypeError.

            Usually, an **exact** stored copy of the data passed by the user under
            ``phase`` is stored in the attribute :attr:`phase`.
            However, in cases where :attr:`phase_scaling` not one, this
            copy is modified to include how the data was wrapped. If the data was
            cropped, then the cropped data is stored, etc. If integer data was passed, the
            equivalent floating point phase is computed and stored in the attribute :attr:`phase`.
        phase_correct : bool
            Whether or not to add :attr:`~slmsuite.hardware.slms.slm.SLM.phase_correction` to ``phase``.
        settle : bool
            Whether to sleep for :attr:`~slmsuite.hardware.slms.slm.SLM.settle_time_s`.

        Returns
        -------
        numpy.ndarray
           :attr:`~slmsuite.hardware.slms.slm.SLM.display`, the integer data sent to the SLM.

        Raises
        ------
        TypeError
            If integer data is incompatible with the bitdepth or if the passed phase is
            otherwise incompatible (not a 2D array or smaller than the SLM shape, etc).
        """
        # Helper variable to speed the case where phase is None.
        zero_phase = False

        # Parse phase.
        if phase is None:
            # Zero the phase pattern.
            self.phase.fill(0)
            zero_phase = True
        else:
            # Make sure the array is an ndarray.
            phase = np.array(phase)

        if phase is not None and isinstance(phase, INTEGER_TYPES):
            # Check the type.
            if phase.dtype != self.display.dtype:
                raise TypeError("Unexpected integer type {}. Expected {}.".format(phase.dtype, self.display.dtype))

            # If integer data was passed, check that we are not out of range.
            if np.any(phase >= self.bitresolution):
                raise TypeError("Integer data must be within the bitdepth ({}-bit) of the SLM.".format(self.bitdepth))

            # Copy the pattern and unpad if necessary.
            if phase.shape != self.shape:
                np.copyto(self.display, toolbox.unpad(phase, self.shape))
            else:
                np.copyto(self.display, phase)

            # Update the phase variable with the integer data that we displayed.
            self.phase = 2 * np.pi - self.display * (2 * np.pi / self.phase_scaling / self.bitresolution)
        else:
            # If float data was passed (or the None case).
            # Copy the pattern and unpad if necessary.
            if phase is not None:
                if self.phase.shape != self.shape:
                    np.copyto(self.phase, toolbox.unpad(self.phase, self.shape))
                else:
                    np.copyto(self.phase, phase)
                    
            # Add phase correction if requested
            if phase_correct:
                if  ("phase" in self.source) and (np.max(self.source['phase']) !=np.min(self.source['phase'])):
                    self.phase +=  self.source["phase"] 
                elif (self.phase_correction is not None):
                    self.phase += self.phase_correction                    
                zero_phase = False
            
            # Pass the data to self.display.
            if zero_phase:
                # If None was passed and phase_correct is False, then use a faster method.
                self.display.fill(0)
            else:
                # Turn the floats in phase space to integer data for the SLM.
                self.display = self._phase2gray(self.phase, out=self.display) 

        self._write_hw(self.display,slot_number = 0)

        # Optional delay.
        if settle:
            tm.sleep(self.settle_time_s)

        return self.display
    

  
    def load_vendor_phase_correction(self, file_path):
        r"""
        Loads vendor-provided phase correction from file,
        setting :attr:`~slmsuite.hardware.slms.slm.SLM.phase_correction`.
        By default, this is interpreted as an image file and is padded or unpadded to
        the shape of the SLM.
        Subclasses should implement vendor-specific routines for loading and
        interpreting the file (e.g. :class:`Santec` loads a .csv).

        Parameters
        ----------
        file_path : str
            File path for the vendor-provided phase correction.

        Returns
        ----------
        numpy.ndarray
            :attr:`~slmsuite.hardware.slms.slm.SLM.phase_correction`,
            the vendor-provided phase correction.
        """
        # Load an invert the image file (see phase sign convention rules in write).
        phase_correction = self.bitresolution - 1 - np.array(Image.open(file_path), dtype=float)

        if phase_correction.ndim != 2:
            raise ValueError("Expected 2D image; found shape {}.".format(phase_correction.shape))

        phase_correction *= 2 * np.pi / (self.phase_scaling * self.bitresolution)

        # Deal with correction shape
        # (this should be made into a toolbox method to supplement pad, unpad)
        file_shape_error = np.sign(np.array(phase_correction.shape) - np.array(self.shape))

        if np.abs(np.diff(file_shape_error)) > 1:
            raise ValueError(
                "Note sure how to pad or unpad correction shape {} to SLM shape {}."
                .format(phase_correction.shape, self.shape)
            )

        if np.any(file_shape_error > 1):
            self.phase_correction = toolbox.unpad(phase_correction, self.shape)
        elif np.any(file_shape_error < 1):
            self.phase_correction = toolbox.pad(phase_correction, self.shape)
        else:
            self.phase_correction = phase_correction
 
        self.phase_correction = -self.phase_correction
        
        return self.phase_correction    
    


    def _Open_Device(bID_size = bID_size):
        r"""
        Establishes the communication with all the LCOS-SLM controllers connected to the USB. 

        NB: make sure that the bID_list has the same lenght of the number of
            connected devices to avoid problem with other functions.
            
        Returns
        -------
        conn_dev : 
            Number of connected devices.
        ID_list :
            ID of the connected devices.
        """
        open_dev = Lcoslib.Open_Dev
        open_dev.argtyes = [c_uint8*bID_size,c_int32]
        open_dev.restype = c_int
        array =c_uint8*bID_size
        ID_list = array(0)
        conn_dev = open_dev(byref(ID_list),bID_size)
        if conn_dev == 0:
            raise RuntimeError("CONNECTION FAILED !!!")
        elif conn_dev ==1 :
            print(conn_dev,'CONNECTED DEVICE')
        else:
            print(conn_dev,' CONNECTED DEVICES')
        return conn_dev, ID_list



    def Close_Device(bID_list,bID_size):
        r"""
        Interrupts the communication with the target devices.
        """
        close_dev = Lcoslib.Close_Dev
        close_dev.argtyes = [c_uint8*bID_size,c_int32]
        close_dev.restype = c_int
        v = close_dev(byref(bID_list),bID_size)
        conn_dev = bID_size
        if v == 1 and bID_size ==1:
            print(bID_size,"The device has been disconnected.")
            conn_dev = 0 
        elif v == 1 and bID_size !=1:
            print(bID_size,"The devices have been disconnected.")
            conn_dev = 0
        else:
            raise RuntimeError("OPERATION FAILED!")



    def _Check_HeadSerial(bID):
        r"""
        Reads the LCOS-SLM head serial number with the desired ID.
        """
        check_serial = Lcoslib.Check_HeadSerial
        check_serial.argtyes = [c_uint8,c_char*11,c_int32]
        hs = c_char*11
        head_serial = hs(0)
        v = check_serial(bID,byref(head_serial),11)
        head_serial = list(head_serial)
        head_serial = [x.decode("utf-8") for x in head_serial]
        head_serial = ''.join(head_serial)
        if v == 1:
            print("The head serial number of the device is: ",head_serial)
        else:
            raise RuntimeError("OPERATION FAILED!")
        return head_serial



    def _Write_FMemArray(bID,array,x_pixel,y_pixel, slot_number):
        r"""
        Writes data to any slot number in the frame memory in array data format. For
        array data, use SXGA as 1D array data with lenght 1272 x 1024 = 1302528. 
        (In this case the dimensions are referred to the display of the  LCOS-SLM X15213-02
         and may be differents for other models of SLM from Hamamatsu).
        
        Parameters
        ----------
        array : 1D numpy.array
            Phase pattern to write in the frame memory of the SLM.
        x_pixel : 
            width of the SLM display in pixels.
        y_pixel : 
            height of the SLM display in pixels.
        """
        write_fmemarray = Lcoslib.Write_FMemArray
        array_size = int(x_pixel*y_pixel)
        arc = c_uint8*array_size
        array_c = arc(0)
        array_c = arc(*array)
        write_fmemarray.argtyes = [c_uint8,c_uint8*array_size,c_int32,c_uint32,c_uint32,c_uint32]
        v = write_fmemarray(bID,byref(array_c), array_size,x_pixel,y_pixel, slot_number)
        if v!=1:
            raise RuntimeError("OPERATION FAILED!")
        return v
    
    
    
    def _Change_DispSlot(bID,slot_number,plot=False,x_pixel = 1272,y_pixel = 1024,stamp = False):
        r"""
        Changes the displayed pattern to the one in the specified slot number, from the frame memory.

        Parameters
        ----------
        plot :
            whether or not to generate the plot of the phase pattern.
        x_pixel : 
            width of the SLM display in pixels.
        y_pixel : 
            height of the SLM display in pixels.
        stamp : 
            whether or not to print a control message.
        """
        change_slot= Lcoslib.Change_DispSlot
        change_slot.argtyes = [c_uint8,c_uint32]
        v = change_slot(bID, slot_number)
        
        if v == 1 and stamp ==True:
            print("Pattern changed into the one located in the frame memory slot number",slot_number,".")
            if plot:
                Check_Disp_IMG(bID=bID, x_pixel=x_pixel, y_pixel=y_pixel)
        if v!=1:
            raise RuntimeError("OPERATION FAILED!")
        return v
    
    

    def Check_Temp(bID):
        r"""
        Reads the temperature of the LCOS-SLM head and controller.
        
        Returns
        -------
        head_temperature :
            temperature SLM's head
        controller_temperature :
            temperature of the SLM's controller
        """
        check_temp = Lcoslib.Check_Temp
        ht = c_double
        head_temperature = ht(0)
        ct = c_double
        controller_temperature = ct(0)
        check_temp.argtyes = [c_uint8,c_double,c_double]
        v = check_temp(bID, byref(head_temperature), byref(controller_temperature))
        if v == 1:
            print("\nHead temperature: ",
                  head_temperature.value,"Celsius\nController temperature: ",
                  controller_temperature.value,"Celsius")
        else:
            raise RuntimeError("OPERATION FAILED!")
    
        
        return head_temperature, controller_temperature



    def Check_LED(bID):
        r"""
        Checks the lighting status of the LED.
        
        Returns
        -------
        led_status : 
            list representing the LED status.
        """
        check_led = Lcoslib.Check_LED
        ls = c_uint32 * 10 
        led_status = ls(0)
        check_led.argtyes = [c_uint8,c_uint32*10]
        v = check_led(bID,byref(led_status))
        led_status = list(led_status)
        if v == 1:
            print("\nLED status:",led_status)
            
        else:
            raise RuntimeError("OPERATION FAILED!")
        return led_status



    def Reboot(bID):
        r"""
        Allows to restart the controller board.
        """
        reboot = Lcoslib.Reboot
        reboot.argtyes = [c_uint8]
        reboot(bID)
        print("USB connection interrupted.")
     
        
     
    def plot_array(array,x_size,y_size):
        r"""
        Plots the image coded in a 1D array in greyscale of 256 shades.

        Parameters
        ----------
        array : 
            phase pattern
        x_size : 
            width of the SLM display in pixels.
        y_size : 
            height of the SLM display in pixels.
        """
        flag  = True
        for i in range(len(array)):
            if (array[i]>= 256 or array[i]<0) and flag :
                raise RuntimeError("Warning: vector not in the correct format. Some value are not in the correct range ([0,255]).")
                flag = False
        if flag:
            display = array.reshape(y_size,x_size)
            plt.imshow(display,cmap = 'gray',vmin = 0,vmax = 255)
            plt.colorbar()  


        

