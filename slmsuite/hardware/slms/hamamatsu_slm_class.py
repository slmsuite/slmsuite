# CLASS TO CONTROL THE LCOS-SLM X15213-02 OF HAMAMATSU


import time
import numpy as np
from PIL import Image
from slmsuite.holography import toolbox
from slmsuite.holography import analysis
import numpy as np
from ctypes import *
import copy
import time as tm
import random as rn
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from slmsuite.misc.math import INTEGER_TYPES
from slmsuite.hardware.slms.slm import SLM
import hamamatsu_slm_interpreter as ham






class Hamamatsu(SLM):
    def __init__(
            self,
            resolution = (1272,1024),
            bitdepth = 8,
            name = 'SLM',
            wav_um =0.759,
            wav_design_um = None,
            pitch_um = (12.5,12.5),
            settle_time_s= 0.1,
            frame_mem_dim = 819,
            head_serial_numb = 'LSH0804453\x00',
            bID_size = 1,
            blaze_angle = None, # in degree
            mask_radius = None # in pixel
            ):
        

        check = False
        n_dev,IDs = ham.Open_Device(bID_size = bID_size)
        self.n_dev = n_dev
        self.IDs = IDs
        if n_dev !=0:
            self.ID =  IDs[0]
            h_ser = ham.Check_HeadSerial(bID = self.ID)
            
            if h_ser == head_serial_numb:
                print('The connected device is the right one.')
                self.head_serial_numb = head_serial_numb
                check = True
                
            else:
                print('The connected device is NOT the right one.')      
        
        else: 
            print('Try again to connect the device.')   
        
        if check:
            
            temp =  ham.Check_Temp(self.ID)
            self.head_temperature = temp[0]
            self.controller_temperature = temp[1]
            
            self.led_status = ham.Check_LED(self.ID)
            
            self.frame_mem_dim = frame_mem_dim
            
            self.memory = np.zeros(frame_mem_dim)
            
            self.occupied_slot = 0
            
            self.occupied_slot_list = []
            
            self.display_memory_slot = None
            
            self.width = resolution[0]
            self.height = resolution[1]
            
            self.phase_corr_gray = None
            
            
            self.blaze_angle = blaze_angle # angle at which the produced patterns are shifted from the zero order
        
            self.mask_radius = mask_radius # radius of the circular region used of the display of the SLM
            
            if self.mask_radius is not  None:
                self.mask = display_reduction(dimension= self.mask_radius,value = 0)
                self.mask = self.mask.astype('int8')
                if wav_design_um is not None:
                    riscaled_value = int((2**bitdepth)*(1 - (wav_um/wav_design_um)))
                    if riscaled_value<0:
                        riscaled_value=0
                    self.mask_correction = display_reduction(dimension= self.mask_radius,value = riscaled_value)
                    self.mask_correction = self.mask_correction.astype('int8')
                else:
                    self.mask_correction = np.zeros([self.height,self.width])
                    self.mask_correction = self.mask_correction.astype('int8')
                    
            
            self.wav_um = wav_um
            
            self.dx = pitch_um[0] / self.wav_um
            self.dy = pitch_um[1] / self.wav_um
            
            
            self.phase_correction = None

            # Make normalized coordinate grids.
            xpix = (self.width - 1) *  np.linspace(-.5, .5, self.width)
            ypix = (self.height - 1) * np.linspace(-.5, .5, self.height)
            self.x_grid, self.y_grid = np.meshgrid(self.dx * xpix, self.dy * ypix)
            
            if blaze_angle is not None:
                
                
                blaze_angle_norm = toolbox.convert_blaze_vector(self.blaze_angle, from_units="deg", to_units="norm")
                self.blaze_phase = toolbox.phase.blaze(grid= (self.x_grid, self.y_grid), vector = blaze_angle_norm)
            
            
          
            super().__init__(
                resolution = resolution,
                bitdepth = bitdepth,
                name = name ,
                wav_um = wav_um,
                wav_design_um = wav_design_um,
                pitch_um = pitch_um,
                settle_time_s = settle_time_s
                )



    def check_temperature(self,critical_head_value = 100,critical_controller_value = 100):
        temp = ham.Check_Temp(self.ID)
        self.head_temperature = temp[0]
        self.controller_temperature = temp[1]
        if ((self.head_temperature.value >= critical_head_value) or (self.controller_temperature.value>= critical_controller_value)):
            print('Temperature above the critical value !!!')
        
       
        
    def check_LED(self):
        self.led_status = ham.Check_LED(self.ID)
        
    
    
    def memory_status(self):
        self.occupied_slot_list = []
        self.occupied_slot = 0
        
        for i in range(self.frame_mem_dim):
            if self.memory[i]==1:
                self.occupied_slot_list.append(i)
                self.occupied_slot+=1
        print('Slot occupied:', self.occupied_slot)
        print('Occupied slots list:',self.occupied_slot_list)   
        
      
        
    def show_display(self):
        dislpay_pattern = ham.Check_Disp_IMG(bID = self.ID, x_pixel=self.width, y_pixel = self.height,plot = True)
        self.dislpay = np.array(dislpay_pattern)
        
    
    
    def info(self,verbose = True):
        print(self.name,'info:\n')
        print('Head serial number:',self.head_serial_numb)
        print('Head temperature:',self.head_temperature.value,' Celsius')
        print('Controller temperature:',self.controller_temperature.value,' Celsius')
        print('Led status:',self.led_status)
        print('Slot occupied:', self.occupied_slot)
        print('Occupied slots list:',self.occupied_slot_list)
        print('Memory slot frame on the dislpay:',self.display_memory_slot)
        if verbose == True:
            self.check_dislpay()
    
    

    '''    
    Method called inside the write method of the class SLM used to write on the 
    SLM dislpay. The array must contains integer values.
    '''
    def _write_hw(self, phase,slot_number):
        arr = phase.reshape(self.height*self.width,)
        ham.Write_FMemArray(bID = self.ID,array = arr,x_pixel= self.width,y_pixel= self.height, slot_number = slot_number)
        ham.Change_DispSlot(bID = self.ID,slot_number = slot_number,x_pixel = self.width,y_pixel =self.height,plot=False)
        self.memory[0] = slot_number
  


    def write(
        self,
        phase,
        phase_correct=True,
        blaze = True,
        settle=True,
        slot_number = 0
    ):
        r"""
        Checks, cleans, and adds to data, then sends the data to the SLM and
        potentially waits for settle. This method calls the SLM-specific private method
        :meth:`_write_hw()` which transfers the data to the SLM.

        Warning
        ~~~~~~~
        Subclasses implementing vendor-specific software *should not* overwrite this
        method. Subclasses *should* overwrite :meth:`_write_hw()` instead.

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

            

     
     
            # Add grating if requested 
            if blaze == True and self.blaze_angle is not None:
                self.phase += self.blaze_phase
            
            
            # Pass the data to self.display.
            if zero_phase:
                # If None was passed and phase_correct is False, then use a faster method.
                self.display.fill(0)
            else:
                # Turn the floats in phase space to integer data for the SLM.
                self.display = self._phase2gray(self.phase, out=self.display) 


        # Write!
        if self.mask_radius is not None:
            self.display = self.display*self.mask
            self.display = self.display + self.mask_correction 
            
        self._write_hw(self.display,slot_number = slot_number)

        # Optional delay.
        if settle:
            time.sleep(self.settle_time_s)

        return self.display
    
    
    
    def change_mask_radius(self,new_mask_radius,slot_number = 0):
        self.mask_radius = new_mask_radius
        self.mask = display_reduction(dimension= self.mask_radius)
        self.mask = self.mask.astype('int8')
        
        
        if self.wav_design_um is not None:
            riscaled_value = int((2**self.bitdepth)*(1 - (self.wav_um/self.wav_design_um)))
            if riscaled_value<0:
                riscaled_value=0
            self.mask_correction = display_reduction(dimension= self.mask_radius,value = riscaled_value)
            self.mask_correction = self.mask_correction.astype('int8')
        else:
            self.mask_correction = np.zeros([self.height,self.width])
            self.mask_correction = self.mask_correction.astype('int8')
            
        self.display = self.display*self.mask
        self.display = self.display + self.mask_correction
        self._write_hw(self.display,slot_number = slot_number)
        
        
        
    # NB: after setting the new blaze angle, the current hologram does not change. You need to write it again.
    def change_blaze_angle(self,new_blaze_angle,slot_number = 0):    
        self.blaze_angle = new_blaze_angle
        blaze_angle_norm = toolbox.convert_blaze_vector(self.blaze_angle, from_units="deg", to_units="norm")
        self.blaze_phase = toolbox.phase.blaze(grid= (self.x_grid, self.y_grid), vector = blaze_angle_norm)
    
    
    
    def change_display_slot(self,slot_number,plot = True) :
        ham.Change_DispSlot(bID = self.ID,slot_number = slot_number,plot=plot,x_pixel = self.width,y_pixel =self.height)
        self.dislpay_memory_slot = slot_number
        
        
    
    def check_display(self):
        ham.Check_Disp_IMG(bID = self.ID, x_pixel = self.width, y_pixel = self.height,plot = True)
        
        
      
    def check_mem_slot(self,slot_number,plot = True):    
        ham.Check_FMem_Slot(bID = self.ID, x_pixel = self.width, y_pixel = self.height, slot_number = slot_number,plot = True)

    

    def write_bmp(self,file_path,slot_number):
        '''
        This method save a phase pattern in memory and then display it on the display of the SLM.
        The input pattern must be in bmp format.
        '''

        ham.Write_FMemBMPPath(bID = self.ID,file_path = file_path,slot_number = slot_number,plot=False)
        self.memory[slot_number] =1
        
     
        
    def load_vendor_phase_correction(self, file_path):
        """
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

        

    def display_sequence(self,slot_numbers,plot = False):
        for i in range(len(slot_numbers)):
            self.change_dislpay_slot(slot_numbers[i],plot = False)                
            tm.sleep(self.settle_time_s)
            
        
        
    def memory_controll(self):
        '''
        This method considers only the patterns inserted after the last time that we have 
        create an object with this class
        '''

        self.occupied_slot_list = []
        for i in range(self.frame_mem_dim):
            if self.memory[i] ==1:
               self.occupied_slot_list.append(i)
        self.occupied_slot= len(self.occupied_slot_list)
        print('Slot occupied:', self.occupied_slot)
        print('Occupied slots list:',self.occupied_slot_list)
      
        
    
    def reboot(self):
        ham.Reboot(self.ID)
   
    
    
    def close(self):
        ham.Close_Device(bID_list = self.IDs, bID_size = self.n_dev)
        
      
        
      
        

# Function that reduce the SLM display active area to a circular region with the specified radius 

def display_reduction(
        dimension,  # dimension of the radius in pixels of the display
        N_x = 1272,
        N_y = 1024,
        value =0    # value to which all the pixels outside the working region are setted
        ):

    
    if value == 0:
        phase_mask = np.ones([N_y,N_x])
        r2 = dimension**2
        for i in range(N_x):
            for j in range(N_y):
                a = int(abs(i-N_x/2))
                b = int(abs(j-N_y/2))
                if a**2 + b**2 >r2:
                    phase_mask[j][i]= int(value)
            
    else:
        phase_mask = np.zeros([N_y,N_x])
        if shape == 'circular':
            r2 = dimension**2
            for i in range(N_x):
                for j in range(N_y):
                    a = int(abs(i-N_x/2))
                    b = int(abs(j-N_y/2))
                    if a**2 + b**2 >r2:
                        phase_mask[j][i]= int(value)
            
    return phase_mask                
        


        
        
        
        
        
        
        
        
        
        
        
        
        
        
        