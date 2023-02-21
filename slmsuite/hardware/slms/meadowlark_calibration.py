"""
Meadowlark SLMs accept 8-bit images as inputs to display. However, without proper calibration,
the output from the SLM will not be linear with respect to the input. Through a 2-column lookup
table (LUT), you can access 2048 (or 4096, depending on the model) values. By assigning the correct
values from the possible 2048 (4096) "voltage" values to each of the 256 display values, you can
get achieve a linear output response.
This file provides automatic generation a LUT for SLMs being used in "amplitude mode", for example,
phase SLMs with a polarizing beamsplitter + half-wave plate at the input.
(The light incident on the SLM is polarized at 45 degrees after rotation by the half-wave plate,
so the SLM effectively performs pixel-wise polarization rotation of the beam. The polarizing beamsplitter
then rejects light in the unrotated polarization, providing amplitude modulation.)
This file generates calibrated.lut, a lookup table that makes the values 0, 1, 2, ..., 255
correspond to a linear increase in intensity on your camera.
Tested with Meadowlark (AVR Optics) P1920-400-800-HDMI-T.
~~~~
Note: this file requires SLM.py (which defines the SLM class) to be in the same directory.
~~~~
Note: when a new calibrated lookup table is loaded onto the SLM, some hardware errors can cause it
to deviate from a linear response, so, as can be seen below, we refine the calibration
by calling the calibrating function a second time.
~~~~
Note: this file assumes an acquisition by a Thorlabs camera, with the associated ThorCam SDK installed
Tested with camera model CS165MU1.
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt
from thorlabs_tsi_sdk.tl_camera import TLCameraSDK
from SLM import SLM
from scipy.interpolate import interp1d

# Create SLM object
slm = SLM()
slm_display_shape = (slm.width,slm.height)

# User options
os.chdir("C:\\ONN_with_SLM\\SDK\\Python Compact Scientific Camera Toolkit\\dlls\\64_lib") # need to change directory for Thorlabs camera
frames_per_trigger = 1 # number of frames to average per software trigger sent to camera
cam_exposure_time = 8 # camera exposure time in ms (make sure you aren't saturating the camera)
just_test_lut = False # skip calibration, and instead just test lookup file currently called in SLM.py
sleep_time = .2 # time to pause between SLM display and camera acquisition
roi_limits = (510, 595, 627, 715) # pixel values on the camera that define the area over which we perform the spatial average (region of interest defined as roi[0]:roi[1],roi[2]:roi[3])
num_available_vals_SLM = 2048 # maximum value available on the SLM (can chose from 2^11 or 2^12 values depending on Meadowlark model)
# There may be multiple cycles of maxima and minima over the full >2*pi phase shift provided by the SLM.
# The two variables below allow you to define the range within which there is only one max and one min that bound a monotonic
# increase in output value in Fig. 10. If you prefer an automatic search, set each value to zero.
idx_lower_bound_g = 0 # e.g., 150 (max 255)
idx_upper_bound_g = 0 # e.g., 195 (max 255)

# Paths for outputs
my_path = 'C:\\Program Files\\Meadowlark Optics\\Blink 1920 HDMI\\LUT Files\\'
fname_outputs_linear_lut = my_path + 'outputs_linear_lut.txt'
fname_outputs_calibrated_lut_init = my_path + 'outputs_calibrated_lut_init.txt'
fname_outputs_calibrated_lut_refined = my_path + 'outputs_calibrated_lut_refined.txt'
fname_initial_lut = my_path + 'initial_calibration.lut'
fname_refined_lut = my_path + 'calibrated.lut'

linear_vect_256 = np.arange(256)
linear_vect_long = np.arange(num_available_vals_SLM)

# Define a function to acquire an image with your camera
def acquire(camera):
    camera.issue_software_trigger()
    frame = camera.get_pending_frame_or_null() # acquires in uint16
    # Try to collect a frame a few times - Thorlabs cameras sometimes skip a frame
    j = 0 # don't get stuck in endless loop if frame isn't coming in
    while (frame is None) & (j < 10):
        print('Output was null in a frame')
        frame = camera.get_pending_frame_or_null()
        j = j+1
    if j == 9:
        raise ValueError("No frame arrived within the timeout!")
    # shape camera output into 2D numpy array
    im = frame.image_buffer.reshape(camera.image_height_pixels, camera.image_width_pixels)
    im = im.astype('double')
    im = np.asarray(im)
    return im

# Resample a vector y = f(x) at new query points: y_resampled = f(xq)
def resample(x, y, xq):
    y_resampled = np.zeros(xq.shape[0])
    f = interp1d(x, y, kind='cubic')
    # quick hack to stay within interpolation range
    y_resampled[0:-8] = f(xq[0:-8])
    y_resampled[-8:] = y_resampled[-9]
    return y_resampled

# Create a LUT based on LUT_vals vector
def save_2col_file(filename, LUT_vals, format):
    np.savetxt(filename, np.concatenate((np.expand_dims(linear_vect_256,1), np.expand_dims(LUT_vals,1)),axis=1), fmt=format, delimiter=' ')
    return

# Define a pattern that creates a uniform display on the SLM
def uniform_for_SLM(shape, max_val):
    uniform = max_val*np.ones((shape[1],shape[0]))
    return uniform

# Return the average value of the ROI of the 2D input array im
def spatially_average_frame(im, roi):
    im_cropped = im[roi[0]:roi[1],roi[2]:roi[3]]
    output_val = np.sum(im_cropped)/im_cropped.shape[0]/im_cropped.shape[1]
    return output_val

# Display an array on SLM for uniform camera output, stepping through the values in linear_vect_256
def display_acquire_avg_SLM_sequence(cam, roi, time_to_sleep, slm):
    im_avged = np.zeros(linear_vect_256.shape[0])
    print('Begin linear sweep through LUT values (256 values in total). Currently at number:')
    for i in range(linear_vect_256.shape[0]):
        matrix_to_display = uniform_for_SLM(slm_display_shape,linear_vect_256[i])
        slm.display_matrix(np.flipud(matrix_to_display)) # Flip matrix for display because of imaging
        time.sleep(time_to_sleep) # pause to let the SLM update before grabbing the camera frame
        im = acquire(cam)
        im_avged[i] = spatially_average_frame(im, roi)
        if (i % 20) == 0:
            print(i)
    print(255)
    fig = plt.figure(num=1)
    plt.imshow(im, cmap='gray')
    plt.title('Raw image acquired at max. value in LUT')
    fig = plt.figure(num=2)
    plt.imshow(im[roi[0]:roi[1],roi[2]:roi[3]], cmap='gray')
    plt.title('Raw image acquired at max. value in LUT, ROI only')
    print('Linear sweep complete.')
    # set minimum to zero, then normalize
    im_avged = im_avged - np.min(im_avged)
    im_avged = im_avged/np.max(im_avged)
    return im_avged

# Smooth an input - you may want to edit this function if your signal is very noisy
def smooth(x):
    window = np.hanning(30)
    window = window / window.sum()
    y = np.convolve(x, window, mode='same')
    y[0:10] = x[10] # hack to eliminate potential strange behavior on edge
    y[-10:] = x[-10]
    return y

# Step through current LUT values to generate new LUT that results in linear output response from input display values
def generate_new_lut(cam, uncalibrated_lut_filename, calibrated_lut_filename, output_filename, uncalibrated_lut_values, is_first_calibration):

    # load uncalibrated LUT into SLM
    print('Loading LUT from: ' + uncalibrated_lut_filename)
    slm.slm_lib.Load_lut(uncalibrated_lut_filename)
    print('LUT loaded')

    # step through all 256 voltage values, displayed uniformly on the SLM, then calculate the spatial average of each image over the region of interest (roi)
    outputs = display_acquire_avg_SLM_sequence(cam, roi_limits, sleep_time, slm)
    save_2col_file(output_filename, outputs, ['%u','%.3f'])
    
    # resample to have num_available_vals_SLM values instead of 256
    linear_vect_256_resampled = np.arange(0, 256, 256/num_available_vals_SLM)
    outputs_resampled = resample(linear_vect_256, outputs, linear_vect_256_resampled)

    if is_first_calibration:
        # find the indices of the voltages for the min and max values in our desired voltage region
        # (where there is a monotonic increase from min to max output)
        if idx_upper_bound_g == 0:
            outputs_resampled_temp = smooth(outputs_resampled)
            plt.figure(num=5)
            plt.plot(linear_vect_long, outputs_resampled_temp)
            plt.plot(linear_vect_long, outputs_resampled)
            idx_lower_bound = np.argmin(outputs_resampled_temp) # find index of minimum value
            idx_upper_bound = np.argmax(np.diff(outputs_resampled_temp[idx_lower_bound:]) < 0) + idx_lower_bound # find index of following maximum value
            idx_lower_bound = max(0,idx_lower_bound-10)
            idx_upper_bound = min(num_available_vals_SLM,idx_upper_bound+10)
        else:
            idx_lower_bound = int(idx_lower_bound_g*num_available_vals_SLM/256)
            idx_upper_bound = int(idx_upper_bound_g*num_available_vals_SLM/256)
        range_min_idx = np.argmin(outputs_resampled[idx_lower_bound:idx_upper_bound])+idx_lower_bound
        range_max_idx = np.argmax(outputs_resampled[idx_lower_bound:idx_upper_bound])+idx_lower_bound
        # only keep outputs within the desired range
        outputs_resampled = outputs_resampled[range_min_idx:(range_max_idx+1)]
        linear_vect_256_resampled = linear_vect_256_resampled[range_min_idx:(range_max_idx+1)]
        slm_vals = linear_vect_long[range_min_idx:(range_max_idx+1)]
    else:
        # resample voltage vals in LUT
        slm_vals = resample(linear_vect_256, uncalibrated_lut_values, linear_vect_256_resampled)

    # define the minimum as 0 and normalize so the maximum is 1
    outputs_resampled = outputs_resampled - np.min(outputs_resampled)
    outputs_resampled = outputs_resampled/np.max(outputs_resampled)

    # fit a 9th-order polynomial to get SLM voltage vals vs. outputs
    calib_fit = np.polyfit(255*outputs_resampled, slm_vals, 9)
    # determine required SLM voltage vals to have linear outputs
    calib_fit_vals = np.round(np.polyval(calib_fit, linear_vect_256))

    if not is_first_calibration:
        calib_fit_vals[0] = uncalibrated_lut_values[0] # make sure we keep true zero

    # create the new LUT from these data
    save_2col_file(calibrated_lut_filename, calib_fit_vals, ['%u','%u'])

    # plot the results
    if is_first_calibration:
        plt.figure(num=10)
    else:
        plt.figure(num=20)
    # fig, (ax0, ax1) = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
    plt.plot(linear_vect_256, outputs, label='Full vector')
    plt.xlabel('Sent to SLM')
    plt.ylabel('Received intensity')
    plt.title('Calibration curve')
    if is_first_calibration:
        plt.plot(linear_vect_256_resampled, outputs_resampled, label='Segment used for new LUT')
        plt.legend()

    return calib_fit_vals

with TLCameraSDK() as camera_sdk:
    available_cameras = camera_sdk.discover_available_cameras()
    if len(available_cameras) < 1:
        raise ValueError("no cameras detected")
    with camera_sdk.open_camera(available_cameras[0]) as camera:
        # adjust camera settings
        camera.frames_per_trigger_zero_for_unlimited = frames_per_trigger # number of frames to average per trigger
        camera.operation_mode = 0 # 0 means software triggered, 1 means hardware triggered
        camera.trigger_polarity = 1
        camera.image_poll_timeout_ms = 4000  # 4 second timeout
        camera.arm(2)
        camera.exposure_time_us = cam_exposure_time*1000 # exposure time in us

        # the first 2 SLM displays are invalid - let's throw 3 away for good measure
        slm.display_matrix(uniform_for_SLM(slm_display_shape, 0))
        slm.display_matrix(uniform_for_SLM(slm_display_shape, 1))
        slm.display_matrix(uniform_for_SLM(slm_display_shape, 2))

        if not just_test_lut:
            # create a linear LUT (i.e., uncalibrated)
            save_2col_file(my_path+'linear_for_first_calib.lut', num_available_vals_SLM/256*linear_vect_256, ['%u','%u'])
            
            # load the linear LUT into SLM, get the averaged camera outputs, and calculate a new, calibrated LUT
            print('Begin initial calibration.')
            calib_fit0_vals = generate_new_lut(camera, my_path+'linear_for_first_calib.lut', fname_initial_lut, fname_outputs_linear_lut, 0, True)
            print('End initial calibration.')

            # load our calculated LUT into SLM, get the averaged camera outputs, and recalibrate
            print('Begin calibration refinement.')
            calib_fit1_vals = generate_new_lut(camera, fname_initial_lut, fname_refined_lut, fname_outputs_calibrated_lut_init, calib_fit0_vals, False)
            print('End calibration refinement.')

            slm.slm_lib.Load_lut(fname_refined_lut) # load new (refined) LUT into SLM

        # test final LUT
        print('Test final LUT by running sweep again.')
        outputs_calibrated_lut2 = display_acquire_avg_SLM_sequence(camera, roi_limits, sleep_time, slm)
        save_2col_file(fname_outputs_calibrated_lut_refined, outputs_calibrated_lut2, ['%u','%.3f'])
        print('Test complete.')
        
        # plot final results
        plt.figure(num=30)
        plt.plot(linear_vect_256, 255*outputs_calibrated_lut2, label='Data')
        plt.plot(linear_vect_256, linear_vect_256, 'r--', label='Ideal (y=255*x)')
        plt.xlabel('Sent to SLM')
        plt.ylabel('Received intensity')
        plt.legend()
        plt.title('Final outputs with calibrated LUT')
        plt.show()