# MODULE FOR THE CONTROL OF THE LCOS-SLM X15213-02 OF HAMAMATSU

# Head serial number of the SLM used: 'LSH0804453'
# Display width: 1272 px ; Dislpay height: 1024 px



import numpy as np
from ctypes import *
import copy
import time as tm
import sys
import random as rn
import matplotlib.pyplot as plt
import matplotlib.image as mpimg




# Insert the full path of the file "hpkSLMdaLV.dll". 
# NB: "hpkSLMdaLV.dll" and "hpkSLMda.dll" must be in the same directory


file_name = "path\\to\\dll\\hpkSLMdaLV.dll"

try:
    Lcoslib = cdll.LoadLibrary(file_name) #for cdll
    #Lcoslib = windll.LoadLibrary(file_name) #for windll
except FileNotFoundError:
    print('File not found !')



# 1
'''
Establishes the communication with all the LCOS SLM controllers connected to the USB. 

NB: make sure that the bID_list has the same lenght of the number of
    connected devices to avoid problem with other functions.
'''
def Open_Device(bID_size = 1):
    open_dev = Lcoslib.Open_Dev
    open_dev.argtyes = [c_uint8*bID_size,c_int32]
    open_dev.restype = c_int
    array =c_uint8*bID_size
    ID_list = array(0)
    conn_dev = open_dev(byref(ID_list),bID_size)
    if conn_dev == 0:
        print("CONNECTION FAILED !!!")
    elif conn_dev ==1 :
        print(conn_dev,'CONNECTED DEVICE')
    else:
        print(conn_dev,' CONNECTED DEVICES')
    return conn_dev, ID_list



# 2
'''
Interrupts the communication with the target devices.
'''
def Close_Device(bID_list,bID_size):
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
        print("OPERATION FAILED!")
    return conn_dev



#3
'''
Reads the LCOS-SLM head serial number.
'''
def Check_HeadSerial(bID):
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
        print("OPERATION FAILED!")
    return head_serial



# 4 
'''
Writes data to any slot number in the frame memory by specifying the BMP file 
path. BMP files should be a SXGA with dimensions 1272 x 1024.
(In this case the dimensions are referred to the display of the  LCOS-SLM X15213-02
 and may be differents for other models of SLM from Hamamatsu)

'''
def Write_FMemBMPPath(bID,file_path,slot_number,plot=False):
    #Lcoslib = cdll.LoadLibrary(file_name) #for cdll
    Lcoslib = windll.LoadLibrary(file_name) #for windll
    write_fmembmp = Lcoslib.Write_FMemBMPPath
    lenght = int(len(file_path))
    write_fmembmp.argtyes = [c_uint8,c_char*lenght,c_uint32]
    fpc = c_char*lenght
    file_path_c = fpc('0'.encode('utf-8'))
    for i in range(lenght):
        file_path_c[i] = file_path[i].encode('utf-8')
    v = write_fmembmp(bID,file_path_c,slot_number)
    if v == 1:
        print("Pattern inserted in the frame memory.")
        if plot:
            image = mpimg.imread(file_path) 
            plt.show()
            plt.imshow(image)
    else:
        print("OPERATION FAILED!")
    return v



# 5
'''
Writes data to any slot number in the frame memory in array data format. For
array data, use SXGA as 1D array data with lenght 1272 x 1024 = 1302528. 
(In this case the dimensions are referred to the display of the  LCOS-SLM X15213-02
 and may be differents for other models of SLM from Hamamatsu).
'''
def Write_FMemArray(bID,array,x_pixel,y_pixel, slot_number,stamp = False):
    write_fmemarray = Lcoslib.Write_FMemArray
    array_size = int(x_pixel*y_pixel)
    arc = c_uint8*array_size
    array_c = arc(0)
    array_c = arc(*array)
    write_fmemarray.argtyes = [c_uint8,c_uint8*array_size,c_int32,c_uint32,c_uint32,c_uint32]
    v = write_fmemarray(bID,byref(array_c), array_size,x_pixel,y_pixel, slot_number)
    if v == 1 and stamp ==True:
        print("Pattern inserted in the slot number ",slot_number,".")
    if v!=1:
        print("OPERATION FAILED!")
    return v



# 6 
''' 
Changes the displayed pattern to the one in the specified slot number, from the frame memory.
'''
def Change_DispSlot(bID,slot_number,plot=False,x_pixel = 1272,y_pixel = 1024,stamp = False):
    change_slot= Lcoslib.Change_DispSlot
    change_slot.argtyes = [c_uint8,c_uint32]
    v = change_slot(bID, slot_number)
    
    if v == 1 and stamp ==True:
        print("Pattern changed into the one located in the frame memory slot number",slot_number,".")
        if plot:
            Check_Disp_IMG(bID=bID, x_pixel=x_pixel, y_pixel=y_pixel)
    if v!=1:
        print("OPERATION FAILED!")
    return v



# 7
'''
Reads the temperature of the LCOS-SLM head and controller.
'''
def Check_Temp(bID):
    check_temp = Lcoslib.Check_Temp
    ht = c_double
    head_temperature = ht(0)
    ct = c_double
    controller_temperature = ct(0)
    check_temp.argtyes = [c_uint8,c_double,c_double]
    v = check_temp(bID, byref(head_temperature), byref(controller_temperature))
    if v == 1:
        print("Operation performed successfully. \nHead temperature: ",
              head_temperature.value,"Celsius\nController temperature: ",
              controller_temperature.value,"Celsius")
    else:
        print("OPERATION FAILED!")
    return head_temperature, controller_temperature



# 8 
'''
Reads the image array data stored in the specified slot number of the frame 
memory.
'''
def Check_FMem_Slot(bID, x_pixel, y_pixel, slot_number,plot = True):
    check_slot = Lcoslib.Check_FMem_Slot
    check_slot.argtyes = [c_uint8, c_int32,c_uint32,c_uint32,c_uint32,c_uint8]
    array_size = int(x_pixel*y_pixel)
    arc = c_uint8*array_size
    array_c = arc(0)
    v = check_slot(bID,array_size, x_pixel, y_pixel, slot_number,byref(array_c))
    array_c_fp = np.array(array_c)
    if v == 1:
        print("Pattern in the frame memory slot:",slot_number)
        if plot:
            plot_array(array_c_fp,x_pixel,y_pixel)     
    else:
        print("OPERATION FAILED!")
    return array_c



# 9 
'''
Reads the image array data displayed in the LCOS-SLM head.
'''
def Check_Disp_IMG(bID, x_pixel, y_pixel,plot = True):
    array_size = int(x_pixel*y_pixel)
    arr = c_uint8 * array_size
    array = arr(0)
    check_disp = Lcoslib.Check_Disp_IMG
    check_disp.argtyes = [c_uint8,c_uint32,c_uint32,c_uint32,c_uint8*array_size]
    v = check_disp(bID, array_size, x_pixel, y_pixel, byref(array))
    arr_fp = np.array(array)
    if v == 1:
        print("Operation performed successfully.")
        if plot == True:
            plot_array(arr_fp,x_pixel,y_pixel)
    else:
        print("OPERATION FAILED!")
    return array
    
 

# 12 
'''
Checks the lighting status of the LED.
'''
def Check_LED(bID):
    check_led = Lcoslib.Check_LED
    ls = c_uint32 * 10 
    led_status = ls(0)
    check_led.argtyes = [c_uint8,c_uint32*10]
    v = check_led(bID,byref(led_status))
    led_status = list(led_status)
    if v == 1:
        print("Operation performed successfully.\nLED status:",led_status)
        
    else:
        print("OPERATION FAILED!")
    return led_status



# 14  
'''
Allows to restart the controller board.
'''
def Reboot(bID):
    reboot = Lcoslib.Reboot
    reboot.argtyes = [c_uint8]
    reboot(bID)
    print("USB connection interrupted.")
 
    

# Plot array
'''
Plots the image coded in a 1D array in greyscale of 256 shades.
'''
def plot_array(array,x_size,y_size):
    flag  = True
    for i in range(len(array)):
        if (array[i]>= 256 or array[i]<0) and flag :
            print("Warning: vector not in the correct format. Some value are not in the correct range ([0,255]).")
            flag = False
    if flag:
        display = array.reshape(y_size,x_size)
        plt.imshow(display,cmap = 'gray',vmin = 0,vmax = 255)
        plt.colorbar()  


    

    
    
    
    
    
    
