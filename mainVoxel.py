#!/usr/bin/python
import os
import numpy as np
import ctypes
from ctypes import *	
import sys
import scipy.misc
import matplotlib.pyplot as plt
from struct import unpack

file = str(sys.argv[1])
class gadget:
    def __init__(self, file_in):
        #--- Open Gadget file
        file = open(file_in,'rb')
        #--- Read header
        dummy = file.read(4)                
        self.npart                     = np.fromfile(file, dtype='i', count=6)
        self.massarr                = np.fromfile(file, dtype='d', count=6)
        self.time                      = (np.fromfile(file, dtype='d', count=1))[0]
        self.redshift                 = (np.fromfile(file, dtype='d', count=1))[0]
        self.flag_sfr                  = (np.fromfile(file, dtype='i', count=1))[0]
        self.flag_feedback       = (np.fromfile(file, dtype='i', count=1))[0]
        self.nparttotal              = np.fromfile(file, dtype='i', count=6)
        self.flag_cooling          = (np.fromfile(file, dtype='i', count=1))[0]
        self.NumFiles               = (np.fromfile(file, dtype='i', count=1))[0]
        self.BoxSize                  = (np.fromfile(file, dtype='d', count=1))[0]
        self.Omega0                 = (np.fromfile(file, dtype='d', count=1))[0]
        self.OmegaLambda      = (np.fromfile(file, dtype='d', count=1))[0]
        self.HubbleParam         = (np.fromfile(file, dtype='d', count=1))[0]
        self.header                    = file.read(256-6*4 - 6*8 - 8 - 8 - 2*4-6*4 -4 -4 -4*8)
        dummy = file.read(4)
        #--- Read positions
        c = (self.npart[0]+self.npart[1]+self.npart[2]+self.npart[3]+self.npart[4]+self.npart[5])
        dummy = file.read(4)

        #self.pos = np.fromfile(file, dtype='f', count=self.npart[0]*3)
        self.pos = np.fromfile(file, dtype='f', count=c*3)        
        
        file.close()

        #self.pos = self.pos.reshape((self.npart[0],3))
        self.pos = self.pos.reshape((c,3))        
       
s = gadget(file)
os.chdir(".")
os.system("clear")
os.system("rm *.so")
os.system("make")

dll = ctypes.CDLL('./libr3d.so', mode=ctypes.RTLD_GLOBAL)

def get_avg():
    global dll
    func = dll.calc_avg    
    func.restype = c_float
    func.argtypes = [c_size_t, POINTER(c_float)]    
    return func

__average = get_avg()

def get_medium():
    global dll
    func = dll.calc_medium
    func.argtypes = [c_size_t, POINTER(c_float)]
    return func

__medium = get_medium()

def get_stdev():
    global dll
    func = dll.calc_StDev
    func.argtypes = [c_size_t, POINTER(c_float)]
    return func

__sd = get_stdev()

def get_max():
    global dll
    func = dll.calc_Max
    func.argtypes = [c_size_t, POINTER(c_float)]
    return func

__max = get_max()

def get_min():
    global dll
    func = dll.calc_Min
    func.argtypes = [c_size_t, POINTER(c_float)]
    return func

__min = get_min()

def get_fft():
    global dll
    func = dll.calc_FFT
    func.argtypes = [c_size_t, POINTER(c_float)]
    return func

__fft = get_fft()

#def voxelization(size, pos, res):
def voxelization(size, pos, res):
    
    return res

#__vox = voxelization()

# convenient python wrapper
# it does all job with types convertation
# from python ones to C++ ones 
def cuda_average(size, pos):
    pos = pos.ctypes.data_as(POINTER(c_float))
    __average(size, pos)        

def cuda_medium(size, pos):
    pos = pos.ctypes.data_as(POINTER(c_float))
    __medium(size, pos)

def cuda_stdev(size, pos):
    pos = pos.ctypes.data_as(POINTER(c_float))
    __sd(size, pos)

def cuda_Max(size, pos):
    pos = pos.ctypes.data_as(POINTER(c_float))
    __max(size, pos)

def cuda_Min(size, pos):
    pos = pos.ctypes.data_as(POINTER(c_float))
    __min(size, pos)

def cuda_FFT(size, pos):
    pos = pos.ctypes.data_as(POINTER(c_float))
    __fft(size, pos)

if __name__ == '__main__':    
    size=len(s.pos)
    res = np.ones(5)
    Nres = np.ones(5)
    #cuda_average(size, s.pos)
    #cuda_medium(size, s.pos)
    #cuda_stdev(size, s.pos)    
    #cuda_MaxMin(size, s.pos)    
    #cuda_FFT(size, s.pos)
    #print size
    #print res
    #print Nres
    #Nres = voxelization(size, s.pos, res)
    x = voxelization(size, s.pos, res)
    print (x)