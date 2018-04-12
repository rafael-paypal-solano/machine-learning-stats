from .Info import Info
from pycuda.gpuarray import GPUArray
import pycuda.gpuarray
import numpy
import pycuda
import skcuda.misc

skcuda.misc.init()

def gpu_sum_along_axis(array, axis = None): # TODO: Forget skuda and use kernel functions !!!
    if axis is None:
        return pycuda.gpuarray.sum(array)
    shape = (
        array.shape[0] if axis == 0 else 1,
        array.shape[1] if axis == 1 else 1
    )
    
    out = skcuda.misc.sum(array, axis = axis)
    return out

class Arrays:
    @classmethod
    def create(clazz, sample):        
        if type(sample) is GPUArray:
            return sample

        if Info.gpu_present():
            if type(sample) is numpy.ndarray:
                return pycuda.gpuarray.to_gpu(sample)
            else:
                return pycuda.gpuarray.to_gpu(numpy.array(sample))

        else:
            if type(sample) is numpy.ndarray:
                return sample
            else:
                return numpy.array(sample)
            

    @classmethod
    def sum(clazz, array, axis = None):
        """
            Args:
                array (numpy.ndarray or pycuda.gpuarray.GPUArray): 2D array
                axis (int): like numpy.ndarray.sum's axis parameter
            Return:
                Sum of all elements in the provided array
        """
        
        if not (type(array) is numpy.ndarray or type(array) is GPUArray):
            raise ValueError('array argument type must be either numpy.ndarray or GPUArray')
        
        if type(array) is numpy.ndarray:
            return numpy.sum(array, axis = axis)
        
        return gpu_sum_along_axis(array, axis)
    
        
