from .Info import InfoSingleton as Info
from pycuda.gpuarray import GPUArray
import numpy
import pycuda
import skcuda.misc
import pandas

def gpu_sum_along_axis(array, axis = None): # TODO: Forget skuda and use kernel functions !!!
    if axis is None:
        return pycuda.gpuarray.sum(array)
    
    return skcuda.misc.sum(array, axis = axis)

class Arrays:
    @classmethod
    def sum(clazz, array, axis = None):
        """
            Args:
                array (numpy.ndarray or pycuda.gpuarray.GPUArray): 2D array
                axis (int): like numpy.ndarray.sum's axis parameter
            Return:
                Sum of all elements in the provided array
        """
        array_type = type(array)

        if not (array_type is numpy.ndarray or array_type is GPUArray):
            raise ValueError('array argument type must be either numpy.ndarray or GPUArray')
        
        if array_type is numpy.ndarray:
            return numpy.sum(array, axis = axis)
        
        return skcuda.misc.sum(array, axis = axis)

    @classmethod
    def mean(clazz, array, axis = None):
        """
            Args:
                array (numpy.ndarray or pycuda.gpuarray.GPUArray): 2D array
                axis (int): like numpy.ndarray.sum's axis parameter
            Return:
                Sum of all elements in the provided array
        """
        array_type = type(array)

        if not (array_type is numpy.ndarray or array_type is GPUArray):
            raise ValueError('array argument type must be either numpy.ndarray or GPUArray')
        
        if array_type is numpy.ndarray:
            return numpy.mean(array, axis = axis)
        
        return skcuda.misc.mean(array, axis = axis).get()

    @classmethod
    def var(clazz, array, axis = None):
        """
            Args:
                array (numpy.ndarray or pycuda.gpuarray.GPUArray): 2D array
                axis (int): like numpy.ndarray.sum's axis parameter
            Return:
                Sum of all elements in the provided array
        """
        array_type = type(array)

        if not (array_type is numpy.ndarray or array_type is GPUArray):
            raise ValueError('array argument type must be either numpy.ndarray or GPUArray')
        
        if array_type is numpy.ndarray:
            return numpy.var(array, axis = axis)
        
        return skcuda.misc.var(array, axis = axis).get()

    @classmethod
    def to_array(clazz, samples):
        """
            Args:
                samples(pandas.DataFrame, numpy.ndarray or pycuda.gpuarray.GPUArray): Source data

            Returns:
                if samples is a pandas.DataFrame, then it converts it into a numpy.ndarray or pycuda.gpuarray.GPUArray depending on gpu availability
        """

        if pycuda.driver.Device.count() == 0:
            if (type(samples) is numpy.ndarray) or (type(samples) is pandas.DataFrame):
                return samples
            else:
                return numpy.array(samples)

        if type(samples) is pycuda.gpuarray.GPUArray:
            return samples
        elif type(samples) is pandas.DataFrame:
            return pycuda.gpuarray.to_gpu(samples.values)
        else:
            return pycuda.gpuarray.to_gpu(numpy.array(samples))        

   