from environment import Arrays
import numpy
from pycuda.gpuarray import GPUArray
import pycuda.gpuarray 
import pycuda
import skcuda.misc
import skcuda.linalg
import pycuda.autoinit
from .Reductor import Reductor

skcuda.misc.init()
class GPUReductor(Reductor):

    @classmethod
    def mean_block(clazz, samples): # Mean for randomized blocks
        """
            Args:
                samples (2d array like): .-
            Return:
                (double): Mean for randomized blocks
        """    
        array = Arrays.to_array(samples)
        k = array.shape[0]
        b = array.shape[1]
        n = b * k
        s = numpy.atleast_1d(skcuda.misc.sum(array).get())[0]
        return (s / n)

    @classmethod
    def correction_for_mean_block(clazz, samples): # CM for randomized blocks
        """
            Args:
                samples (2d array like): .-
            Return:
                (double): CM for randomized blocks
        """    
        array = Arrays.to_array(samples)
        k = array.shape[0]
        b = array.shape[1]
        n = b * k
        s = numpy.square(numpy.atleast_1d(skcuda.misc.sum(array).get())[0])
        return (s / n)

    @classmethod
    def sum_of_squares_block(clazz, samples, cm = None): # SSB
        """
            Args:
                samples (2d array like): .-
                cm (double):  CM for randomized blocks
            Return:
                (double): Sum of squares for randomized block
        """      
        array = Arrays.to_array(samples)
        k = array.shape[0]        
        column_sum = numpy.sum(numpy.square(skcuda.misc.sum(array, axis=0).get())) / k        

        if cm is None:
            cm = GPUReductor.correction_for_mean_block(array)

        ssb = column_sum- cm
        return ssb

    @classmethod
    def sum_of_squares_total_block(clazz, samples, cm = None): # SST for randomized blocks
        """
            Args:
                samples (2d array like): .-
                cm (double):  CM for randomized blocks
            Return:
                (double): SST for randomized blocks
        """      
        array = Arrays.to_array(samples)
        b = array.shape[1]        
        row_sum = numpy.sum(numpy.square(skcuda.misc.sum(array, axis=1).get())) / b        

        if cm is None:
            cm =  GPUReductor.correction_for_mean_block(array)

        sst = row_sum- cm
        return sst

    @classmethod    
    def standard_squared_error_block(clazz, samples, cm = None): # SSE for randomized blocks
        """
            Args:
                samples (square array like): .-
                cm (double):  CM for randomized blocks
            Return:
                (double): SSE for randomized blocks
        """   
        array = Arrays.to_array(samples)
        s =  numpy.square(skcuda.linalg.norm(array))
        
        if cm is None:
            cm =  GPUReductor.correction_for_mean_block(array)        

        ssb = GPUReductor.sum_of_squares_block(array)
        sst = GPUReductor.sum_of_squares_total_block(array)
        total_ss = s - cm
        sse = total_ss - ssb - sst

        return sse

