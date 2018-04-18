import numpy
import pathos.multiprocessing as multiprocessing
import scipy.stats as stats
import pycuda.autoinit
import pycuda.gpuarray
import pycuda
import pycuda.driver
import skcuda.misc
import skcuda.linalg
import pandas
from .Reductor import Reductor

class CPUReductor(Reductor):
    @classmethod
    def mean_block(clazz, samples): # Mean for randomized blocks
        """
            Args:
                samples (2d array like): .-
            Return:
                (double): Mean for randomized blocks
        """    
        array = Reductor.to_array(samples)
        k = array.shape[0]
        b = array.shape[1]
        n = b * k
        s = numpy.sum(array)
        return (s / n)

    @classmethod
    def correction_for_mean_block(clazz, samples): # CM for randomized blocks
        """
            Args:
                samples (2d array like): .-
            Return:
                (double): CM for randomized blocks
        """    

        array = Reductor.to_array(samples)
        k = array.shape[0]
        b = array.shape[1]
        n = b * k
        s = numpy.square(numpy.sum(array))
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
        array = Reductor.to_array(samples)
        k = array.shape[0]
        column_sum = numpy.sum(numpy.square(numpy.sum(array, axis=0))) / k        

        if cm is None:
            cm = CPUReductor.correction_for_mean_block(array)

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
        array = Reductor.to_array(samples)
        b = array.shape[1]        
        row_sum = numpy.sum(numpy.square(numpy.sum(array, axis=1))) / b        

        if cm is None:
            cm = CPUReductor.correction_for_mean_block(array)

        sst = row_sum- cm
        return sst

    @classmethod    
    def standard_squared_error_block(clazz, samples, cm = None): # SSE for randomized blocks
        """
            Args:
                samples (2d array like): .-
                cm (double):  CM for randomized blocks
            Return:
                (double): SSE for randomized blocks
        """     
        array = Reductor.to_array(samples)
        s = numpy.sum(numpy.square(array))
        
        if cm is None:
            cm = CPUReductor.correction_for_mean_block(array)        

        ssb = CPUReductor.sum_of_squares_block(array)
        sst = CPUReductor.sum_of_squares_total_block(array)
        total_ss = s - cm
        sse = total_ss - ssb - sst
        return sse