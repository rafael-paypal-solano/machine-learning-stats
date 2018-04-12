import numpy
import pathos.multiprocessing as multiprocessing
import scipy.stats as stats

#
#   TODO: 
#    1) Install CuPy https://docs-cupy.chainer.org/en/stable/install.html#dependencies
#    2) Make randomized block functions GPU Aware: If GPU is present use it, otherwise default to numpy (http://stsievert.com/blog/2016/07/01/numpy-gpu/).
#
def to_array(samples):
    if type(samples) is numpy.ndarray:
        return samples
    else:
        return numpy.array(samples)

def init_pool(pool):
    if pool is None:
        return multiprocessing.Pool(multiprocessing.cpu_count())

    return pool

def cell_count(samples, processing_pool):
    if type(samples) is numpy.ndarray:
        return samples.size
    else:
        return numpy.sum(processing_pool.map(lambda sample: len(sample), samples))
    

def cell_sum(samples, processing_pool):
    if type(samples) is numpy.ndarray:
        return numpy.sum(samples)
    else:
        return numpy.sum(processing_pool.map(lambda x: numpy.sum(x), samples))

class GPUReductor:
    pass

class CPUReductor:
    @classmethod
    def mean_block(clazz, samples): # Mean for randomized blocks
        """
            Args:
                samples (2d array like): .-
            Return:
                (double): Mean for randomized blocks
        """    
        array = samples if type(samples) is numpy.ndarray else numpy.array(samples)
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

        array = to_array(samples)
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
        array = to_array(samples)
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
        array = to_array(samples)
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
        array = to_array(samples)
        s = numpy.sum(numpy.square(array))
        
        if cm is None:
            cm = CPUReductor.correction_for_mean_block(array)        

        ssb = CPUReductor.sum_of_squares_block(array)
        sst = CPUReductor.sum_of_squares_total_block(array)
        total_ss = s - cm
        sse = total_ss - ssb - sst

        return sse

class Reductor:
    implementation = CPUReductor

    @classmethod
    def to_array(clazz, samples):
        return to_array(samples)

    @classmethod    
    def correction_for_mean(clazz, samples, pool = None): # aka CM
        """
            Args:
                samples (2d array like): .-
                pool (optional pathos.multiprocessing.Pool or multiprocessing.Pool): Required to parallelize operations.            
            Return:
                (sum of all observations)^2 divided by # of observations 
        """    

        processing_pool = init_pool(pool)        
        n = cell_count(samples, processing_pool)
        s = numpy.square(cell_sum(samples, processing_pool))

        return (s / n)

        
    @classmethod    
    def sum_of_squares_total(clazz, samples, pool = None): # aka SST
        """
            Args:
                samples (2d array like): .-
                pool (optional pathos.multiprocessing.Pool or multiprocessing.Pool): Required to parallelize operations.            
            Return:
                The sum of the squares of the difference of the dependent variable and its mean (https://en.wikipedia.org/wiki/Total_sum_of_squares)
        """
        Y = samples
        processing_pool = init_pool(pool)
        s = numpy.sum(processing_pool.map(lambda y: numpy.square(numpy.sum(y)) / len(y), Y))
        cm = Reductor.correction_for_mean(Y, processing_pool)

        return (s - cm)


    @classmethod    
    def total_ss(clazz, samples, pool = None): # aka Total SS
        """
            Args:
                samples (2d array like): .-
                pool (optional pathos.multiprocessing.Pool or multiprocessing.Pool): Required to parallelize operations.            
            Return:
                Total SS
        """
        processing_pool = init_pool(pool)
        cm = Reductor.correction_for_mean(samples, pool)
        s = numpy.sum(processing_pool.map(lambda Y: numpy.sum(tuple(map(lambda y: numpy.square(y), Y))), samples))
        return (s - cm)

    @classmethod    
    def standard_squared_error(clazz, samples, pool = None): # aka SSE
        """
            Args:
                samples (2d array like): .-
                pool (optional pathos.multiprocessing.Pool or multiprocessing.Pool): Required to parallelize operations.            
            Return:
                SSE
        """
        return Reductor.total_ss(samples, pool) - Reductor.sum_of_squares_total(samples, pool)

    @classmethod    
    def mean_squared_error(clazz, samples, pool =  None):# aka MSE
        """
            Args:
                samples (2d array like): .-
                pool (optional pathos.multiprocessing.Pool or multiprocessing.Pool): Required to parallelize operations.            
            Return:
                The mean squared error (MSE) or mean squared deviation (MSD).
        """    
        processing_pool = init_pool(pool)
        n = numpy.sum(processing_pool.map(lambda sample: len(sample),samples))
        return Reductor.standard_squared_error(samples, pool) / ( n - len(samples))

 

    @classmethod
    def mean_block(clazz, samples): # Mean for randomized blocks
        """
            Args:
                samples (2d array like): .-
            Return:
                (double): Mean for randomized blocks
        """    

        return clazz.implementation.mean_block(samples)

    @classmethod
    def correction_for_mean_block(clazz, samples): # CM for randomized blocks
        """
            Args:
                samples (2d array like): .-
            Return:
                (double): CM for randomized blocks
        """    

        return clazz.implementation.correction_for_mean_block(samples)

    @classmethod
    def sum_of_squares_block(clazz, samples, cm = None): # SSB
        """
            Args:
                samples (2d array like): .-
                cm (double):  CM for randomized blocks
            Return:
                (double): Sum of squares for randomized block
        """      
        return clazz.implementation.sum_of_squares_block(samples, cm)

    @classmethod
    def sum_of_squares_total_block(clazz, samples, cm = None): # SST for randomized blocks
        """
            Args:
                samples (2d array like): .-
                cm (double):  CM for randomized blocks
            Return:
                (double): SST for randomized blocks
        """      
        return clazz.implementation.sum_of_squares_total_block(samples, cm)

    @classmethod    
    def standard_squared_error_block(clazz, samples, cm = None): # SSE for randomized blocks
        """
            Args:
                samples (2d array like): .-
                cm (double):  CM for randomized blocks
            Return:
                (double): SSE for randomized blocks
        """     
        return clazz.implementation.standard_squared_error_block(samples, cm)

        