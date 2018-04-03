import numpy
import pathos.multiprocessing as multiprocessing
import scipy.stats as stats

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


class Reductor:

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
                pool (optional pathos.multiprocessing.Pool or multiprocessing.Pool): Required to parallelize operations.            
            Return:
                (sum of all observations)^2 divided by b*n where b == # of blocks and k == # of treatments.
        """    

        processing_pool = init_pool(pool)
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
                pool (optional pathos.multiprocessing.Pool or multiprocessing.Pool): Required to parallelize operations.            
            Return:
                (sum of all observations)^2 divided by b*n where b == # of blocks and k == # of treatments.
        """    

        array = to_array(samples)
        k = array.shape[0]
        b = array.shape[1]
        n = b * k
        s = numpy.square(numpy.sum(array))
        return (s / n)

    @classmethod
    def sum_of_squares_block(clazz, samples): # SSB
        array = to_array(samples)
        k = array.shape[0]
        column_sum = numpy.sum(numpy.square(numpy.sum(array, axis=0))) / k        
        cm = Reductor.correction_for_mean_block(array)
#       print(column_sum, cm, k)
        ssb = column_sum- cm
        return ssb

    @classmethod
    def sum_of_squares_total_block(clazz, samples): # SST for randomized blocks
        array = to_array(samples)
        b = array.shape[1]        
        row_sum = numpy.sum(numpy.square(numpy.sum(array, axis=1))) / b        
        cm = Reductor.correction_for_mean_block(array)
#       print(row_sum, cm, b)
        sst = row_sum- cm
        return sst

    @classmethod    
    def standard_squared_error_block(clazz, samples): # aka SSE blocks
        array = to_array(samples)
        s = numpy.sum(numpy.square(array))
        cm = Reductor.correction_for_mean_block(array)        
        ssb = Reductor.sum_of_squares_block(array)
        sst = Reductor.sum_of_squares_total_block(array)
        total_ss = s - cm
        sse = total_ss - ssb - sst

        return sse

    