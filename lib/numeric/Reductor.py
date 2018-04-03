import numpy
import pathos.multiprocessing as multiprocessing
import scipy.stats as stats

def init_pool(pool):
    if pool is None:
        return multiprocessing.Pool(multiprocessing.cpu_count())

    return pool

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

        n = numpy.sum(processing_pool.map(lambda sample: len(sample),samples))
        s = numpy.square(numpy.sum(processing_pool.map(lambda x: numpy.sum(x), samples)))

        return (s / n)


    @classmethod
    def mean_block(clazz, samples, pool = None): # CM for randomized blocks
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
        s = numpy.sum(processing_pool.map(lambda x: numpy.sum(x), samples))
        return (s / n)

    @classmethod
    def correction_for_mean_block(clazz, samples, pool = None): # Mean for randomized blocks
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
        s = numpy.square(numpy.sum(processing_pool.map(lambda x: numpy.sum(x), samples)))
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

