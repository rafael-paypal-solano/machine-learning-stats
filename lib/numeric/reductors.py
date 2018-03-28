import numpy
import pathos.multiprocessing as multiprocessing
import scipy.stats as stats

def init_pool(pool):
    if pool is None:
        return multiprocessing.Pool(multiprocessing.cpu_count())

    return pool

def correction_for_mean(samples, pool = None): # aka CM
    """
        Args:
            samples (tuple of sequences): A sequence is a tuple, list, set or 1-dim numpy.ndarray .
            pool (optional pathos.multiprocessing.Pool or multiprocessing.Pool): Required to parallelize operations.            
        Return:
            (sum of all observations)^2 divided by # of observations 
    """    

    processing_pool = init_pool(pool)

    n = numpy.sum(processing_pool.map(lambda sample: len(sample),samples))
    s = numpy.square(numpy.sum(processing_pool.map(lambda x: numpy.sum(x), samples)))

    return (s / n)


def sum_of_squares_total(samples, pool = None): # aka SST
    """
        Args:
            samples (tuple of sequences): A sequence is a tuple, list, set or 1-dim numpy.ndarray .
            pool (optional pathos.multiprocessing.Pool or multiprocessing.Pool): Required to parallelize operations.            
        Return:
            The sum of the squares of the difference of the dependent variable and its mean (https://en.wikipedia.org/wiki/Total_sum_of_squares)
    """
    Y = samples
    processing_pool = init_pool(pool)
    s = numpy.sum(processing_pool.map(lambda y: numpy.square(numpy.sum(y)) / len(y), Y))
    cm = correction_for_mean(Y, processing_pool)

    return (s - cm)


def total_ss(samples, pool = None): # aka Total SS
    """
        Args:
            samples (tuple of sequences): A sequence is a tuple, list, set or 1-dim numpy.ndarray .
            pool (optional pathos.multiprocessing.Pool or multiprocessing.Pool): Required to parallelize operations.            
        Return:
            Total SS
    """
    processing_pool = init_pool(pool)
    cm = correction_for_mean(samples, pool)
    s = numpy.sum(processing_pool.map(lambda Y: numpy.sum(tuple(map(lambda y: numpy.square(y), Y))), samples))
    return (s - cm)

def standard_squared_error(samples, pool = None): # aka SSE
    """
        Args:
            samples (tuple of sequences): A sequence is a tuple, list, set or 1-dim numpy.ndarray .
            pool (optional pathos.multiprocessing.Pool or multiprocessing.Pool): Required to parallelize operations.            
        Return:
            SSE
    """
    return total_ss(samples, pool) - sum_of_squares_total(samples, pool)

def mean_squared_error(samples, pool =  None):# aka MSE
    """
        Args:
            samples (tuple of sequences): A sequence is a tuple, list, set or 1-dim numpy.ndarray .
            pool (optional pathos.multiprocessing.Pool or multiprocessing.Pool): Required to parallelize operations.            
        Return:
            The mean squared error (MSE) or mean squared deviation (MSD).
    """    
    processing_pool = init_pool(pool)
    n = numpy.sum(processing_pool.map(lambda sample: len(sample),samples))
    return standard_squared_error(samples, pool) / ( n - len(samples))

