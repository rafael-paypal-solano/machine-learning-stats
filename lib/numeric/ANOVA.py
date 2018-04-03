import numpy
import pathos.multiprocessing as multiprocessing
import scipy.stats as stats
from .Reductor import Reductor
import itertools

def init_pool(pool):
    if pool is None:
        return multiprocessing.Pool(multiprocessing.cpu_count())

    return pool
class ANOVA:

    @classmethod
    def oneway_anova_interval(clazz, samples, i, alpha, pool =  None):
        processing_pool = init_pool(pool)
        n = numpy.sum(processing_pool.map(lambda sample: len(sample),samples))
        k = len(samples)
        Y = numpy.mean(samples[i])
        MSE = Reductor.mean_squared_error(samples, processing_pool)
        df = n - k
        t = stats.t.isf(alpha/2, df)
        
        return (Y , t * numpy.sqrt(MSE / len(samples[i])), i)

    @classmethod
    def oneway_anova_diff_interval(clazz, samples, i, j, alpha, pool =  None):
        processing_pool = init_pool(pool)     
        n = numpy.sum(processing_pool.map(lambda sample: len(sample),samples))   
        Y0 = numpy.mean(samples[i])
        Y1 = numpy.mean(samples[j])
        k = len(samples)
        Y = Y0 - Y1
        MSE = Reductor.mean_squared_error(samples, processing_pool)
        df = n - k
        t = stats.t.isf(alpha/2, df)
        return (Y , t * numpy.sqrt(MSE * (1/len(samples[i]) + 1/len(samples[j]))), (i,j))

    @classmethod
    def oneway_anova_diff_intervals(clazz, samples, alpha):
        indexes = itertools.combinations(range(0, len(samples)), 2)
        intervals = tuple(map(lambda index: ANOVA.oneway_anova_diff_interval(samples, index[0], index[1], alpha) , indexes))
        return intervals