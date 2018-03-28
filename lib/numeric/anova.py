import numpy
import pathos.multiprocessing as multiprocessing
import scipy.stats as stats
from .reductors import *

def oneway_anova_interval(samples, i, alpha, pool =  None):
    processing_pool = init_pool(pool)
    n = numpy.sum(processing_pool.map(lambda sample: len(sample),samples))
    k = len(samples)
    Y = numpy.mean(samples[i])
    MSE = mean_squared_error(samples, processing_pool)
    df = n - k
    t = stats.t.isf(alpha/2, df)
    
    return (Y , t * numpy.sqrt(MSE / len(samples[i])))

def oneway_anova_diff_interval(samples, i, j, alpha, pool =  None):
    Y0 = numpy.mean(samples[i])
    Y1 = numpy.mean(samples[j])
    k = len(samples)
    Y = numpy.mean(samples[i])
    MSE = mean_squared_error(samples, processing_pool)
    df = n - k
    return (Y , t * numpy.sqrt(MSE / (len(samples[i]) + len(samples[j]))))