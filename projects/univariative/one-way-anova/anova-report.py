import pandas
import sys
from scipy import stats
import pathos.multiprocessing as multiprocessing
import numpy


def CM(samples, pool = None):
    """
        Args:
            samples (tuple of sequences): A sequence is a tuple, list, set or 1-dim numpy.ndarray .
            pool (optional pathos.multiprocessing.Pool or multiprocessing.Pool): Required to parallelize operations.            
        Return:

    """    

    if pool is None:
        processing_pool = multiprocessing.Pool(multiprocessing.cpu_count())
    else:
        processing_pool = pool

    n = numpy.sum(tuple(map(lambda sample: len(sample),samples)))
    s = numpy.square(numpy.sum(processing_pool.map(lambda x: numpy.sum(x), samples)))

    return (s / n)

if __name__ == "__main__":
    input_file = sys.argv[1]
    alpha = 0.05
    dataset = pandas.read_csv(input_file,  infer_datetime_format = True)
    
    samples = tuple(map(lambda c: dataset[c].dropna(), dataset.columns))    
    F, P = stats.f_oneway(*samples)
    index = 0

    print("BASIC STATS")
    
    for statistic in map(lambda sample: stats.describe(sample), samples):
        print(
            "\t %s: n=%d, mean=%10.4f, variance=%10.4f, skewness=%10.4f, kurtosis=%10.4f" % (
                dataset.columns[index], statistic.nobs, statistic.mean, statistic.variance, statistic.skewness, statistic.kurtosis
            )
        )        
        index += 1

    print()
    print("ONE-WAY ANOVA")    
    print("\tCritical Values: F =%12.4f, P=%8.5f" % (F, P))
    print("\tF-Test: Are all means equals (H0) ?  %s " % ("Reject" if P < alpha else "Don't Reject")  )

    print(standard_squared_error(samples))
