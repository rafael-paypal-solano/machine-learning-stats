import pandas
import sys
from scipy import stats
import numpy
import numeric



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

    print("\tSST:\t\t\t%12.4f" % numeric.sum_of_squares_total(samples))    
    print("\tSSE:\t\t\t%12.4f" % numeric.standard_squared_error(samples))
    print("\tMSE:\t\t\t%12.4f" %numeric.mean_squared_error(samples))
    print("\tConfidence Intervals:")

    if P < alpha:
        for i in range(0, len(samples)):
            mean, length = numeric.oneway_anova_interval(samples, i, alpha)
            print("\t%12.4f %12.4f" % (mean, length))