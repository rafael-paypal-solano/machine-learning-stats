import pandas
import sys
from scipy import stats
import numpy
import numeric

#
# Credits https://academic.macewan.ca/burok/Stat252/notes/RBD.pdf
# https://docs.scipy.org/doc/scipy/reference/tutorial/stats/continuous.html
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.f.html
#

if __name__ == "__main__":
    sample = numpy.array(( # Exercise 13.45
        (11, 13, 16, 10),
        (15, 17, 20, 12),
        (10, 15, 13, 10)
    ))

    k = sample.shape[0]
    b = sample.shape[1]
    n = b * k
    df = n - k -b + 1
    mean = numeric.Reductor.correction_for_mean_block(sample)
    ssb = numeric.Reductor.sum_of_squares_block(sample)
    sst = numeric.Reductor.sum_of_squares_total_block(sample)
    sse = numeric.Reductor.standard_squared_error_block(sample)
    mst = sst / (k-1)
    msb = ssb / (b-1)
    mse = sse / df
    Ft = mst/mse
    Fb = msb/mse
    Pt = stats.f.sf(Ft, k-1, df)
    Pb = stats.f.sf(Fb, b-1, df)

    print("SSB = %8.2f, SST = %8.2f, SSE = %8.2f" % (ssb, sst, sse))
    print("MSB = %8.2f, MST = %8.2f, MSE = %8.2f" % (msb, mst, mse))
    print("F(T) = %8.2f, F(B) = %8.2f" % (Ft, Fb))
    print("P(T) = %f, P(B) = %f" % (Pt, Pb))
