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
    input_file = sys.argv[1]
    alpha = 0.05
    sample = pandas.read_csv(input_file,  infer_datetime_format = True)    
    
    block = numeric.ANOVARandomizedBlock(sample)
    print("SSB = %8.2f, SST = %8.2f, SSE = %8.2f" % (block.ssb, block.sst, block.sse))
    print("MSB = %8.2f, MST = %8.2f, MSE = %8.2f" % (block.msb, block.mst, block.mse))
    print("F(T) = %8.2f, F(B) = %8.2f" % (block.Ft, block.Fb))
    print("P(T) = %f, P(B) = %f" % (block.Pt, block.Pb))
