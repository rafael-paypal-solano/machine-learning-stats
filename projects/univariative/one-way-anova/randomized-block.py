import pandas
import sys
from scipy import stats
import numpy
import numeric
from numeric import Reductor, CPUReductor, GPUReductor
import pycuda.autoinit
import pycuda.gpuarray
import skcuda.misc
skcuda.misc.init()
#
# Credits https://academic.macewan.ca/burok/Stat252/notes/RBD.pdf
# https://docs.scipy.org/doc/scipy/reference/tutorial/stats/continuous.html
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.f.html
# https://ssor.vcu.edu/media/statistics/resources/spss/SPSS.Blocks.PC.pdf
#

def print_intervals(intervals, title):
    print(title)
    for interval in intervals:
            print("(%d,%d): Δμ=%10.4f, length=%10.4f, Δμ ± length = (%10.4f, %10.4f)" % (
                interval.pair[0], 
                interval.pair[1], 
                interval.mean, 
                interval.length, 
                interval.mean -  interval.length, 
                interval.mean + interval.length)
            )

def print_report(array, reductor):
    alpha = 0.05
    block = numeric.ANOVARandomizedBlock(array, 0.05, reductor)
    print("SSB = %8.2f, SST = %8.2f, SSE = %8.2f, CM=%8.2f" % (block.ssb, block.sst, block.sse, block.cm))
    print("MSB = %8.2f, MST = %8.2f, MSE = %8.2f" % (block.msb, block.mst, block.mse))
    print("F(T) = %8.2f, F(B) = %8.2f" % (block.Ft, block.Fb))
    print("P(T) = %f, P(B) = %f" % (block.Pt, block.Pb))
    
    print()
    print_intervals(block.turkey_intervals, "TURKEY-KRAMMER INTERVALS")

    print()
    print_intervals(block.bonferroni_intervals, "BONFERRONI INTERVALS")

if __name__ == "__main__":
    
    
    input_file = sys.argv[1]
    
    array = pandas.read_csv(input_file,  infer_datetime_format = True).values.astype(numpy.float64)

    print_report(array, CPUReductor)
    print_report(pycuda.gpuarray.to_gpu(array), GPUReductor)