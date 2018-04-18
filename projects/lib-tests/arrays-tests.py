from environment import Arrays
import pycuda
import numpy
from pycuda.gpuarray import GPUArray
import pycuda.gpuarray 
import pycuda
import skcuda.misc
import skcuda.linalg
from numeric import GPUReductor
import pandas
import sys

if __name__ == "__main__":
    input_file = sys.argv[1]    
    a = pandas.read_csv(input_file,  infer_datetime_format = True).values.astype(numpy.float64)
    
#    a = numpy.array(
#        (
#            (1,   3.0, 5),
#            (7.0, 11, 9),
#            (13, 17.0, 19),
#        )
#    )
    b = Arrays.to_array(a)

    print(type(a), type(b))

    print(Arrays.sum(a))
    print(Arrays.sum(b))

    c = Arrays.sum(a,0)
    d = Arrays.sum(b,0)
    f = Arrays.sum(a,1)
    g = Arrays.sum(b,1)

    print(c)
    print(d)        
    print(f)
    print(g)        
    
    print(GPUReductor.correction_for_mean_block(b))    
    print(GPUReductor.sum_of_squares_block(b))
    print(GPUReductor.sum_of_squares_total_block(b))
    print(GPUReductor.mean_block(b))
    print(GPUReductor.standard_squared_error_block(b))