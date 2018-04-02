cimport numpy
import numpy

DOUBLE_TYPE = numpy.double
ctypedef numpy.double_t DOUBLE_TYPE_T

def create_diff_matrix(numpy.ndarray y, int order):
    assert y.dtype == DOUBLE_TYPE
    cdef numpy.ndarray array = numpy.zeros([order+1, order+1], dtype=DOUBLE_TYPE)

    for i in range(0, order+1):
        array[0, i] = y[i]

    for i in range(1, order + 1):
        for k in range(i, order + 1):
            array[i, k] = array[i-1, k] - array[i-1, k-1] 
    
    return array

def update_diff_matrix(numpy.ndarray diff, int order, double y):
    