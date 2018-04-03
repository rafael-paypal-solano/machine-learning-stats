cimport numpy
import numpy

DOUBLE_TYPE = numpy.double
ctypedef numpy.double_t DOUBLE_TYPE_T

def create_diff_matrix(numpy.ndarray y, int order):
    assert y.dtype == DOUBLE_TYPE
    cdef numpy.ndarray diff = numpy.zeros([order+1, order+1], dtype=DOUBLE_TYPE)

    for i in range(0, order+1):
        diff[0, i] = y[i]

    for i in range(1, order + 1):
        for k in range(i, order + 1):
            diff[i, k] = diff[i-1, k] - diff[i-1, k-1] 
    
    return diff

def update_diff_matrix(numpy.ndarray diff, int order, double y):
    assert diff.dtype == DOUBLE_TYPE
    for i in range(0, order):

        for k in range(i , order):
            diff[i, k] = diff[i, k + 1]        

    diff[0, order] = y

    for i in  range(1, order + 1):
        diff[i, order] = diff[i-1, order] - diff[i-1, order-1]     