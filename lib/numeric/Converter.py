import numpy

class Converter:
    @classmethod

    def to_numpy_array(clazz, y)
        """
            Converts the array-like argument into a numpy.ndarray

            Args:
                y (array-like or numpy.ndarray of floats): 
            Returns:
                numpy.ndarray.
        """
        if not (type(y) is numpy.ndarray):
            return numpy.array(y)
        return y
        