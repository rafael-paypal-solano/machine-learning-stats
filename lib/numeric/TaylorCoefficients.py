import numpy

MIN_TAYLOR_TIME_SEQ_ORDER = 1
MAX_TAYLOR_TIME_SEQ_ORDER = 4

def create_diff_array(y, order):
    array = numpy.zeros((order+1, order+1))
    r = range(0, order+1)

    for i in r:
        array[0, i] = y[i]

    Q = range(1, order + 1)
    for i in Q:
        R = range(i, order + 1)
        
        for k in R:
            array[i, k] = array[i-1, k] - array[i-1, k-1] 

    return array

class TaylorCoefficients:
    def __init__(self, y, order = MAX_TAYLOR_TIME_SEQ_ORDER, pool = None):
        """
            Args:
                 y (sequence of numbers -- numpy.ndarray, list, set, tuple, etc -- ): This is an no-autoregressive time series.
                 order (int): MIN_TAYLOR_TIME_SEQ_ORDER <= order <=  MIN_TAYLOR_TIME_SEQ_ORDER
        """

        
        if order < MIN_TAYLOR_TIME_SEQ_ORDER or order > MAX_TAYLOR_TIME_SEQ_ORDER:
            raise ValueError(
                "order parameter ({order}) is out of [{MIN_TAYLOR_TIME_SEQ_ORDER}, {MAX_TAYLOR_TIME_SEQ_ORDER}] interval".format(
                    order = order,
                    MIN_TAYLOR_TIME_SEQ_ORDER = MIN_TAYLOR_TIME_SEQ_ORDER,
                    MAX_TAYLOR_TIME_SEQ_ORDER = MAX_TAYLOR_TIME_SEQ_ORDER
                )
            )

        if len(y) < order:
            raise ValueError(
                "length of y ({length}) < (order+1) ({order_squared})".format(
                    length = len(y),
                    order_squared = numpy.square(order+1)
                )
            )

        self.__order__ = numpy.square(order)
        self.__diff_array__ = create_diff_array (y, order)
