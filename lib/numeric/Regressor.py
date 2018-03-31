import numpy
import scipy.stats as stats
import pathos.multiprocessing as multiprocessing

MIN_TAYLOR_TIME_SEQ_ORDER = 1
MAX_TAYLOR_TIME_SEQ_ORDER = 4

class Regressor:
    def taylor_series(y, order, order = MIN_TAYLOR_TIME_SEQ_ORDER, pool = None):
        """
            Args:
                 y (sequence of numbers -- numpy.ndarray, list, set, tuple, etc -- ): This is an no-autoregressive time series.
                 order (int): MIN_TAYLOR_TIME_SEQ_ORDER <= order <=  MIN_TAYLOR_TIME_SEQ_ORDER

            Returns:
                A Taylor time series.
        """

# class Regressor:    
#     def __init__(y, order = MIN_TAYLOR_TIME_SEQ_ORDER, pool = None):
#         """
#             Args:
#                 y (sequence of numbers -- numpy.ndarray, list, set, tuple, etc -- ): This is an no-autoregressive time series.
#                 order (int): MIN_TAYLOR_TIME_SEQ_ORDER <= order <=  MIN_TAYLOR_TIME_SEQ_ORDER
                
#         """
#         if order < MIN_TAYLOR_TIME_SEQ_ORDER or order > MIN_TAYLOR_TIME_SEQ_ORDER:
#             raise ValueError(
#                 "order parameter ({order}) is out of [{MIN_TAYLOR_TIME_SEQ_ORDER}, {MAX_TAYLOR_TIME_SEQ_ORDER}] interval".format(
#                     order = order,
#                     MIN_TAYLOR_TIME_SEQ_ORDER = MIN_TAYLOR_TIME_SEQ_ORDER,
#                     MAX_TAYLOR_TIME_SEQ_ORDER = MAX_TAYLOR_TIME_SEQ_ORDER
#                 )
#             )

#         if len(y) < order:
#             raise ValueError(
#                 "length of y ({length}) < (order+1)^2 ({order_squared})".format(
#                     length = len(y),
#                     order_squared = numpy.square(order+1)
#                 )
#             )

#         self.__order__ = numpy.square(order)
#         self.__diff_array__ = numpy.array((order+1, order+1))

#         array = self.__diff_array__ # Short hand
#         r = range(0, len(self))

#     def order(self):
#         return self.__len__()

#     def __len__(self):
#         return self.__order__

#     def __getitem__(self, sliced):
#         return self.__diff_array__[sliced]

