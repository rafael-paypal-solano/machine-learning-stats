import numpy
import scipy.stats as stats
import pathos.multiprocessing as multiprocessing

MIN_TAYLOR_TIME_SEQ_ORDER = 4
MAX_TAYLOR_TIME_SEQ_ORDER = 8

def taylor_time_seq(y, order = MIN_TAYLOR_TIME_SEQ_ORDER, pool = None):
    """
        Args:
            y (sequence of numbers): numpy.ndarray, list, set, tuple
            order (int): MIN_TAYLOR_TIME_SEQ_ORDER <= order <=  MIN_TAYLOR_TIME_SEQ_ORDER
            
        Returns:

    """