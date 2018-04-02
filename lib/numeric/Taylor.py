import numpy
from scipy import linalg
import statsmodels.api as api
import native

MIN_TAYLOR_TIME_SEQ_ORDER = 1
DEFAULT_TAYLOR_TIME_SEQ_ORDER = 6
MAX_TAYLOR_TIME_SEQ_ORDER = 8

def create_diff_matrix(y, order):
    return native.create_diff_matrix(y.astype(float), order)

def fill_regressors_matrix(y, order, diff_matrix, X):
    diff = numpy.copy(diff_matrix)
    vertical = range(0, X.shape[0])
    horizontal = range(0, X.shape[1])
    m = order + 1
    Q = range(0, order)
    R = range(1, order + 1)

    for I in vertical:

        #
        #  Fills a Row.
        #        
        for K in horizontal:
            X[I,K] = diff[K, order]

        #
        # Updates difference matrix.    
        #        
        for i in Q:

            for k in range( i , order):
                diff[i, k] = diff[i, k + 1]        

        diff[0, order] = y[m + I]    

        for i in R:
            diff[i, order] = diff[i-1, order] - diff[i-1, order-1] 

def create_regression_model(y, order, diff_matrix):
    l = len(y)
    columns = order + 1
    rows = l - columns
    X = numpy.zeros((rows, columns))
    Y = y[-rows:]
    fill_regressors_matrix(y, order, diff_matrix, X)
    return (X,Y)
    

class Taylor:

    @classmethod
    def autoregressive_model(clazz, y, order = DEFAULT_TAYLOR_TIME_SEQ_ORDER, pool = None):
        """
            Args:
                 y (sequence of numbers -- numpy.ndarray, list, set, tuple, etc -- ): This is an non-autoregressive time series.
                 order (int): MIN_TAYLOR_TIME_SEQ_ORDER <= order <=  MIN_TAYLOR_TIME_SEQ_ORDER

            Returns:
                 statsmodels.api.OLS. The model's coefficients can be used to create an autoregresive time series.
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
                    order_squared = order+1
                )
            )

        diff_array = create_diff_matrix (y, order)
        X, Y = create_regression_model(y, order, diff_array)    

        #
        #  TODO: Fails with non linear models.
        #
        return api.OLS(Y, X)


    #
    #   Test for Significance of Regression (F-statistic, Prob (F-statistic)    )
    #   H0: X2 == X2 == X3 == ... == Xn == 0
    #   H1: Xj != 0 for at least one j
    #

    #
    #   Test for Significance of Slope j (t, P>|t|)
    #   H0: Xj == 0 
    #   H1: Xj != 0 
    #
