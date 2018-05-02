import convergence
import scipy.special
import scipy.stats
import numpy

class GammaSequence(object):
    def __init__(self, K, ϴ, N=10):
        self.__K__ =  K # shape
        self.__K2__ = K * K
        self.__ϴ__ = ϴ # scale
        self.__N__ = N
        self.__n__ = 1

    def __iter__(self):
        return self

    def __next__(self):
        x = 1
        if self.__n__ > self.__N__:
            raise StopIteration
        else:
            k = (self.__K2__ * (self.__N__ / self.__n__)) / self.__K__ 
            ϴ = self.__ϴ__
            self.__n__ += 1
            return lambda x: scipy.stats.gamma.pdf(x, k, scale=ϴ)