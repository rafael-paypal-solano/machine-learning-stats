from .Reductor import Reductor
import scipy.stats as stats

class ANOVARandomizedBlock:
    def __init__(self,sample):
        array = Reductor.to_array(sample)
        self.__k__ = sample.shape[0]
        self.__b__ = sample.shape[1]        
        self.__n__ = self.b * self.k
        self.__cm__ = Reductor.correction_for_mean_block(array)       
        self.__ssb__ = Reductor.sum_of_squares_block(array, self.cm)
        self.__sst__ = Reductor.sum_of_squares_total_block(array, self.cm)
        self.__sse__ = Reductor.standard_squared_error_block(array, self.cm)
        self.__df__ = self.n - self.k - self.b + 1

        self.__mst__ = self.sst / (self.k-1)
        self.__msb__ = self.ssb / (self.b-1)
        self.__mse__ = self.sse / self.df
        self.__Ft__ = self.mst/self.mse
        self.__Fb__ = self.msb/self.mse
        self.__Pt__ = stats.f.sf(self.Ft, self.k-1, self.df)
        self.__Pb__ = stats.f.sf(self.Fb, self.b-1, self.df)


    @property
    def mst(self):
        return self.__mst__

    @property
    def msb(self):
        return self.__msb__

    @property
    def mse(self):
        return self.__mse__

    @property
    def df(self):
        return self.__df__

    @property
    def b(self):
        return self.__b__

    @property
    def k(self):
        return self.__k__

    @property
    def n(self):
        return self.__n__

    @property
    def cm(self):
        return self.__cm__

    @property
    def ssb(self):
        return self.__ssb__

    @property
    def sst(self):
        return self.__sst__

    @property
    def sse(self):
        return self.__sse__

    @property
    def Ft(self):
        return self.__Ft__

    @property
    def Fb(self):
        return self.__Fb__

    @property
    def Pt(self):
        return self.__Pt__

    @property
    def Pb(self):
        return self.__Pb__        

    