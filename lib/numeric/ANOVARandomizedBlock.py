from .Reductor import Reductor
import scipy.stats as stats
from itertools import combinations
from statsmodels.stats.libqsturng import qsturng
from .ANOVATreatmentInterval import ANOVATreatmentInterval

# TODO: Finish create_bonferroni_intervals following the specs in the fouth link in the list below this line.
import numpy
#
#   Credits
#       https://academic.macewan.ca/burok/Stat252/notes/RBD.pdf (In depth theretical explanations)
#       http://faculty.fiu.edu/~howellip/anova23_bon.pdf (Summarizes all formulas)
#       https://ssor.vcu.edu/media/statistics/resources/spss/SPSS.Blocks.PC.pdf (Example for turkey-krammer intervals)
#       https://www.spcforexcel.com/knowledge/comparing-processes/bonferronis-method (Example for bonferrony intervals)
#       http://www2.hawaii.edu/~taylor/z631/multcomp.pdf

# Best available when  confidence intervals are needed  or sample sizes are not equal
# Which treatment means are significantly different from each other?"
# 
def create_turkey_intervals(row_mean, row_var, mse, alpha, k, b, df, pairs):  # Which treatment means are significantly different from each other ? (small sample)
    Q = qsturng(1-alpha, k, df)
    L = Q * numpy.sqrt(mse/b)
        
    return tuple(map(lambda pair: ANOVATreatmentInterval(row_mean[pair[0]] - row_mean[pair[1]],  L, pair), pairs))

# Extremely simple but not powerful
# 
#
def create_bonferroni_intervals(row_mean, row_var, mse, alpha, k, b, df, pairs): # Which treatment means are significantly different from each other ? (large sample)
    c = k * (k-1) / 2
    T = stats.t.isf(alpha/(2*k), df)
    L = T * numpy.sqrt((2 * mse)/b)
    
    return tuple(map(lambda pair: ANOVATreatmentInterval(row_mean[pair[0]] - row_mean[pair[1]],  L, pair), pairs))

class ANOVARandomizedBlock:
    def __init__(self, sample, alpha=0.05):
        array = Reductor.to_array(sample)
        self.__k__ = sample.shape[0]
        self.__b__ = sample.shape[1]        
        self.__n__ = self.b * self.k
        self.__cm__ = Reductor.correction_for_mean_block(array)       
        self.__ssb__ = Reductor.sum_of_squares_block(array, self.cm)    # Blocks/Subjects
        self.__sst__ = Reductor.sum_of_squares_total_block(array, self.cm) #  Treatments
        self.__sse__ = Reductor.standard_squared_error_block(array, self.cm)
        self.__df__ = self.n - self.k - self.b + 1

        self.__mst__ = self.sst / (self.k-1)
        self.__msb__ = self.ssb / (self.b-1)
        self.__mse__ = self.sse / self.df
        self.__Ft__ = self.mst/self.mse #  F-Ratio used to test H0 for treatments
        self.__Fb__ = self.msb/self.mse #  F-Ratio used to test H0 for subjects
        self.__Pt__ = stats.f.sf(self.Ft, self.k-1, self.df)
        self.__Pb__ = stats.f.sf(self.Fb, self.b-1, self.df)

        row_mean = numpy.mean(array, axis=1)        
        row_var = numpy.var(array, axis=1)
        row_pairs = sorted(combinations(range(0, len(row_mean)), 2), key = lambda p: p[0])        
        self.__turkey_intervals__ = create_turkey_intervals(row_mean, row_var, self.mse, alpha, self.k, self.b, self.df, row_pairs)
        self.__bonferroni_intervals__ = create_bonferroni_intervals(row_mean, row_var, self.mse, alpha, self.k, self.b, self.df, row_pairs)
        

    @property
    def turkey_intervals(self):
        return self.__turkey_intervals__

    @property
    def bonferroni_intervals(self):
        return self.__bonferroni_intervals__

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

    