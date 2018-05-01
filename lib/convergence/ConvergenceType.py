from enum import Enum    
import pathos.multiprocessing as multiprocessing
from .ConvergenceState import ConvergenceState

#
# Credits:
#   Python Iterator Tutorial (https://www.datacamp.com/community/tutorials/python-iterator-tutorial)
#     

def is_almost_surely_reduce(x, L, S, alpha):
    return reduce(lambda state, s: state(x(s)-L(s)), S, ConvergenceState(alpha))

def is_almost_surely_map(X, L, S, alpha, pool):
    return pool.map(lambda x: is_almost_sure_reduce(x, L, S, alpha), X)

class ConvergenceType(Enum):
    """
        Contains static functions to find out convergence nature of random sequences.
    """
    ALMOST_SURE = 0
    MEAN=1,
    PROBABILITY,
    DISTRIBUTION

    @classmethod
    def is_almost_surely(clazz, X, L, S, P, alpha = 5e-6, pool = None)
        """
            Args:
                X (iterable of functions): A sequence of random variables.
                L (function): Limit function / random variable.
                S (iterable of numbers): The sample space.        
                P (function): A probability function whose domain is the sample space.
                alpha (optional float): Positive real number in the (0,1) range which indicates the tolerance level used to check convergence
                pool (optional pathos.multiprocessing.Pool): - 

            Rerturns:
                (bool): If The sequence 
        """
        processing_pool = pool if not (pool is None) else multiprocessing.Pool(multiprocessing.cpu_count())    
        states = is_almost_surely_map(X, L, s, alpha, processing_pool)
        in_range = sum(lambda state: state., )