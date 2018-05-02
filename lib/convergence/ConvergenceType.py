from enum import Enum    
import pathos.multiprocessing as multiprocessing
from .ConvergenceState import ConvergenceState
import functools

#
# Credits:
#   Python Iterator Tutorial (https://www.datacamp.com/community/tutorials/python-iterator-tutorial)
#     

def is_almost_surely_reduce(x, L, S, alpha):    
    return functools.reduce(lambda state, s: state(x(s)-L(s)), S, ConvergenceState(alpha))

def is_almost_surely_map(X, L, S, alpha, pool):
    return pool.map(lambda x: is_almost_surely_reduce(x, L, S, alpha), X)

class ConvergenceType(Enum):
    """
        Contains static functions to find out convergence nature of random sequences.
    """
    ALMOST_SURE = 0
    MEAN = 1
    PROBABILITY = 2
    DISTRIBUTION = 3

    @classmethod
    def is_almost_surely(clazz, X, L, S, alpha = 5e-6, pool = None):
        """
            Args:
                X (iterable of functions): A sequence of random variables.
                L (function): Limit function / random variable.
                S (iterable of numbers): The sample space.        
                alpha (optional float): Positive real number in the (0,1) range which indicates the tolerance level used to check convergence
                pool (optional pathos.multiprocessing.Pool): - 

            Rerturns:
                (bool): true if the random sequence X almost surely converge to limit function L
        """
        processing_pool = pool if not (pool is None) else multiprocessing.Pool(multiprocessing.cpu_count())    
        states = is_almost_surely_map(X, L, S, alpha, processing_pool)        
        in_range_count = functools.reduce(lambda s, state: s + (1 if state.in_range > 0 else 0) , states, 0)
        return in_range_count > 0
        
