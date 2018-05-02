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
<<<<<<< HEAD
    MEAN = 1
=======
    MEAN=1 
>>>>>>> refs/remotes/origin/master
    PROBABILITY = 2
    DISTRIBUTION = 3

    @classmethod
<<<<<<< HEAD
    def is_almost_surely(clazz, X, L, S, alpha = 5e-6, pool = None)
=======
    def is_almost_surely(clazz, X, L, S, P, alpha = 5e-6, pool = None):
>>>>>>> refs/remotes/origin/master
        """
            Args:
                X (iterable of functions): A sequence of random variables.
                L (function): Limit function / random variable.
                S (iterable of numbers): The sample space.        
                alpha (optional float): Positive real number in the (0,1) range which indicates the tolerance level used to check convergence
                pool (optional pathos.multiprocessing.Pool): - 

            Rerturns:
                (bool): If The sequence 
        """
        processing_pool = pool if not (pool is None) else multiprocessing.Pool(multiprocessing.cpu_count())    
        states = is_almost_surely_map(X, L, s, alpha, processing_pool)
<<<<<<< HEAD
        in_range_count = reduce(lambda count, state: count + 1 if state.in_range else 0 , states, 0)
        return in_range_count == len(states)
=======
        in_range = sum(processing_pool.map(lambda state: 1 if state.in_range else 0, states))
        return in_range
>>>>>>> refs/remotes/origin/master
