import convergence
import scipy.special
import scipy.stats
import numpy

def test_gamma_sequence():
    ϴ = 1.5
    s = 2
    K = 1
    L = lambda s : scipy.stats.expon.pdf(s, scale = ϴ )
    N = 10
    S = numpy.random.normal(10,2,N*N)
    accept = convergence.ConvergenceType.is_almost_surely(convergence.GammaSequence(K, ϴ, N), L, S)
    reject = convergence.ConvergenceType.is_almost_surely(convergence.GammaSequence(2*K, ϴ, N), L, S)
    print("accept=%s, reject=%s" %(accept, reject))

def test_almost_surely():
    test_gamma_sequence()

if __name__ == "__main__":   
    test_almost_surely()