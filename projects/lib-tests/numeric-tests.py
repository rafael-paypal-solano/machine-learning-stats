import numeric
import numpy

BASE_MODELS = [
    lambda x, A, B, C, D, E, F: x,
    lambda x, A, B, C, D, E, F: numpy.square(x),
    lambda x, A, B, C, D, E, F: numpy.exp(x),
    lambda x, A, B, C, D, E, F: numpy.log(x),
    lambda x, A, B, C, D, E, F: numpy.cos(x),
    lambda x, A, B, C, D, E, F: numpy.cos(x) + cos.sin(x),
    lambda x, A, B, C, D, E, F: (A * numpy.square(x) + B * x + C) * numpy.exp(D * x) * (numpy.cos(E * x) + cos(F * y))
]

if __name__ == "__main__":
    sequence = numpy.array(tuple(map(lambda x: numpy.log(2*x+2), range(0, 64))))
    numeric.TaylorCoefficients(sequence)