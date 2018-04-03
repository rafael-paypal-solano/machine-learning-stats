import numeric
import numpy
import pathos.multiprocessing as multiprocessing
import numpy.linalg

BASE_MODELS = [
    lambda x: x,
    lambda x: numpy.square(x),
    lambda x: numpy.exp(x),
    lambda x: numpy.log(x),
    lambda x: numpy.cos(x),
    lambda x: numpy.cos(numpy.radians(x)) + numpy.sin(numpy.radians(x)),
    lambda x: (numpy.cos(numpy.radians(x) + numpy.sin(numpy.radians(x)))) * numpy.exp(x),
]

SAMPLE_LENGTH = 100

def sample_model(fn):
    length = SAMPLE_LENGTH
    sequence = numpy.array(tuple(map(lambda x: fn(x+1), range(0, length))))    
    model = numeric.Taylor.autoregressive_model(sequence)
    return model

def sample_model_with_noise(fn): # Use Taylor
    length = SAMPLE_LENGTH
    sequence = numpy.array(tuple(map(lambda y: numpy.random.normal(0,1) * 10 + fn(y+1), numpy.abs(numpy.random.normal(0,1,length) * numpy.random.beta(2., 1., length) * 10))))
    model = numeric.Taylor.autoregressive_model(sequence)
    return model

def test_model(asserter, model):
    models = tuple(map(model, BASE_MODELS))

    for model in models:
        print(model.fit().rsquared)
        if  not asserter(model.fit().rsquared):
            raise ValueError("Assertion failure")

def test_randomized_models():
    test_model(lambda a: numpy.abs(a) > 0.80, sample_model)
    test_model(lambda a: numpy.abs(a) < 0.40, sample_model_with_noise)


if __name__ == "__main__":
    test_randomized_models()
    