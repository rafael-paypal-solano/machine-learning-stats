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

def sample_model_with_noise(fn):
    length = SAMPLE_LENGTH
    sequence = numpy.array(tuple(map(lambda y: y + fn(y+1), numpy.abs(numpy.random.normal(0,1,length) * numpy.random.beta(2., 1., length) * 10))))
    model = numeric.Taylor.autoregressive_model(sequence)
    return model


if __name__ == "__main__":
    models = tuple(map(sample_model, BASE_MODELS))
    index = 1

    for model in models:

        try:
            print(model.fit().rsquared)            
        except:            
            print("Model #%d failed" % index)

        index += 1