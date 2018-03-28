import numpy
import scipy.stats as stats
import pathos.multiprocessing as multiprocessing

MODELS = (
    lambda x: 1
    lambda x: x,    
    lambda x: numpy.square(x),
    lambda x: numpy.exp(x),
    lambda x: numpy.log(x),
    lambda x: numpy.cos(x),
    lambda x: numpy.sin(x)
)

MODELS_RANGE = range(0, len(models))

def model_table(x, pool = None):
    if pool is None:
        processing_pool = multiprocessing.Pool(multiprocessing.cpu_count())
    else:
        processing_pool = pool

    y = tuple(processing_pool.map(lambda model: model(x), MODELS))