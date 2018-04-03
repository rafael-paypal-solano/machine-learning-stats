from distutils.core import setup
from Cython.Build import cythonize
import numpy
import os

path = os.path.dirname(os.path.realpath(__file__))
setup(
    name = "Native Booster",
    ext_modules = cythonize( path + '/matrices.pyx', ),  # accepts a glob pattern
    include_dirs=[numpy.get_include()]
)