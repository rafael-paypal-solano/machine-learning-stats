import pathos.multiprocessing as multiprocessing
import numpy
from scipy import linalg
import statsmodels.api as api
import sys
import pandas

if __name__ == "__main__":
    health_file = sys.argv[1]
    summary_file = sys.argv[2]
    pool = multiprocessing.Pool(multiprocessing.cpu_count())    

    health_data = pandas.read_csv(health_file)
    X = health_data.drop('X5', axis = 1)
    print(type(X))
    Y = health_data['X5']

    model = api.OLS(Y, X).fit()
    print(model.summary())