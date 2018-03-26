import numpy
import sys
import pandas
from sklearn.linear_model import LinearRegression

if __name__ == "__main__":
    sales_file = sys.argv[1]
    sales_data = pandas.read_csv(sales_file, parse_dates=[0], infer_datetime_format = True)
    X = sales_data.drop('Q (Sales)', axis = 1)
    Y = sales_data['Q (Sales)']
    X_train, X_test, Y_train, Y_test = sklearn.cross_validation.train_test_split(X, Y, test_size = 0.33, random_state = 5)