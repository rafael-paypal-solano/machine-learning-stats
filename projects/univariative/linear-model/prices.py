import numpy
import sys
import pandas
from sklearn.linear_model import LinearRegression
from sklearn import cross_validation

if __name__ == "__main__":
    sales_file = sys.argv[1]
    sales_data = pandas.read_csv(sales_file, parse_dates=[0], infer_datetime_format = True)
    X = sales_data.drop('Sales', axis = 1)    
    Y = numpy.array(tuple(map( lambda x: float(x), sales_data['Sales'])))

    X_train, X_test, Y_train, Y_test = cross_validation.train_test_split(X, Y, test_size = 0.33, random_state = 5)

    linear_regression = LinearRegression()
    linear_regression.fit(X_train, Y_train)
    pred_train = linear_regression.predict(X_train)
    pred_test = linear_regression.predict(X_test)

    rmse_train = numpy.sqrt(numpy.mean((Y_train - linear_regression.predict(X_train)) ** 2))
    rmse_test = numpy.sqrt(numpy.mean((Y_test - linear_regression.predict(X_test)) ** 2))
    rmse_ratio = numpy.abs(rmse_train - rmse_test) / numpy.max((rmse_train, rmse_test))

    print("Fit a model X_train, and calculate RMSE with Y_train: %8.2f" % rmse_train)
    print("Fit a model X_train, and calculate RMSE with X_test, Y_test: %8.2f" % rmse_test)
    print("RMSE = %5.2f" % rmse_ratio )
