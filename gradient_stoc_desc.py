from numpy import genfromtxt
from sklearn import svm
import numpy

# Load training data
X = genfromtxt('train_X.csv', delimiter=',')
Y = genfromtxt('train_Y.csv', delimiter=',')

X_val = genfromtxt('val_X.csv', delimiter=',')
Y_val = genfromtxt('val_Y.csv', delimiter=',')

def run_basic_time( model, model_name, X, Y, X_val, Y_val):
    new_values = [ [x] for x in range(len(X))]
    X = numpy.append(X, new_values, 1)

    # Load validation data
    model.fit(X, Y)

    new_values = [ [10000] for x in range(len(X_val))]
    X_val = numpy.append(X_val, new_values, 1)

    # Now predict validation output
    Y_pred = model.predict(X_val)

    # Crop impossible values
    Y_pred[Y_pred < 0] = 0
    Y_pred[Y_pred > 600] = 600

    # Calculate score
    err_Y=Y_pred-Y_val
    score = (sum(abs(err_Y[err_Y<0]))*10+sum(abs(err_Y[err_Y>=0])))/len(X)

    print score, model_name, "Basic time"

def run_no_time( model, model_name, X, Y, X_val, Y_val):
    # Load validation data
    model.fit(X, Y)

    # Now predict validation output
    Y_pred = model.predict(X_val)

    # Crop impossible values
    Y_pred[Y_pred < 0] = 0
    Y_pred[Y_pred > 600] = 600

    # Calculate score
    err_Y=Y_pred-Y_val
    score = (sum(abs(err_Y[err_Y<0]))*10+sum(abs(err_Y[err_Y>=0])))/len(X)

    print score, model_name, "no time"


def run_model( model, model_name, X, Y, X_val, Y_val):

    run_basic_time(model, model_name, X, Y, X_val, Y_val)
    run_no_time(model, model_name, X, Y, X_val, Y_val)

# Create linear regression object
#regr = linear_model.LinearRegression() 648.56
run_model(svm.SVR(kernel="rbf"), "SVR rbf", X, Y, X_val, Y_val)
run_model(svm.NuSVR(kernel="rbf"), "NuSVR rbf", X, Y, X_val, Y_val)
run_model(svm.SVR(kernel="linear"), "SVR linear", X, Y, X_val, Y_val)
run_model(svm.NuSVR(kernel="linear"), "NuSVR linear", X, Y, X_val, Y_val)
run_model(svm.SVR(kernel="polynomial"), "SVR polynomial", X, Y, X_val, Y_val)
run_model(svm.NuSVR(kernel="polynomial"), "NuSVR polynomial", X, Y, X_val, Y_val)
run_model(svm.SVR(kernel="sigmoid"), "SVR sigmoid", X, Y, X_val, Y_val)
run_model(svm.NuSVR(kernel="sigmoid"), "NuSVR sigmoid", X, Y, X_val, Y_val)

#Still to test the polynomial stuff



