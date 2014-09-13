from numpy import genfromtxt
from sklearn import linear_model
import numpy

# Load training data
X = genfromtxt('train_X.csv', delimiter=',')
Y = genfromtxt('train_Y.csv', delimiter=',')

def run_model( model, model_name, X, Y):
    # Load validation data
    model.fit(X, Y)

    X_val = genfromtxt('val_X.csv', delimiter=',')
    Y_val = genfromtxt('val_Y.csv', delimiter=',')
    new_values = [ [x] for x in range(len(X_val))]
    X_val = numpy.append(X_val, new_values, 1)
    # Now predict validation output
    Y_pred = model.predict(X_val)

    # Crop impossible values
    Y_pred[Y_pred < 0] = 0
    Y_pred[Y_pred > 600] = 600

    # Calculate score
    err_Y=Y_pred-Y_val
    score = (sum(abs(err_Y[err_Y<0]))*10+sum(abs(err_Y[err_Y>=0])))/len(X)

    print score, model_name

# Create linear regression object
#regr = linear_model.LinearRegression() 648.56
new_values = [ [x] for x in range(len(X))]
X = numpy.append(X, new_values, 1)
run_model(linear_model.Ridge(alpha=0.5), "linear ridge time series, .5", X, Y)
run_model(linear_model.RidgeCV(), "linear ridgeCV time series, ", X, Y)
