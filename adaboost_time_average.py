from numpy import genfromtxt
from sklearn import linear_model
from sklearn import ensemble
from sklearn.tree import DecisionTreeRegressor
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

def run_average_time_model( model, model_name, X, Y):
    # Load validation data
    ytotal = 0

    X_val = genfromtxt('val_X.csv', delimiter=',')
    Y_val = genfromtxt('val_Y.csv', delimiter=',')


    for i in range(1,500):
	    xsplit = numpy.array_split(X, i)[-1]
	    ysplit = numpy.array_split(Y, i)[-1]
	    model.fit(xsplit, ysplit)
	    # Now predict validation output
	    ytotal += model.predict(X_val)

    Y_pred =  ytotal/len(range(1,10))

    # Crop impossible values
    Y_pred[Y_pred < 0] = 0
    Y_pred[Y_pred > 600] = 600

    # Calculate score
    err_Y=Y_pred-Y_val
    score = (sum(abs(err_Y[err_Y<0]))*10+sum(abs(err_Y[err_Y>=0])))/len(X)

    print score, model_name

run_average_time_model(ensemble.AdaBoostRegressor(loss="exponential"), " average of 10 segmented exponential loss Ada Boost regressor ", X, Y)
