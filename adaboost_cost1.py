from numpy import genfromtxt
from sklearn import ensemble
from sklearn import linear_model
from sklearn.tree import DecisionTreeRegressor
import numpy

# Load training data
X = genfromtxt('train_X.csv', delimiter=',')
Y = genfromtxt('train_Y.csv', delimiter=',')

X_val = genfromtxt('val_X.csv', delimiter=',')
Y_val = genfromtxt('val_Y.csv', delimiter=',')

def run_basic_time( model, model_name, X, Y, X_val, Y_val):
    new_values = [ [x] for x in range(len(X))]
    X = numpy.append(X, new_values, 1)
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler().fit(X)
    X = scaler.transform(X)
    max_time_val = X[-1][-1] *2 - X[-2][-1]

    # Load validation data
    model.fit(X, Y)

    new_values = [ [max_time_val] for x in range(len(X_val))]
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

def run_model( model, model_name, X, Y, X_val, Y_val):

    run_basic_time(model, model_name, X, Y, X_val, Y_val)

run_model(ensemble.AdaBoostRegressor(loss="square"), "StandardScaler applied square Ada Boost regressor modified sklearn", X, Y, X_val, Y_val)


