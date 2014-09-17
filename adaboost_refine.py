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

def run_model( model, model_name, X, Y, X_val, Y_val):

    run_basic_time(model, model_name, X, Y, X_val, Y_val)

# lets try and just use the last 1000 ones.
# Create linear regression object
#regr = linear_model.LinearRegression() 648.56
#from sklearn.pipeline import Pipeline
#clf = Pipeline([
#    ('feature select', linear_model.RandomizedLasso()),
#    ('fit', ensemble.AdaBoostRegressor())])

run_model(ensemble.AdaBoostRegressor(loss="square", n_estimators=1000), "square 1000 estimator Ada Boost regressor ", X, Y, X_val, Y_val)
run_model(ensemble.AdaBoostRegressor(loss="exponential", n_estimators=1000), "exponential 1000 estimator Ada Boost regressor ", X, Y, X_val, Y_val)


