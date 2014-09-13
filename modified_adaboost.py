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
#    new_values = [ [x] for x in range(len(X_val))]
#    X_val = numpy.append(X_val, new_values, 1)

    # Now predict validation output
    Y_pred = model.predict(X_val)

    # Crop impossible values
    Y_pred[Y_pred < 0] = 0
    Y_pred[Y_pred > 600] = 600

    # Calculate score
    err_Y=Y_pred-Y_val
    score = (sum(abs(err_Y[err_Y<0]))*10+sum(abs(err_Y[err_Y>=0])))/len(X)

    print score, model_name

def average_model( models, model_name, X, Y):
    # Load validation data
    for model in models:
	model.fit(X, Y)

    X_val = genfromtxt('val_X.csv', delimiter=',')
    Y_val = genfromtxt('val_Y.csv', delimiter=',')
#    new_values = [ [x] for x in range(len(X_val))]
#    X_val = numpy.append(X_val, new_values, 1)

    # Now predict validation output
    ytotal = 0
    for model in models:
	ytotal += model.predict(X_val)

    Y_pred =  ytotal/len(models)

    # Crop impossible values
    Y_pred[Y_pred < 0] = 0
    Y_pred[Y_pred > 600] = 600

    # Calculate score
    err_Y=Y_pred-Y_val
    score = (sum(abs(err_Y[err_Y<0]))*10+sum(abs(err_Y[err_Y>=0])))/len(X)

    print score, model_name


#new_values = [ [x] for x in range(len(X))]
#X = numpy.append(X, new_values, 1)


#run_model(ensemble.ExtraTreesRegressor(n_jobs=8), "ExtraTrees regressor n-jobs time series", X, Y)
#run_model(ensemble.ExtraTreesRegressor(n_estimators=8, n_jobs=8), " 8 estimator ExtraTrees regressor n-jobs time series", X, Y)
#run_model(ensemble.ExtraTreesRegressor(n_estimators=15, n_jobs=8), "15 estimator ExtraTrees regressor n-jobs time series", X, Y)
#average_model(
#[ensemble.ExtraTreesRegressor(n_estimators=40, n_jobs=8),
#ensemble.ExtraTreesRegressor(n_estimators=8, n_jobs=8),
##linear_model.Ridge(alpha=.5)],
# "40 estimator ExtraTrees regressor n-jobs time series averaged with 8 estimator and ridge.", X, Y)
run_model(ensemble.AdaBoostRegressor(), " Ada Boost regressor ", X, Y)
run_model(ensemble.AdaBoostRegressor(n_estimators=100), " estimators 100 Ada Boost regressor ", X, Y)
run_model(ensemble.AdaBoostRegressor(DecisionTreeRegressor(max_depth=4),n_estimators=100), " decision tree regressor depth 4 estimators 100 Ada Boost regressor ", X, Y)
run_model(ensemble.AdaBoostRegressor(n_estimators=300), " estimators 300 Ada Boost regressor ", X, Y)
