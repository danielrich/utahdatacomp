from numpy import genfromtxt, savetxt
from sklearn import ensemble
from sklearn import linear_model
from sklearn.tree import DecisionTreeRegressor
import numpy

# Load training data
X = genfromtxt('train_X_final.csv', delimiter=',')
Y = genfromtxt('train_Y_final.csv', delimiter=',')

X_val = genfromtxt('val_X_final.csv', delimiter=',')
#Y_val = genfromtxt('val_Y.csv', delimiter=',')

# Ok so I think the best way to track the temporal changes is a set of 2000 particle filters that are each tracking
# the state of a given context. Each context has some % that it contibutes to the failed dies. For example
# context x1 may contributes a failure of 20% and x2 will contribute 10%. Together when both contexts are present in
# a given X then 32% of chips will fail.
# To start the initial weighting take the intercept for t=0 and /(number of active contexts) for an intial weight. # NOTE there are always 122 active contexts

def run_basic_time( model, model_name, X, Y, X_val):
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

    savetxt('final_pred_y.csv', Y_pred, delimiter=',')

    # Calculate score
    #err_Y=Y_pred-Y_val
    #score = (sum(abs(err_Y[err_Y<0]))*10+sum(abs(err_Y[err_Y>=0])))/len(X)

    #print score, model_name, "Basic time"

def run_model( model, model_name, X, Y, X_val):

    run_basic_time(model, model_name, X, Y, X_val)

run_model(ensemble.AdaBoostRegressor(loss="square"), "StandardScaler applied square Ada Boost regressor modified sklearn with .07x neg", X, Y, X_val)
