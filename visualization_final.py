from numpy import genfromtxt, savetxt
from sklearn import ensemble
from sklearn import linear_model
from sklearn.tree import DecisionTreeRegressor
import numpy
from pylab import *

# Load training data
X = genfromtxt('train_X_final.csv', delimiter=',')
Y = genfromtxt('train_Y_final.csv', delimiter=',')

X_val = genfromtxt('val_X_final.csv', delimiter=',')
#Y_val = genfromtxt('val_Y.csv', delimiter=',')

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

def gen_visuals(X, Y, X_val):
    plot(Y, range(Y))
    grid(True)
    savefig("y_plot.png")

gen_visuals(X, Y, X_val)


