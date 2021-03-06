from numpy import genfromtxt, savetxt
from sklearn import ensemble
from sklearn import linear_model
from sklearn.tree import DecisionTreeRegressor
import numpy

# Load training data
X = genfromtxt('train_X_final.csv', delimiter=',')
Y = genfromtxt('train_Y_final.csv', delimiter=',')

X_val = genfromtxt('val_X_final.csv', delimiter=',')


def make_black_maps_class(Y):
    for i in range(len(Y)):
        if Y[i] == 600:
            pass
        else:
            Y[i] = 180
    return Y

model = ensemble.AdaBoostClassifier()


new_values = [ [x] for x in range(len(X))]
X = numpy.append(X, new_values, 1)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler().fit(X)
X = scaler.transform(X)
max_time_val = X[-1][-1] *2 - X[-2][-1]

Y = make_black_maps_class(Y)
# Load validation data
import pdb;pdb.set_trace()
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

#from sklearn import grid_search

#parameters = {"n_estimators":(50,100,200,300,400,500), "base_estimator__max_depth":(2,3,4,5,6,7,8,9,10)}
#model = ensemble.AdaBoostRegressor(DecisionTreeRegressor())
#grid_model = grid_search.GridSearchCV(model, parameters, n_jobs=4)
#run_model(grid_model, " decision tree regressor depth gridsearch Ada Boost regressor modified sklearn ", X, Y, X_val, Y_val)

#print grid_model.get_params()

