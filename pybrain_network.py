from numpy import genfromtxt, savetxt
from sklearn import ensemble
from sklearn import neighbors
from sklearn import linear_model
from sklearn import tree
from sklearn import lda
from sklearn import qda
from sklearn import naive_bayes
from sklearn import svm
import numpy
from numpy import log
import sys
# Load training data
X = genfromtxt('train_X_final.csv', delimiter=',')
Y = genfromtxt('train_Y_final.csv', delimiter=',')

X_val = genfromtxt('val_X_final.csv', delimiter=',')


def make_black_maps_class(Y):
    for i in range(len(Y)):
        if Y[i] > 550:
            Y[i] = 600
        if Y[i] >= 500:
            Y[i] = 525
        if Y[i] >= 450:
            Y[i] = 475
        if Y[i] >= 400:
            Y[i] = 425
        if Y[i] >= 350:
            Y[i] = 375
        if Y[i] >= 300:
            Y[i] = 325
        if Y[i] >= 250:
            Y[i] = 275
        if Y[i] >= 200:
            Y[i] = 225
        if Y[i] >= 150:
            Y[i] = 175
        if Y[i] >= 100:
            Y[i] = 125
        else:
            Y[i] = 85
    return Y


def run_model( model, model_name, X, Y, X_val):

    new_values = [ [x] for x in range(len(X))]
    X = numpy.append(X, new_values, 1)
    max_time_val = X[-1][-1] *2 - X[-2][-1]

    Y = make_black_maps_class(Y)
    # Load validation data
    model.fit(X, Y)

    new_values = [ [max_time_val] for x in range(len(X_val))]
    X_val = numpy.append(X_val, new_values, 1)

    # Now predict validation output
    Y_pred = model.predict(X_val)

    # Crop impossible values
    Y_pred[Y_pred < 0] = 0
    Y_pred[Y_pred > 600] = 600

    savetxt('final_pred_y{0}.csv'.format(model_name), Y_pred, delimiter=',')

    black_map_count = 0
    for y in Y_pred:
        if y == 600:
            black_map_count += 1

    print black_map_count, model_name
    sys.stdout.flush()


#run_model(ensemble.AdaBoostClassifier(), "Adaboost class ", X, Y, X_val)
#run_model(neighbors.KNeighborsClassifier(), "kNN ", X, Y, X_val)
#run_model(tree.DecisionTreeClassifier(), "decicion tree ", X, Y, X_val)
#run_model(ensemble.RandomForestClassifier(), "random forest class ", X, Y, X_val)
run_model(lda.LDA(), "LDA_lot_class", X, Y, X_val)
#run_model(linear_model.LogisticRegression(), "LogisticRegression ", X, Y, X_val)
#run_model(naive_bayes.GaussianNB(), "gaussianNB ", X, Y, X_val)
#run_model(svm.SVC(), "svc ", X, Y, X_val)