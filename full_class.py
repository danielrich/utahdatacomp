from numpy import genfromtxt, savetxt
from sklearn import linear_model
from sklearn import lda
import numpy
from numpy import log
import sys
# Load training data
X = genfromtxt('train_X_final.csv', delimiter=',')
Y = genfromtxt('train_Y_final.csv', delimiter=',')

X_val = genfromtxt('val_X_final.csv', delimiter=',')

# make 600 classes Eek!
def make_classes(Y):
    import math
    for i in range(len(Y)):
        Y[i] = math.floor(Y[i])
    return Y

def get_highest_prob(pred_proba, y_pred, value):
    import pdb;pdb.set_trace()


def run_model( model, model_name, X, Y, X_val):

    new_values = [ [x] for x in range(len(X))]
    X = numpy.append(X, new_values, 1)
    max_time_val = X[-1][-1] *2 - X[-2][-1]


    Y = make_classes(Y)
    # Load validation data
    model.fit(X, Y)

    new_values = [ [max_time_val] for x in range(len(X_val))]
    X_val = numpy.append(X_val, new_values, 1)

    # Now predict validation output
    Y_pred_proba = model.predict_proba(X_val)
    Y_pred = [ 0 for x in X_val ]
    classes = range(1, 601)
    classes.reverse() # becaue of the penalty we want to start high first.
    for i in classes:
        count = 0
        for y in Y:
            if y == i:
                count +=1

        for j in range(count):
            from sklearn.externals import joblib
            joblib.dump(model, "onevsrestpickler.pkl")
            index = get_highest_prob(Y_pred_proba, Y_pred, i)
            Y_pred[index] = i



    # I want to make sure that I have the same number of each class as seen before.
    # so out of the 10,000 points I scan through and find the top x most likely members of a
    # given class. Start with the black maps and work down.

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

from sklearn.multiclass import OneVsRestClassifier
run_model(OneVsRestClassifier(lda.LDA()), "LDA_onevsrest_class", X, Y, X_val)
