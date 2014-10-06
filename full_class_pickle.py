from numpy import genfromtxt, savetxt
from sklearn import linear_model
from sklearn import lda
import numpy
from numpy import log
import sys
from sklearn.externals import joblib
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

def get_highest_prob(pred_proba, y_pred, prob_index, count):

    # annotate with the original ordering before doing any sorting. Create a list of tuples (order_num, list_of_probabilities)

    result_sort = sorted(pred_proba, key=lambda X: X[prob_index]) # Dang I can't sort it I lose the ordering that way.
    points = []
    for prob in result_sort:
        if y

    import pdb;pdb.set_trace()


def run_model(model_name, X, Y, X_val):

    new_values = [ [x] for x in range(len(X))]
    X = numpy.append(X, new_values, 1)
    max_time_val = X[-1][-1] *2 - X[-2][-1]


    Y = make_classes(Y)
    # Load validation data
    model = joblib.load('onevsrestpickler.pkl')

    new_values = [ [max_time_val] for x in range(len(X_val))]
    X_val = numpy.append(X_val, new_values, 1)

    # Now predict validation output
    Y_pred_proba = model.predict_proba(X_val)
    Y_pred = [ 0 for x in X_val ]
    classes = model.classes_ #sorted(set(Y))
    classes.reverse() # becaue of the penalty we want to start high first.
    for class_index in range(len(classes)):
        count = 0
        for y in Y:
            if y == classes[class_index]:
                count +=1

        prob_index = (class_index - (len(classes) - 1)) * -1
        predict_points = get_highest_prob(Y_pred_proba, Y_pred, prob_index, count)
        # it returns a list of indexes to replace
        for  point in predict_points:
            Y_pred[point] = classes(class_index)



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
run_model( "LDA_onevsrest_class", X, Y, X_val)
