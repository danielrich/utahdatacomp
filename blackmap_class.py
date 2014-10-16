from numpy import genfromtxt, savetxt
from sklearn import lda
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

def run_model( model, model_name, X, Y, X_val):

    # Add time column to training set.
    new_values = [ [x] for x in range(len(X))]
    X = numpy.append(X, new_values, 1)
    max_time_val = X[-1][-1] *2 - X[-2][-1]

    Y_black = make_black_maps_class(Y)
    # Load validation data
    model.fit(X, Y_black)

    # add time column to validation set.
    new_values = [ [max_time_val] for x in range(len(X_val))]
    X_val = numpy.append(X_val, new_values, 1)

    # Now predict validation output
    Y_pred = model.predict(X_val)

    savetxt('final_pred_y{0}.csv'.format(model_name), Y_pred, delimiter=',')

run_model(lda.LDA(), "LDA", X, Y, X_val)
