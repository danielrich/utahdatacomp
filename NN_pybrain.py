from numpy import genfromtxt, savetxt
import numpy
from numpy import log
import sys
# Load training data
X = genfromtxt('train_X_final.csv', delimiter=',')
Y = genfromtxt('train_Y_final.csv', delimiter=',')

X_val = genfromtxt('val_X_final.csv', delimiter=',')

new_values = [ [x] for x in range(len(X))]
X = numpy.append(X, new_values, 1)
max_time_val = X[-1][-1] *2 - X[-2][-1]

new_values = [ [max_time_val] for x in range(len(X_val))]
X_val = numpy.append(X_val, new_values, 1)


import pybrain
from pybrain.tools.shortcuts import buildNetwork
net = buildNetwork(2001, 4000, 601)

from pybrain.datasets import SupervisedDataSet
ds = SupervisedDataSet(2001, 601)

for i in range(len(X)):
    ds.addSample(X[i], pybrain.utilities.one_to_n(Y[i], 601))

from pybrain.supervised.trainers import BackpropTrainer
trainer = BackpropTrainer(net, ds)

for i in range(20):
    trainer.train()#trains for 1 epoch
    Y_pred = []
    for x in X_val:
        out_y = net.activate(x)
        max_value = -100
        index = 180
        for y_i in range(len(out_y)):
            if max_value == -100:
                max_value = out_y[y_i]
            if out_y[y_i] >= max_value:
                max_value = out_y[y_i]
                index = y_i
        Y_pred.append(index)

    savetxt('final_pred_yNN_epoch{0}.csv'.format(i), Y_pred, delimiter=',')

    black_map_count = 0
    for y in Y_pred:
        if y == 600:
            black_map_count += 1

    print black_map_count, "pybrain epoch", i
    sys.stdout.flush()

