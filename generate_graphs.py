import matplotlib
matplotlib.use('Agg')
from numpy import genfromtxt
import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

bad_predictions = genfromtxt('predictions/final_pred_y.csv', delimiter=',')
ok_predictions = genfromtxt('predictions/final_pred_yLDA_185.csv', delimiter=',')
best_predictions  = genfromtxt('predictions/final_pred_yprobalities.csv', delimiter=',')
original_y = genfromtxt('data/train_Y_final.csv', delimiter=',')

num_bins = 300
# the histogram of the data
plt.hist(bad_predictions, num_bins, facecolor='green', alpha=0.5)
plt.hist(original_y, num_bins, facecolor='red', alpha=0.5)
plt.hist(ok_predictions, num_bins, facecolor='blue', alpha=0.5)
#plt.hist(best_predictions, num_bins, facecolor='black', alpha=0.5)

#plt.plot(bins,'r--')
plt.xlabel('Die failure')
plt.ylabel('frequency of occurences')
plt.title('Histogram of classifying blackmaps')

# Tweak spacing to prevent clipping of ylabel
#plt.subplots_adjust(left=0.15)
plt.xlim([550, 600])
plt.ylim([0, 600])
plt.savefig('graphs/zoomed.png')

