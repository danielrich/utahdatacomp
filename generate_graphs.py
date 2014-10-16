import matplotlib
matplotlib.use('Agg')
from numpy import genfromtxt
import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

bad_predictions = genfromtxt('predictions/final_pred_y.csv', delimiter=',')

num_bins = 50
# the histogram of the data
n, bins, patches = plt.hist(bad_predictions, num_bins, normed=1, facecolor='green', alpha=0.5)

plt.plot(bins,'r--')
plt.xlabel('Die failure')
plt.ylabel('frequency of occurences')
plt.title('Histogram of adaboost predictions')

# Tweak spacing to prevent clipping of ylabel
#plt.subplots_adjust(left=0.15)
plt.savefig('graphs/bad_histogram.png')

