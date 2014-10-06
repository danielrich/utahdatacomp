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

# Each context will essentially have a guassian distribution of particles that represent the current contributing factor for that context.
# To do a prediction we basically add up all of the active contexts and we have a nice gaussian distribution we take the mean of that
# distribution for our predicted value. To observe or take that new Y into effect we look at the error between our predicted value
# and the actual value and we divide that out among each context. Each context gets it in proportion to its uncertainty.
#
# For example if you have two contexts with std deviations of 1 and 2 and you have a total error on your prediction of 3. Then you divide
# it so that the second context takes the blame for 2x the error for the first(since it's deviation is 2x as much or it's confidence is 1/2x as much)
# I am not sure whether I should apply 100% of that error to the mean and I am also not sure what I should do with the distribution as a result.
# In this trivial easy case that means the first context would get shifted up by 1 and the second by 2. We then go through each context and apply
# 2 different noise factors( 1. a regression to the mean(so just a shift towards the mean by some decaying ammount), 2. A widening of the std deviation.
#
# Note as I think about it I don't think I need to do a particle filter totally. The sum of two normal distributions is
# new mean = meanA + meanB
# std devation^2 = std devA^2 + std devB^2
# This assumes independence which I think I can assume(hopefully this may bite me)
class my_filter():

    num_points = 0
    total_value = 0

    actual_value = 180/122 # we return a good average as default
    allowed_error = 20 # I am allowing error of 1-20 which represent shared of assigned error
    def pred_new_point(self):
        return self.actual_value

    def advance_time(self):
        self.allowed_error += 1
        if self.allowed_error > 20:
            self.allowed_error = 20
        #also perform a revert to the mean


class overall_model():

    filters = []
    def fit(self, X, Y):
        self.filters = [my_filter() for x in X[0]]
        for time in range(len(X)):
            predicted = self.predict_point(X[time])
            pred_error = Y[time] - predicted
            error_shared_points = 0
            for column in range(len(X[0])):
                if X[time][column]:
                    error_shared_points += self.filters[column].allowed_error
            point_value = pred_error / error_shared_points
            for column in range(len(X[0])):
                if X[time][column]:
                    self.filters[column].actual_value = self.filters[column].allowed_error * point_value
                    self.filters[column].allowed_error = 0
            for filt in self.filters:
                filt.advance_time()


    def predict_point(self, x):
        total = 0
        for i in range(len(x)):
            if x[i]:
                total += self.filters[i].pred_new_point()
        return total

    def predict(self, X):
        pred = [ self.predict_point(x) for x in X ]
        return pred

def run_model( model, model_name, X, Y, X_val):
    #new_values = [ [x] for x in range(len(X))]
    #X = numpy.append(X, new_values, 1)
    from sklearn.preprocessing import StandardScaler
    #scaler = StandardScaler().fit(X)
    #X = scaler.transform(X)
    #max_time_val = X[-1][-1] *2 - X[-2][-1]

    # Load validation data
    model.fit(X, Y)

    #new_values = [ [max_time_val] for x in range(len(X_val))]
    #X_val = numpy.append(X_val, new_values, 1)

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

run_model(overall_model(), "moving_average", X, Y, X_val)
