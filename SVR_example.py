from numpy import genfromtxt
from sklearn.svm import SVR

# Load training data
X = genfromtxt('train_X.csv', delimiter=',')
Y = genfromtxt('train_Y.csv', delimiter=',')

# Create support vector regression object
regr = SVR(C=1.0, epsilon=0.2)

# Train the model using the training sets
regr.fit(X, Y)

# Load validation data
X_val = genfromtxt('val_X.csv', delimiter=',')
Y_val = genfromtxt('val_Y.csv', delimiter=',')

# Now predict validation output
Y_pred = regr.predict(X_val)

# Crop impossible values
Y_pred[Y_pred < 0] = 0
Y_pred[Y_pred > 600] = 600

# Calculate score
err_Y=Y_pred-Y_val
score = (sum(abs(err_Y[err_Y<0]))*10+sum(abs(err_Y[err_Y>=0])))/len(err_Y)

print score

