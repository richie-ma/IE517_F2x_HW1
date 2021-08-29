#Our first machine learning model
#Garreta and Moncecchi pp 10-20
#uses Iris database and SGD classifier

## Treasury Squeeze raw score data
import pandas as pd
data =  pd.read_csv("C:/Users/ruchuan2/Box/IE 517 Machine Learning in FIN Lab\HW1\Treasury Squeeze raw score data.csv", header='infer')


import sklearn
print( 'The scikit learn version is {}.'.format(sklearn.__version__))

del data['rowindex']
del data['contract']

X_data, y_data = data[['price_crossing','price_distortion','roll_start','roll_heart','near_minus_next','ctd_last_first','ctd1_percent','delivery_cost','delivery_ratio']], data["squeeze"]
print( X_data.shape, y_data.shape)
#(900, 10) (900,)


from sklearn.model_selection import train_test_split
from sklearn import preprocessing
# Get dataset with only the first two attributes
## here I use the first two attributes

X, y = X_data[['price_crossing','price_distortion']], y_data
# Split the dataset into a training and a testing set
# Test set will be the 25% taken randomly
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.25, random_state=33)
print( X_train.shape, y_train.shape)
#(675, 2) (675,)
# Standardize the features
scaler = preprocessing.StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

import matplotlib.pyplot as plt
colors = ['red', 'blue']
#it doesnt like "xrange" changed to "range"
for i in range(len(colors)):
    xs = X_train[:, 0][y_train == i]
    ys = X_train[:, 1][y_train == i]
    plt.scatter(xs, ys, c=colors[i])
    plt.legend(data['squeeze'])
    plt.xlabel('price crossing')
    plt.ylabel('price distortion')


#found a typo here... incorrect from book followed by corrected code
#from sklearn.linear_modelsklearn._model import SGDClassifier
from sklearn.linear_model import SGDClassifier
clf = SGDClassifier()
clf.fit(X_train, y_train)

print( clf.coef_)
##[[0.71635898 0.41164431]]

print(type(clf.coef_))


print( clf.intercept_)
#[-0.17113937]

#don't forget to import Numpy as np here
import numpy as np

plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=plt.cm.prism)
ys = (clf.intercept_- xs * clf.coef_[:,0]) / clf.coef_[:,1]
plt.xlabel("price crossing")
plt.ylabel("price distortion")
plt.plot(xs, ys)
    
print( clf.predict(scaler.transform([[4.7, 3.1]])) )
#[1]

print( clf.decision_function(scaler.transform([[4.7, 3.1]])) )
##[9.60316745]

from sklearn import metrics
y_train_pred = clf.predict(X_train)
print( metrics.accuracy_score(y_train, y_train_pred) )
#0.5392592592592592

y_pred = clf.predict(X_test)
print( metrics.accuracy_score(y_test, y_pred) )
##0.5911111111111111

print( metrics.classification_report(y_test, y_pred) )

 #             precision    recall  f1-score   support

  #         0       0.67      0.63      0.65       136
   #        1       0.48      0.53      0.51        89

#    accuracy                           0.59       225
#   macro avg       0.58      0.58      0.58       225
# weighted avg       0.60      0.59      0.59       225

print( metrics.confusion_matrix(y_test, y_pred) )
#[[86 50]
#[42 47]]

print("My name is Richie Ma")
print("My NetID is: ruchuan2")
print("I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation.")
######STOP HERE######################









##error in scikit learn package, which version??
from sklearn.cross_validation import cross_val_score, KFold
from sklearn.pipeline import Pipeline
# create a composite estimator made by a pipeline of the standarization and the linear model
clf = Pipeline([(
        'scaler', StandardScaler()),
        ('linear_model', SGDClassifier())
])
# create a k-fold cross validation iterator of k=5 folds
cv = KFold(X.shape[0], 5, shuffle=True, random_state=33)
# by default the score used is the one returned by score method of the estimator (accuracy)
scores = cross_val_score(clf, X, y, cv=cv)
print( scores )
#[ 0.66666667 0.93333333 0.66666667 0.7 0.6 ]


from scipy.stats import sem
def mean_score(scores): return ("Mean score: {0:.3f} (+/- {1:.3f})").format(np.mean(scores), sem(scores))
print( mean_score(scores) )
#Mean score: 0.713 (+/-0.057)


