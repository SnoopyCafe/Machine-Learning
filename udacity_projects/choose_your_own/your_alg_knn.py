#!/usr/bin/python

import matplotlib.pyplot as plt
from class_vis import prettyPicture
from prep_terrain_data import makeTerrainData

features_train, labels_train, features_test, labels_test = makeTerrainData()


### the training data (features_train, labels_train) have both "fast" and "slow"
### points mixed together--separate them so we can give them different colors
### in the scatterplot and identify them visually
grade_fast = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==0]
bumpy_fast = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==0]
grade_slow = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==1]
bumpy_slow = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==1]


def show_plot():
    #### visualization
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.0)
    plt.scatter(bumpy_fast, grade_fast, color = "b", label="fast")
    plt.scatter(grade_slow, bumpy_slow, color = "r", label="slow")
    plt.legend()
    plt.xlabel("bumpiness")
    plt.ylabel("grade")
    plt.show()

################################################################################


### your code here!  name your classifier object clf if you want the 
### visualization code (prettyPicture) to show you the decision boundary

from classifiers.ClassifyKNearestNeighbors import classify
from time import time
from sklearn.metrics import accuracy_score

t0 = time()
clf = classify(features_train, labels_train, features_test, labels_test)
print ("training time: {:.2f}s".format(round(time()-t0, 3)))
print "number of features: %s " % (len(features_train[0]))

p0 = time()

try:
    pred = clf.predict(features_test)
    print ("prediction time: {:.2f}s".format(round(time() - p0, 3)))
    print ("accuracy {:.3f}".format(accuracy_score(pred, labels_test)))
    prettyPicture(clf, features_test, labels_test)
except NameError:
    pass

# Results:
# training time: 1.19s
# number of features: 2
# prediction time: 0.03s
# accuracy 0.924
