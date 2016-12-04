#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 3 (decision tree) mini-project.

    Use a Decision Tree to identify emails from the Enron corpus by author:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()



#########################################################
### your code goes here ###

from sklearn import tree
from sklearn.metrics import accuracy_score
from class_vis import prettyPicture, output_image

t0 = time()
clf = tree.DecisionTreeClassifier(min_samples_split=40, criterion="gini")

# features_train = features_train[:len(features_train)/100]
# labels_train = labels_train[:len(labels_train)/100]

clf.fit(features_train, labels_train)
print ("training time: {:.2f}s".format(round(time()-t0, 3)))
print "number of features: %s " % (len(features_train[0]))

p0 = time()

pred = clf.predict(features_test)
print ("prediction time: {:.2f}s".format(round(time()-p0, 3)))

print ("accuracy {:.3f}".format(accuracy_score(pred, labels_test)))
print "number of features_tests: %s " % (len(features_test))
print "number of labels_tests: %s " % (len(labels_test))

### draw the decision boundary with the text points overlaid
# prettyPicture(clf, features_test, labels_test)

#########################################################


