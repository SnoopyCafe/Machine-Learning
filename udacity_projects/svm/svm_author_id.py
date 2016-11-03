#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess
#from class_vis import prettyPicture

### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()


#########################################################
### your code goes here ###

#########################################################

from sklearn import svm
from sklearn.metrics import accuracy_score

# clf = svm.SVC(kernel="linear") # accuracy = .984
clf = svm.SVC(kernel="rbf",C=10000)  # .991 using full data set
# clf = svm.SVC(kernel="linear", gamma=500, C=10.0)

t0 = time()

# One way to speed up an algorithm is to train it on a smaller training dataset.
# The tradeoff is that the accuracy almost always goes down when you do this.
#features_train = features_train[:len(features_train)/100]
#labels_train = labels_train[:len(labels_train)/100]

clf.fit(features_train, labels_train)
print "training time:", round(time()-t0, 3), "s"

p0 = time()
pred = clf.predict(features_test)
print "prediction time:", round(time()-p0, 3), "s"
print ( "accuracy {:.3f}".format(accuracy_score(pred, labels_test)))

chris = [x+1 for x in pred if x == 1 ]
print len(pred),len(chris)

# Predicted 877 emails where from Chris

#print(pred[10],pred[26],pred[50])
### draw the decision boundary with the text points overlaid
#prettyPicture(clf, features_test, labels_test)