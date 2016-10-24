def classify(features_train, labels_train, features_test, labels_test):
    ### import the sklearn module for GaussianNB
    ### create classifier
    ### fit the classifier on the training features and labels
    ### return the fit classifier


    ### your code goes here!
    from sklearn.naive_bayes import GaussianNB
    from sklearn.metrics import accuracy_score

    clf = GaussianNB()
    clf.fit(features_train, labels_train)
    pred = clf.predict(features_test)
    print pred
    print ( "accuracy {:.3f}".format(accuracy_score(pred, labels_test)))
    ### accuracy = no. of points classified correctly / all points (in test set)
    return clf
    # print(clf.predict([[-0.8, -1]]))