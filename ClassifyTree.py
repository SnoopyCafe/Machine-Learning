def classify(features_train, labels_train, features_test, labels_test):
    ### import the sklearn module for GaussianNB
    ### create classifier
    ### fit the classifier on the training features and labels
    ### return the fit classifier


    ### your code goes here!
    from sklearn import tree
    from sklearn.metrics import accuracy_score

    clf = tree.DecisionTreeClassifier(min_weight_fraction_leaf=.5, criterion="gini")
    clf.fit(features_train, labels_train)
    pred = clf.predict(features_test)

    print ( "accuracy {:.3f}".format(accuracy_score(pred, labels_test)))
    return clf
