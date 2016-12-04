def classify(features_train, labels_train, features_test, labels_test):
    ### import the sklearn module for GaussianNB
    ### create classifier
    ### fit the classifier on the training features and labels
    ### return the fit classifier


    ### your code goes here!
    from sklearn.ensemble import ExtraTreesClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.metrics import accuracy_score

    # Create and fit an AdaBoosted decision tree
    clf = ExtraTreesClassifier(n_estimators=200)

    clf.fit(features_train, labels_train)
    pred = clf.predict(features_test)

    return clf
