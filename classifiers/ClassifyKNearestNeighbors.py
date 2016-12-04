def classify(features_train, labels_train, features_test, labels_test):
    ### import the sklearn module for classifier
    ### create classifier
    ### fit the classifier on the training features and labels
    ### return the fit classifier


    ### your code goes here!
    from sklearn.neighbors import KNeighborsClassifier

    # Create and fit an AdaBoosted decision tree
    clf = KNeighborsClassifier(n_neighbors=5, algorithm='auto')

    clf.fit(features_train, labels_train)
    pred = clf.predict(features_test)

    return clf

# Results:
# training time: 0.08s
# number of features: 2
# prediction time: 0.00s
# accuracy 0.920