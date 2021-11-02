from sklearn.base import clone
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold


def cross_val(clf, X, y, folds, shuffle=False):
    kf = KFold(n_splits=folds, shuffle=shuffle)
    trained_clf = []
    accuracy_list = []

    for train_index, test_index in kf.split(X):
        X_train_fold, X_test_fold = X[train_index], X[test_index]
        y_train_fold, y_test_fold = y[train_index], y[test_index]

        clf_fold = clone(clf)
        clf_fold.fit(X_train_fold, y_train_fold)

        prediction = clf_fold.predict(X_test_fold)
        accuracy = accuracy_score(y_test_fold, prediction)

        trained_clf.append(clf_fold)
        accuracy_list.append(accuracy)

    return trained_clf, accuracy_list
