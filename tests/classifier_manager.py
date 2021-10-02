from sklearn.calibration import CalibratedClassifierCV
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import numpy as np

classifier_names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Gaussian Process",
                    "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
                    "Naive Bayes", "QDA"]

classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.025),
    SVC(gamma=2, C=1),
    GaussianProcessClassifier(1.0 * RBF(1.0)),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    MLPClassifier(alpha=1, max_iter=1000),
    AdaBoostClassifier(),
    GaussianNB(),
    QuadraticDiscriminantAnalysis()]


def get_classifier(name):
    return CalibratedClassifierCV(classifiers[classifier_names.index(name)])


def fit_classifier(classifier, train_x, train_y):
    classifier.fit(train_x, train_y)


def predict_classifier(classifier, test):
    if hasattr(classifier, "decision_function"):
        Z = classifier.decision_function(test)
    else:
        Z = classifier.predict_proba(test)[:, 1]

    return Z


def classifier_score(classifier, x_test, y_test):
    score = classifier.score(x_test, y_test)
    return score
