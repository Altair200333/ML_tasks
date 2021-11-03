from knn_weights import *
from sklearn.base import BaseEstimator, ClassifierMixin

from space_metric import *
from knn_plot import *
from knn_tests import *

class KnnPowerClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, metric, kernel, k, iterations):
        self.metric = metric
        self.kernel = kernel
        self.k = k
        self.iterations = iterations
        
        self.reduced_weights = np.array([])
        self.reduced_x = np.array([])
        self.reduced_y = np.array([])

        self.max_distance = 0

    def fit(self, X, y):
        self.max_distance =  max([max([self.metric(y, x) for x in X]) for y in X]) * 2 #max(self.metric(X[:, np.newaxis, :] - X[np.newaxis, :, :])) 
        weights = remove_redundant_points(X, y, self.k, self.metric, self.kernel, self.max_distance, self.iterations)
        non_zero_ids = [i for i in range(weights.shape[0]) if weights[i] > 0]

        self.reduced_x = X[non_zero_ids]
        self.reduced_y = y[non_zero_ids]
        self.reduced_weights = weights[non_zero_ids]
        
    def predict(self, X):
        return np.asarray([knn(_x, self.reduced_x, self.reduced_y, self.k, self.metric, self.kernel, self.max_distance, self.reduced_weights) for _x in X])