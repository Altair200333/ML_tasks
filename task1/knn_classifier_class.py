from knn_weights import *
from sklearn.base import BaseEstimator, ClassifierMixin

from space_metric import *
from knn_plot import *
from knn_tests import *

class KnnPowerClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, metric, kernel, k, iterations, max_scale = 2):
        self.metric = metric
        self.kernel = kernel
        self.k = k
        self.iterations = iterations
        self.max_dst_scale = max_scale

        self.reduced_weights = self.current_weights = np.array([])
        self.reduced_x = self.current_x = np.array([])
        self.reduced_y = self.current_y = np.array([])

        self.max_distance = 0
        self.trained = False

    
    def reduce(self):
        if not self.trained:
            raise Exception('model is not trained')

        self.current_x = self.reduced_x
        self.current_y = self.reduced_y
        self.current_weights = self.reduced_weights

    
    def fit(self, X, y):
        self.trained = True

        self.max_distance =  max([max([self.metric(y, x) for x in X]) for y in X]) * self.max_dst_scale #max(self.metric(X[:, np.newaxis, :] - X[np.newaxis, :, :])) 
        
        weights = remove_redundant_points(X, y, self.k, self.metric, self.kernel, self.max_distance, self.iterations)
        non_zero_ids = [i for i in range(weights.shape[0]) if weights[i] > 0]

        self.reduced_x = X[non_zero_ids]
        self.reduced_y = y[non_zero_ids]
        self.reduced_weights = weights[non_zero_ids]
        
        #for now use full data
        self.current_x = X
        self.current_y = y
        self.current_weights = weights


    def predict(self, X):
        return np.asarray([knn(_x, self.current_x, self.current_y, self.k, self.metric, self.kernel, self.max_distance, self.current_weights) for _x in X])