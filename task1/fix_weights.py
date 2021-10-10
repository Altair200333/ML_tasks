from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from knn_weights import *
from matplotlib.colors import ListedColormap

from space_metric import *
from knn_plot import plot_map
from knn_tests import *

iris = datasets.load_iris()
X = iris.data[:, :]
# X = StandardScaler().fit_transform(X)
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.4, random_state=42)

metric = get_dst_metric(2)
kernel = triangular_window
k = 10
max_distance = max([metric(X_train[0], x) for x in X_train]) * 0.25

weights_1 = np.arange(0, len(X_train))
print('accuracy on self before: ', measure_accuracy(X_train, y_train, X_train, y_train, k, metric, kernel, max_distance))

weights = remove_redundant_points(X_train, y_train, k, metric, kernel, 30)
print(weights)
print('accuracy on self raw weights: ', measure_accuracy(X_train, y_train, X_train, y_train, k, metric, kernel,
                                                         max_distance, weights))

non_zero_ids = [i for i in range(weights.shape[0]) if weights[i] > 0]

reduced_x = X_train[non_zero_ids]
reduced_y = y_train[non_zero_ids]
reduced_weights = weights[non_zero_ids]

print(reduced_weights)

print('accuracy on self: ', measure_accuracy(X_train, y_train, reduced_x, reduced_y, k, metric, kernel, max_distance, reduced_weights))
