from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from knn_weights import *
from matplotlib.colors import ListedColormap

from space_metric import *
from knn_plot import plot_map

iris = datasets.load_iris()
X = iris.data[:, :2]
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.4, random_state=42)

metric = get_dst_metric(2)
kernel = epanchinkow_window
k = 10

print('accuracy before: ', measure_accuracy(X_test, y_test, X_train, y_train, k, metric, kernel))

weights = remove_redundant_points(X_train, y_train, k, metric, kernel)

non_zero_ids = [i for i in np.arange(weights.shape[0]) if weights[i] > 0]

reduced_x = X_train[non_zero_ids]
reduced_y = y_train[non_zero_ids]

# print(reduced_y)

print('accuracy after: ', measure_accuracy(X_test, y_test, reduced_x, reduced_y, k, metric, kernel, weights))

plot_map(X_test, y_test, reduced_x, reduced_y,  k, metric, kernel)
plt.show()
