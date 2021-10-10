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
X = iris.data[:, :4]
# X = StandardScaler().fit_transform(X)
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.4, random_state=43)

metric = get_dst_metric(2)
kernel = triangular_window
k = 20

best_accuracy, best_features, best_k, best_metric, best_kernel = knn_grid_search(iris, [2, 3, 4], [4, 6, 8, 10, 12, 14, 16],
                                                                                 [get_dst_metric(2), get_dst_metric(3), get_dst_metric(4)],
                                                                                 [triangular_window], 0.2)

print('best accuracy: ', best_accuracy, 'k: ', best_k, 'metric: ', str(best_metric), 'kernel: ', str(best_kernel))

print('accuracy before: ', measure_accuracy(X_test, y_test, X_train, y_train, k, metric, kernel))

test_predictions = predict_array(X_test, X_train, y_train, k, metric, kernel)
predictions_mask = (test_predictions == y_test)
correct_predictions_ids = [i for i in np.arange(predictions_mask.shape[0]) if predictions_mask[i]]
wrong_predictions_ids = [i for i in np.arange(predictions_mask.shape[0]) if not predictions_mask[i]]
print(correct_predictions_ids)
print(wrong_predictions_ids)

weights = remove_redundant_points(X_train, y_train, k, metric, kernel)

non_zero_ids = [i for i in np.arange(weights.shape[0]) if weights[i] > 0]

reduced_x = X_train[non_zero_ids]
reduced_y = y_train[non_zero_ids]
reduced_weights = weights[non_zero_ids]
# print(reduced_y)

print('accuracy after weights: ',
      measure_accuracy(X_test, y_test, reduced_x, reduced_y, k, metric, kernel, reduced_weights))
print('accuracy after: ', measure_accuracy(X_test, y_test, reduced_x, reduced_y, k, metric, kernel))

print('accuracy on self: ', measure_accuracy(X_train, y_train, reduced_x, reduced_y, k, metric, kernel, reduced_weights))

# plot_map(X_test, y_test, reduced_x, reduced_y,  k, metric, kernel)
plt.show()
