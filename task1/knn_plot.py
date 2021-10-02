from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
from knn_weights import *


def plot_map(x_test, y_test, x_train, y_train, k, metric, kernel):
    h = .1  # step size in the mesh
    x_min, x_max = x_test[:, 0].min() - .5, x_test[:, 0].max() + .5
    y_min, y_max = x_test[:, 1].min() - .5, x_test[:, 1].max() + .5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    cm = plt.cm.RdBu
    plt.scatter(x_train[:, 0], x_train[:, 1], c=y_train, cmap=cm, edgecolors='k')

    Z = predict_array(np.c_[xx.ravel(), yy.ravel()], x_train, y_train, k, metric, kernel)
    plt.contourf(xx, yy, Z.reshape(xx.shape), cmap=cm, alpha=.77)
