from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
from knn_weights import *


def plot_map(x_test, y_test, x_train, y_train, k, metric, kernel, max_distance):
    h = .1  # step size in the mesh
    x_min, x_max = x_test[:, 0].min() - .5, x_test[:, 0].max() + .5
    y_min, y_max = x_test[:, 1].min() - .5, x_test[:, 1].max() + .5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    cm = plt.cm.RdBu
    plt.scatter(x_test[:, 0], x_test[:, 1], c=y_test, cmap=cm, edgecolors='k')

    Z = predict_array(np.c_[xx.ravel(), yy.ravel()], x_train, y_train, k, metric, kernel, max_distance)
    plt.contourf(xx, yy, Z.reshape(xx.shape), cmap=cm, alpha=.77)


def set_ax_lims(axs, xlims, ylims):
    for ax in axs:
        ax.set_xlim(xlims[0], xlims[1])
        ax.set_ylim(ylims[0], ylims[1])
