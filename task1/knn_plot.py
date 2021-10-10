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

    fig, ax = plt.subplots()
    scatter_markers(ax, x_train[:, 0], x_train[:, 1], y_train, 30, 'o')
    #plt.scatter(x_test[:, 0], x_test[:, 1], c=y_test, cmap=cm, edgecolors='k')

    Z = predict_array(np.c_[xx.ravel(), yy.ravel()], x_train, y_train, k, metric, kernel, max_distance)
    ax.contourf(xx, yy, Z.reshape(xx.shape), cmap=cm, alpha=.3)


def set_ax_lims(axs, xlims, ylims):
    for ax in axs:
        ax.set_xlim(xlims[0], xlims[1])
        ax.set_ylim(ylims[0], ylims[1])


def scatter_markers(ax, x, y, labels, size, marker='o'):
    cdict = {0: '#003049', 1: '#d62828', 2: '#f77f00', 3: '#fcbf49'}

    for g in np.unique(labels):
        ix = np.where(labels == g)
        ax.scatter(x[ix], y[ix], c=cdict[g], label=g, s=size, marker=marker)


def plot_correct_wrong(predictions, X_test, y_test, show = False):
    predictions_mask = (predictions == y_test)
    correct_predictions_ids = [i for i in range(predictions_mask.shape[0]) if predictions_mask[i]]
    wrong_predictions_ids = [i for i in range(predictions_mask.shape[0]) if not predictions_mask[i]]

    plt.rcParams['figure.figsize'] = [9, 6]

    ax1 = plt.subplot(1, 1, 1)

    scatter_markers(ax1, X_test[correct_predictions_ids, 0], X_test[correct_predictions_ids, 1],
                    y_test[correct_predictions_ids],
                    20)

    scatter_markers(ax1, X_test[wrong_predictions_ids, 0], X_test[wrong_predictions_ids, 1],
                    y_test[wrong_predictions_ids],
                    40, marker='x')

    x_min, x_max = X_test[:, 0].min() - .5, X_test[:, 0].max() + .5
    y_min, y_max = X_test[:, 1].min() - .5, X_test[:, 1].max() + .5

    set_ax_lims([ax1], [x_min, x_max], [y_min, y_max])

    if show:
        plt.tight_layout()
        plt.legend()
        plt.show()