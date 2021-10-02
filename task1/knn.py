import numpy as np
from space_metric import *


def compute_distances(point, dataset, metric):
    return list(map(lambda x: metric(x, point), dataset))


def accumulate_around(y, kernel, distances, labels):
    max_distance = distances[-1]

    values = list(
        map(lambda i: (np.float64(y == labels[i]) * kernel(distances[i] / max_distance)), np.arange(len(distances))))
    return np.add.reduce(values)


def knn(point, dataset, labels, k, metric, kernel):
    distances = np.array(compute_distances(point, dataset, metric))
    sorted_args = np.argsort(distances)

    # print('distances', sorted_args)

    k_nearest_points = dataset[sorted_args][:k]
    k_nearest_labels = labels[sorted_args][:k]
    k_nearest_distances = distances[sorted_args][:k]

    # print('nearest points', k_nearest_points)
    # print('nearest labels', k_nearest_labels)
    # print('nearest distan', k_nearest_distances)

    max_label = np.max(k_nearest_labels)

    class_values = list(map(lambda i: accumulate_around(i, kernel, k_nearest_distances, k_nearest_labels),
                            np.arange(max_label + 1)))  # np.zeros(max_label + 1)
    determined_class = np.argmax(class_values)
    # print(determined_class)
    return determined_class


def measure_accuracy(x_test, y_test, x_dataset, y_dataset, metric, kernel):
    right = 0
    for i in range(len(x_test)):
        label = knn(x_test[i], x_dataset, y_dataset, 10, metric, kernel)
        right += np.float64(label == y_test[i])

    return right / len(x_test)