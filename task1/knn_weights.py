import numpy as np
from space_metric import *


def compute_distances(point, dataset, metric):
    return list(map(lambda x: metric(x, point), dataset))


def accumulate_around_label(y, kernel, distances, labels, weights):
    max_distance = distances[-1]

    values = list(
        map(lambda i: (np.float64(y == labels[i]) * weights[i] * kernel(distances[i] / max_distance)),
            np.arange(len(distances))))
    return np.add.reduce(values)


def knn(point, dataset, labels, k, metric, kernel, weights=None):
    distances = np.array(compute_distances(point, dataset, metric))
    sorted_args = np.argsort(distances)

    k_nearest_labels = labels[sorted_args][:k]
    k_nearest_distances = distances[sorted_args][:k]

    max_label = np.max(k_nearest_labels)

    if weights is not None:
        weights = weights[sorted_args][:k]
    else:
        weights = np.ones(sorted_args.shape, dtype=np.float64)

    class_values = list(
        map(lambda i: accumulate_around_label(i, kernel, k_nearest_distances, k_nearest_labels, weights),
            np.arange(max_label + 1)))
    determined_class = np.argmax(class_values)

    return determined_class


def predict_array(points, dataset, labels, k, metric, kernel, weights=None):
    return np.array(list(map(lambda x: knn(x, dataset, labels, k, metric, kernel, weights), points)))


def remove_redundant_points(x_dataset, y_dataset, k, metric, kernel, iterations=10):
    weights = np.zeros(y_dataset.shape)

    for iteration in range(iterations):
        for idx, (x, y) in enumerate(zip(x_dataset, y_dataset)):
            predicted = knn(x, x_dataset, y_dataset, k, metric, kernel, weights)
            if predicted != y:
                weights[idx] += 1

    return weights


def measure_accuracy(x_test, y_test, x_dataset, y_dataset, k, metric, kernel, weights=None):
    right = 0
    for i in range(len(y_test)):
        if knn(x_test[i], x_dataset, y_dataset, k, metric, kernel, weights) == y_test[i]:
            right += 1

    return right / len(x_test)
