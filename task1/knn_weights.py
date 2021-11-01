import numpy as np
from space_metric import *


def compute_distances(point, dataset, metric):
    return [metric(x, point) for x in dataset]


def accumulate_for_label(y, kernel, distances, labels, weights, max_distance):
    return np.add.reduce([np.float64(y == labels[i]) * np.float64(weights[i]) * kernel(distances[i] / max_distance) for i in range(len(distances))])


def knn(point, dataset, labels, k, metric, kernel, max_distance, weights=None):
    distances = np.array(compute_distances(point, dataset, metric))
    sorted_args = np.argsort(distances)[:k]

    k_nearest_labels = labels[sorted_args]
    k_nearest_distances = distances[sorted_args]

    max_label = np.max(labels)

    if weights is not None:
        weights = weights[sorted_args]
    else:
        weights = np.ones(sorted_args.shape, dtype=np.float64)

    class_values = [accumulate_for_label(i, kernel, k_nearest_distances, k_nearest_labels, weights, max_distance) for
                    i in np.arange(max_label + 1)]

    determined_class = np.argmax(class_values)

    return determined_class


def predict_array(points, dataset, labels, k, metric, kernel, max_distance, weights=None):
    return np.asarray([knn(x, dataset, labels, k, metric, kernel, max_distance, weights) for x in points])# np.array(list(map(lambda x: knn(x, dataset, labels, k, metric, kernel, max_distance, weights), points)))


def remove_redundant_points(x_dataset, y_dataset, k, metric, kernel, max_distance, iterations=10):
    weights = np.zeros(y_dataset.shape)

    for iteration in range(iterations):
        misses = 0
        for idx, (x, y) in enumerate(zip(x_dataset, y_dataset)):
            predicted = knn(x, x_dataset, y_dataset, k, metric, kernel, max_distance, weights)
            if predicted != y:
                misses += 1
                weights[idx] += 1

    # print(measure_accuracy(x_dataset, y_dataset, x_dataset, y_dataset, k, metric, kernel, weights))
    return weights


def measure_accuracy(x_test, y_test, x_dataset, y_dataset, k, metric, kernel, max_distance, weights=None):
    right = 0
    for i in range(len(y_test)):
        if knn(x_test[i], x_dataset, y_dataset, k, metric, kernel, max_distance, weights) == y_test[i]:
            right += 1

    return right / len(x_test)

def compute_accuracy(y_pred, y_truth):
    a = [1 for i, j in zip(y_pred, y_truth) if i in range(j-5,j+6)]

    return np.add.reduce(a)/len(y_pred)