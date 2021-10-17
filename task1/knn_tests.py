from knn_weights import *
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from space_metric import *
from knn_plot import *


def measure_knn_data(dataset, features, k, metric, kernel, test_ratio, max_distance, shuffle_train=False, iterations = 10):
    X = dataset.data[:, :features]
    y = dataset.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_ratio, random_state=42)

    if shuffle_train:
        X_train, y_train = shuffle(X_train, y_train)

    self_acc = measure_accuracy(X_train, y_train, X_train, y_train, k, metric, kernel, max_distance)
    test_acc = measure_accuracy(X_test, y_test, X_train, y_train, k, metric, kernel, max_distance)

    test_predictions = predict_array(X_test, X_train, y_train, k, metric, kernel, max_distance)
    predictions_mask = (test_predictions == y_test)
    correct_predictions_ids = [i for i in np.arange(predictions_mask.shape[0]) if predictions_mask[i]]
    wrong_predictions_ids = [i for i in np.arange(predictions_mask.shape[0]) if not predictions_mask[i]]

    weights = remove_redundant_points(X_train, y_train, k, metric, kernel, max_distance, iterations)

    non_zero_ids = [i for i in np.arange(weights.shape[0]) if weights[i] > 0]

    reduced_x = X_train[non_zero_ids]
    reduced_y = y_train[non_zero_ids]
    reduced_weights = weights[non_zero_ids]

    self_acc_w = measure_accuracy(X_train, y_train, reduced_x, reduced_y, k, metric, kernel, max_distance,
                                  reduced_weights)
    test_acc_w = measure_accuracy(X_test, y_test, reduced_x, reduced_y, k, metric, kernel, max_distance,
                                  reduced_weights)

    return self_acc, test_acc, self_acc_w, test_acc_w


def measure_knn(dataset, features, k, metric, kernel, test_ratio, max_distance, shuffle_train=False, iterations = 10):
    self_acc, test_acc, self_acc_w, test_acc_w = measure_knn_data(dataset, features, k, metric, kernel, test_ratio, max_distance, shuffle_train, iterations)

    print('accuracy on self: ', self_acc)
    print('accuracy on test: ', test_acc)

    print('accuracy on itself after weights: ', self_acc_w)
    print('accuracy on test after weights: ', test_acc_w)


def knn_grid_search(dataset, features_to_test, k_to_test, metrics_to_test, kernels_to_test, test_ratio, seed=42):
    best_accuracy = -1.0

    best_features = None
    best_k = None
    best_metric = None
    best_kernel = None

    for features in features_to_test:
        X = dataset.data[:, :features]
        y = dataset.target
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_ratio, random_state=seed)

        for k in k_to_test:
            for metric in metrics_to_test:
                for kernel in kernels_to_test:
                    accuracy = measure_accuracy(X_test, y_test, X_train, y_train, k, metric, kernel)
                    if accuracy > best_accuracy:
                        best_accuracy = accuracy

                        best_features = features
                        best_k = k
                        best_metric = metric
                        best_kernel = kernel

    return best_accuracy, best_features, best_k, best_metric, best_kernel
