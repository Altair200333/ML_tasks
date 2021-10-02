from classifier_manager import *
from dataset_manager import *
from metrics import *
import numpy as np
from numpy import trapz


def get_fpr_tpr(labels, predictions):
    sorted_args = np.argsort(predictions)

    output_sorted = predictions[sorted_args][::-1]
    y_test_sorted = labels[sorted_args][::-1]

    thresholds = np.insert(np.append(output_sorted, 0), 0, 1)
    tpr = list(map(lambda item: TPRate(output_sorted, y_test_sorted, item), thresholds))
    fpr = list(map(lambda item: FPRate(output_sorted, y_test_sorted, item), thresholds))

    return fpr, tpr


def get_roc_curve(predictions, labels):
    fpr, tpr = get_fpr_tpr(labels, predictions)

    return fpr, tpr


def get_auc(x, y):
    area = trapz(y, x)
    return area


def get_precision_recall(labels, predictions):
    sorted_args = np.argsort(predictions)

    output_sorted = predictions[sorted_args][::-1]
    y_test_sorted = labels[sorted_args][::-1]

    thresholds = output_sorted

    precisions = list(map(lambda item: precision(output_sorted, y_test_sorted, item), thresholds))
    racalls = list(map(lambda item: recall(output_sorted, y_test_sorted, item), thresholds))

    return racalls, precisions


def get_pr_curve(predictions, labels):
    return get_precision_recall(labels, predictions)
