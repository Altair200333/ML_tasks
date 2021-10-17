from classifier_manager import *
from dataset_manager import *
from metrics import *
import numpy as np
from numpy import trapz


def get_fpr_tpr(labels, predictions):
    sorted_args = np.argsort(predictions)

    output_sorted = predictions[sorted_args][::-1]
    y_test_sorted = labels[sorted_args][::-1]

    eps = 1e-6
    thresholds = np.insert(np.append(output_sorted, 0 - eps), 0, 1 + eps)
    tpr = [TPRate(output_sorted, y_test_sorted, item) for item in thresholds]
    fpr = [FPRate(output_sorted, y_test_sorted, item) for item in thresholds]

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

    #thresholds = np.linspace(0, 1, num=100)[::-1]
    eps = 1e-6
    thresholds = np.insert(np.append(output_sorted, 0 - eps), 0, 1 + eps)
    precisions = [precision(output_sorted, y_test_sorted, item) for item in thresholds]
    racalls = [recall(output_sorted, y_test_sorted, item) for item in thresholds]

    return racalls, precisions


def get_pr_curve(predictions, labels):
    return get_precision_recall(labels, predictions)
