import numpy as np


def get_tp(predict, ground_truth, threshold):
    return np.add.reduce((predict[:] >= threshold) & (ground_truth == 1))


def get_fp(predict, ground_truth, threshold):
    return np.add.reduce((predict[:] >= threshold) & (ground_truth == 0))


def get_fn(predict, ground_truth, threshold):
    return np.add.reduce((predict[:] < threshold) & (ground_truth == 1))


def precision(predict, ground_truth, threshold):
    tp = get_tp(predict, ground_truth, threshold)
    fp = get_fp(predict, ground_truth, threshold)
    if tp + fp == 0:
        return 1
    return tp / (tp + fp)


def recall(predict, ground_truth, threshold):
    tp = get_tp(predict, ground_truth, threshold)
    fn = get_fn(predict, ground_truth, threshold)

    return tp / (tp + fn)


def TPRate(predict, ground_truth, threshold):
    tp = get_tp(predict, ground_truth, threshold)
    true_count = np.add.reduce(ground_truth == 1)
    return tp / true_count


def FPRate(predict, ground_truth, threshold):
    fp = np.add.reduce((predict[:] >= threshold) & (ground_truth == 0))
    false_count = np.add.reduce(ground_truth == 0)
    return fp / false_count
