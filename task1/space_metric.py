import functools

import numpy as np
from scipy.spatial import distance


def minkowski_dist(a, b, n=2):
    return distance.minkowski(a, b, n)


def get_dst_metric(n=2):
    return functools.partial(minkowski_dist, n=n)


def epanchinkow_window(r):
    return 0.75 * (1 - r ** 2) * np.float64(abs(r) <= 1.0)


def square_window(r):
    return 15.0 / 16.0 * (1 - r ** 2) * np.float64(abs(r) <= 1.0)


def triangular_window(r):
    return (1 - abs(r)) * np.float64(abs(r) <= 1.0)


def gauss_window(r):
    return (2*np.pi)**(-0.5)*np.exp(-0.5*(r**2))


