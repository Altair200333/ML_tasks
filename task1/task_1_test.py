from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from knn import *

from space_metric import *

iris = datasets.load_iris()
X = iris.data[:, :2]
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.4, random_state=42)

metric = get_dst_metric(2)

print(measure_accuracy(X_test, y_test, X_train, y_train, metric, triangular_window))
