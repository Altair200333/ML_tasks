import numpy as np
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X, y = make_classification(n_features=2, n_redundant=0, n_informative=2,
                           random_state=1, n_clusters_per_class=1, n_samples=500)
rng = np.random.RandomState(2)
X += 2 * rng.uniform(size=X.shape)
linearly_separable = (X, y)

dataset_names = ["moons", "circles", "linear"]

datasets = [make_moons(noise=0.3, random_state=0, n_samples=500),
            make_circles(noise=0.2, factor=0.5, random_state=1, n_samples=500),
            linearly_separable
            ]


def get_dataset(name):
    return datasets[dataset_names.index(name)]


def split_dataset(dataset, seed = 42):
    X, y = dataset
    X = StandardScaler().fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.4, random_state=seed)
    return X_train, X_test, y_train, y_test
