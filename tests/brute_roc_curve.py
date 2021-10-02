import numpy as np
import matplotlib.pyplot as plt
from plot_tools import *

# prediction, class
data = np.array([
    [0.5, 0],
    [0.1, 0],
    [0.2, 0],
    [0.6, 1],
    [0.2, 1],
    [0.2, 0],
    [0.3, 1],
    [0.0, 0],
])


def generate_test_data(count: int, corruption: float):
    data = []
    preds = np.random.uniform(low=0.0, high=1.0, size=(count,))
    offsets = np.random.uniform(low=-corruption*0.5, high=corruption*0.5, size=(count,))
    for i in range(count):
        data.append([preds[i], int(np.clip(np.round(preds[i] + offsets[i]), 0, 1))])
    return np.array(data)


data = generate_test_data(100, 2.0)
sorted_data = data[np.argsort(data[:, 0])][::-1]
print(sorted_data)

predictions = sorted_data[:, 0]
print(predictions)

threshold = 0.25
thresh_labels = predictions > threshold
print(thresh_labels)

correct_labels = sorted_data[thresh_labels, 1] == 1
print(correct_labels)

m = correct_labels.shape[0]
print(m)
n = sorted_data.shape[0] - np.count_nonzero(sorted_data, axis=0)[1]
print(n)

x = []
y = []

cur_pos = [0, 0]
for i in range(sorted_data.shape[0]):
    if sorted_data[i, 1] == 1:
        cur_pos[1] += 1
    else:
        cur_pos[0] += 1

    x.append(cur_pos[0])
    y.append(cur_pos[1])

plot(x, y, False)
plt.show()
