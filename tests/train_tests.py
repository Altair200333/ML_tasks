from matplotlib.colors import ListedColormap

from classifier_manager import *
from dataset_manager import *
from metrics import *
from roc_pr_tools import *
from plot_tools import *

classifier = get_classifier(classifier_names[1])
X_train, X_test, y_train, y_test = split_dataset(get_dataset("linear"))

fit_classifier(classifier, X_train, y_train)

output = predict_classifier(classifier, X_test)

fpr, tpr = get_roc_curve(output, y_test)
print("Area under curve: ", get_auc(fpr, tpr))

recalls, precisions = get_pr_curve(output, y_test)
print("Area under curve: ", get_auc(recalls, precisions ))

showPlots([{'data': tpr, 'x': fpr, 'label': "ROC"},
           {'data': precisions, 'x': recalls, 'label': "PR"}])



#h = .02  # step size in the mesh
#x_min, x_max = X_test[:, 0].min() - .5, X_test[:, 0].max() + .5
#y_min, y_max = X_test[:, 1].min() - .5, X_test[:, 1].max() + .5
#xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
#                     np.arange(y_min, y_max, h))
#
#cm = plt.cm.RdBu
#cm_bright = ListedColormap(['#FF0000', '#0000FF'])
#
#plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright,edgecolors='k')
#
#Z = predict_classifier(classifier, np.c_[xx.ravel(), yy.ravel()])
#plt.contourf(xx, yy, Z.reshape(xx.shape), cmap=cm, alpha=.8)

show_plot()