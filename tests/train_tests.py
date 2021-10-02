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

show_plot()