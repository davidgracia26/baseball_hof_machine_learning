from sklearn import metrics
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def print_confusion_matrix(model, y_actual, x_test):
    """Prints a confusion matrix for a model"""
    predicts = model.predict(x_test)
    cm = metrics.confusion_matrix(y_actual, predicts)

    df_cm = pd.DataFrame(cm, index=[i for i in ["Actual - No", "Actual - Yes"]],
                         columns=[i for i in ['Predicted - No', 'Predicted - Yes']])
    group_counts = ["{0:0.0f}".format(value) for value in
                    cm.flatten()]
    group_percentages = ["{0:.2%}".format(value) for value in
                         cm.flatten() / np.sum(cm)]
    labels = [f"{v1}\n{v2}" for v1, v2 in
              zip(group_counts, group_percentages)]
    labels = np.asarray(labels).reshape(2, 2)
    plt.figure(figsize=(10, 7))
    sns.heatmap(df_cm, annot=labels, fmt='')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# def get_scores(model):
#     """Returns a dataframe of Accuracy, Recall, Precision, and F1 scores for Test and Train data of a given model"""
#     pred_train = model.predict(x_train)
#     pred_test = model.predict(x_test)
#
#     scores = pd.DataFrame(
#         [{
#             "Accuracy": metrics.accuracy_score(y_train, pred_train),
#             "Recall": metrics.recall_score(y_train, pred_train),
#             "Precision": metrics.precision_score(y_train, pred_train),
#             "F1": metrics.f1_score(y_train, pred_train),
#             "ROC-AUC": metrics.roc_auc_score(y_train, pred_train)
#         },
#             {
#                 "Accuracy": metrics.accuracy_score(y_test, pred_test),
#                 "Recall": metrics.recall_score(y_test, pred_test),
#                 "Precision": metrics.precision_score(y_test, pred_test),
#                 "F1": metrics.f1_score(y_test, pred_test),
#                 "ROC-AUC": metrics.roc_auc_score(y_test, pred_test)
#             }], index=['Train', 'Test'])
#
#     return scores