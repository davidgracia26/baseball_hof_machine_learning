import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import metrics
import pandas as pd
from sklearn.tree import plot_tree
import numpy as np


class EvalUtils:
    def __init__(self):
        pass

    # michael's code
    def print_confusion_matrix(self, model, X_test, y_actual):
        """Prints a confusion matrix for a model"""
        predicts = model.predict(X_test)
        cm = metrics.confusion_matrix(y_actual, predicts)

        df_cm = pd.DataFrame(
            cm,
            index=[i for i in ["Actual - No", "Actual - Yes"]],
            columns=[i for i in ["Predicted - No", "Predicted - Yes"]],
        )
        group_counts = ["{0:0.0f}".format(value) for value in cm.flatten()]
        group_percentages = [
            "{0:.2%}".format(value) for value in cm.flatten() / np.sum(cm)
        ]
        labels = [f"{v1}\n{v2}" for v1, v2 in zip(group_counts, group_percentages)]
        labels = np.asarray(labels).reshape(2, 2)
        plt.figure(figsize=(10, 7))
        sns.heatmap(df_cm, annot=labels, fmt="")
        plt.ylabel("True label")
        plt.xlabel("Predicted label")

    def get_scores(self, model, X_train, X_test, y_train, y_test):
        """Returns a dataframe of Accuracy, Recall, Precision, and F1 scores for Test and Train data of a given model"""
        pred_train = model.predict(X_train)
        pred_test = model.predict(X_test)

        scores = pd.DataFrame(
            [
                {
                    "Accuracy": metrics.accuracy_score(y_train, pred_train),
                    "Recall": metrics.recall_score(y_train, pred_train),
                    "Precision": metrics.precision_score(y_train, pred_train),
                    "F1": metrics.f1_score(y_train, pred_train),
                    "ROC-AUC": metrics.roc_auc_score(y_train, pred_train),
                },
                {
                    "Accuracy": metrics.accuracy_score(y_test, pred_test),
                    "Recall": metrics.recall_score(y_test, pred_test),
                    "Precision": metrics.precision_score(y_test, pred_test),
                    "F1": metrics.f1_score(y_test, pred_test),
                    "ROC-AUC": metrics.roc_auc_score(y_test, pred_test),
                },
            ],
            index=["Train", "Test"],
        )

        return scores

    # function to visualize each decision tree built
    def print_tree(self, model, X_train):
        """
        Prints a decision tree including feature names
        """
        # feature_names = X_train.columns.to_list()
        feature_names = X_train.tolist()
        plt.figure(figsize=(20, 30))
        out = plot_tree(
            model,
            feature_names=feature_names,
            filled=True,
            fontsize=9,
            node_ids=False,
            class_names=None,
        )
        # below code will add arrows to the decision tree split if they are missing
        for o in out:
            arrow = o.arrow_patch
            if arrow is not None:
                arrow.set_edgecolor("black")
                arrow.set_linewidth(1)
        plt.show()

    def get_importances(self, model, X_train):
        """Print the Importances graph for any classifier"""
        imp = (
            pd.DataFrame(
                model.feature_importances_,
                columns=["Importance"],
                index=X_train,
            )
            .sort_values(by="Importance", ascending=True)
            .query("Importance > 0")
        )
        plt.figure(figsize=(12, 12))
        plt.title("Feature Importances")
        plt.barh(
            range(len(imp.index)), imp["Importance"], color="violet", align="center"
        )
        plt.yticks(range(len(imp.index)), imp.index)
        plt.xlabel("Relative Importance")
        plt.show()

    def plot_precision_recall_curve(self, model, X_test, y_test):
        """Plots the Precision / Recall curves and returns the optimal balanced threshold"""
        y_scores = model.predict(X_test)  # [:, 1]
        precisions, recalls, thresholds = metrics.precision_recall_curve(
            y_test, y_scores, pos_label=1
        )
        display1 = metrics.PrecisionRecallDisplay(precisions, recalls)
        display1.plot()

    # Get the ROC AUC score and ROC curve and plot
    def print_roc_auc(self, model, X_test, y_test):
        """Print the ROC Curve and AUC"""
        logit_roc_auc_test = metrics.roc_auc_score(
            y_test, model.predict_proba(X_test)[:, 1]
        )
        fpr, tpr, thresholds = metrics.roc_curve(
            y_test, model.predict_proba(X_test)[:, 1]
        )
        sns.lineplot(x=fpr, y=tpr)
        plt.plot([0, 1], [0, 1], "r--")
        plt.xlabel("FPR")
        plt.ylabel("TPR")
        plt.title("ROC-AUC-Score (area = %0.2f)" % logit_roc_auc_test)
        plt.show()
