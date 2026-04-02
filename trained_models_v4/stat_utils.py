from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import OneClassSVM
from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    RandomizedSearchCV,
    StratifiedKFold,
    KFold,
    ShuffleSplit,
    RepeatedStratifiedKFold,
)
from sklearn import metrics

from trained_models_v4.ml_utils import print_confusion_matrix, get_scores
from trained_models_v4.preprocess import Preprocess
from sklearn.preprocessing import StandardScaler
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import mutual_info_classif
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFE, RFECV
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC
from sklearn.linear_model import Lasso
import numpy as np
import pprint
import os
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
from trained_models_v4.eval_utils import EvalUtils
from sklearn.datasets import make_classification
from sklearn.metrics import precision_score, classification_report
import warnings
from sklearn.exceptions import ConvergenceWarning, UndefinedMetricWarning
import sys


def recursive_vif_selection(
    data: pd.DataFrame, vif_threshold: float = 5.0
) -> pd.DataFrame:
    """
    Recursively selects features based on Variance Inflation Factor (VIF).

    VIF = 1: No multicollinearity
    1 < VIF < 5 Low to moderate multicollinearity. May not be a major concern.
    5 < VIF < 10: High multicollinearity. Consider investigating and potentially removing variables
    VIF >= 10: Very high multicollinearity. Likely needs to be addressed to stabilize the model

    Args:
        data (pd.DataFrame): DataFrame containing the features.
        vif_threshold (float): Threshold for VIF. Features with VIF above this
                               value will be removed iteratively.

    Returns:
        pd.DataFrame: DataFrame containing the selected features (those with VIF
                      below the threshold).
    """
    if data.empty:
        return data

    # Add a constant for VIF calculation
    data_with_const = add_constant(data)

    vif = pd.DataFrame()
    vif["Feature"] = data_with_const.columns
    vif["VIF"] = [
        variance_inflation_factor(data_with_const.values, i)
        for i in range(data_with_const.shape[1])
    ]

    # Remove the constant row
    vif = vif[vif["Feature"] != "const"]

    # Find the feature with the highest VIF
    highest_vif_feature = vif.loc[vif["VIF"].idxmax()]

    if highest_vif_feature["VIF"] > vif_threshold:
        print(
            f"Removing '{highest_vif_feature['Feature']}' with VIF = {highest_vif_feature['VIF']:.2f}"
        )
        data = data.drop(columns=[highest_vif_feature["Feature"]])
        return recursive_vif_selection(data, vif_threshold)
    else:
        print("All remaining features have VIF below the threshold.")
        return data
