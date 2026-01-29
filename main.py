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
)
from sklearn import metrics

from ml_utils import print_confusion_matrix
from preprocess import Preprocess
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
from eval_utils import EvalUtils
from sklearn.datasets import make_classification
from sklearn.metrics import precision_score, classification_report


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


eval_utils = EvalUtils()

preprocess = Preprocess()
df = preprocess.get_df_for_modeling()
all_cols = list(df.columns.values)

pprint.pprint(all_cols)


feature_cols = [
    x
    for x in all_cols
    if x
    in [
        "batting_PA",
        "batting_AB",
        "batting_AVG",
        "batting_R",
        "batting_H",
        "batting_2B",
        "batting_3B",
        "batting_HR",
        "batting_RBI",
        "batting_SB",
        "batting_CS",
        "batting_BB",
        "batting_SO",
        "batting_IBB",
        "batting_HBP",
        "batting_SH",
        "batting_SF",
        "batting_GIDP",
        "batting_1B",
        "batting_TB",
        "batting_OBP",
        "batting_SLG",
        "batting_OPS",
        "batting_SBP",
        "batting_BAbip",
        "batting_ISO",
        "batting_PowerSpeedNumber",
        "batting_ABPerHR",
        "batting_ABPerK",
        "post_batting_PA",
        "post_batting_AB",
        "post_batting_AVG",
        "post_batting_R",
        "post_batting_H",
        "post_batting_2B",
        "post_batting_3B",
        "post_batting_HR",
        "post_batting_RBI",
        "post_batting_SB",
        "post_batting_CS",
        "post_batting_BB",
        "post_batting_SO",
        "post_batting_IBB",
        "post_batting_HBP",
        "post_batting_SH",
        "post_batting_SF",
        "post_batting_GIDP",
        "post_batting_1B",
        "post_batting_TB",
        "post_batting_OBP",
        "post_batting_SLG",
        "post_batting_OPS",
        "post_batting_SBP",
        "post_batting_BAbip",
        "post_batting_ISO",
        "post_batting_PowerSpeedNumber",
        "post_batting_ABPerHR",
        "post_batting_ABPerK",
        "ASGs",
        "TripleCrowns",
        "MVPs",
        "ROYs",
        "WSMVPs",
        "GoldGloves",
        "ASGMVPs",
        "NLCSMVPs",
        "ALCSMVPs",
        "SilverSluggers",
        "PEDUser",
    ]
]
label_col = "inducted"

X = df[feature_cols]
y = df[label_col]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

derived_features = recursive_vif_selection(
    X_train[feature_cols], vif_threshold=1.01
).columns.tolist()
# ['batting_BAbip',
#  'batting_ABPerK',
#  'TripleCrowns',
#  'ROYs',
#  'NLCSMVPs',
#  'ALCSMVPs',
#  'PEDUser']
pprint.pprint(derived_features)

ss_train = StandardScaler()
X_train = ss_train.fit_transform(X_train[derived_features])

ss_test = StandardScaler()
X_test = ss_test.fit_transform(X_test[derived_features])

# param_grid_xgb = {
#     "n_estimators": [100, 200, 300],
#     "learning_rate": [0.01, 0.05, 0.1],
#     "max_depth": [3, 4, 5],
#     "min_child_weight": [1, 3, 5],
#     "gamma": [0, 0.1, 0.2],
#     "subsample": [0.8, 1.0],
#     "colsample_bytree": [0.8, 1.0],
#     "reg_alpha": [0, 0.1],
#     "reg_lambda": [1, 1.5],
#     "objective": ["binary:logistic"],  # Or your specific objective
#     "scale_pos_weight": [1],  # Adjust if you have imbalanced data
# }

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# # Initialize GridSearchCV
# # grid_search_xgb = GridSearchCV(
# #     estimator=xgb.XGBClassifier(
# #         random_state=42, use_label_encoder=False, eval_metric="logloss"
# #     ),
# #     param_grid=param_grid_xgb,
# #     scoring="precision",  # Or your desired metric
# #     cv=cv,
# #     n_jobs=-1,  # Use all available cores
# #     verbose=2,
# # )

# # grid_search_xgb.fit(X_train, y_train)

# random_search_xgb = RandomizedSearchCV(
#     estimator=xgb.XGBClassifier(
#         random_state=42, use_label_encoder=False, eval_metric="logloss"
#     ),
#     param_distributions=param_grid_xgb,
#     n_iter=10,  # Number of parameter settings that are sampled
#     scoring="precision",
#     cv=cv,
#     n_jobs=-1,
#     verbose=2,
#     random_state=42,
# )

# random_search_xgb.fit(X_train, y_train)

# # if isinstance(grid_search_xgb, GridSearchCV):
# #     print("Best Hyperparameters (GridSearchCV):", grid_search_xgb.best_params_)
# #     print("Best Score (GridSearchCV):", grid_search_xgb.best_score_)
# #     best_xgb_model = grid_search_xgb.best_estimator_
# if isinstance(
#     random_search_xgb, RandomizedSearchCV
# ):  # isinstance(random_search_xgb, RandomizedSearchCV)
#     print("Best Hyperparameters (RandomizedSearchCV):", random_search_xgb.best_params_)
#     print("Best Score (RandomizedSearchCV):", random_search_xgb.best_score_)
#     best_xgb_model = random_search_xgb.best_estimator_

# y_pred_xgb = best_xgb_model.predict(X_test)
# precision_xgb = precision_score(y_test, y_pred_xgb)
# print("Test Precision of the Best XGBoost Model:", precision_xgb)

# print("\nClassification Report of the Best XGBoost Model:")
# print(classification_report(y_test, y_pred_xgb))
# -------------------


# 2. Define Parameter Distribution
# param_distributions = {
#     "C": np.logspace(-3, 3, 7),
#     "penalty": ["l1", "l2"],
#     "loss": ["hinge", "squared_hinge"],
#     "dual": [True, False],
#     "tol": np.logspace(-5, -2, 4),
# }

param_grid = [
    {
        "C": [0.001, 0.01, 0.1, 1, 10, 100],
        "penalty": ["l1"],
        "loss": ["squared_hinge"],
        "dual": [False],
        "tol": [1e-4, 1e-3],
    },
    {
        "C": [0.001, 0.01, 0.1, 1, 10, 100],
        "penalty": ["l2"],
        "loss": ["hinge", "squared_hinge"],
        "dual": [True, False],
        "tol": [1e-4, 1e-3],
    },
]

grid_search = GridSearchCV(
    estimator=LinearSVC(random_state=42, max_iter=1000),
    param_grid=param_grid,
    cv=cv,
    scoring="precision",  # Or your preferred metric
    n_jobs=-1,  # Use all available cores
    verbose=2,
)

# 3. Initialize and Fit RandomizedSearchCV
# random_search = RandomizedSearchCV(
#     estimator=LinearSVC(random_state=42, max_iter=1000),
#     param_distributions=param_distributions,
#     n_iter=20,
#     cv=5,
#     scoring="accuracy",
#     n_jobs=-1,
#     verbose=0,
#     random_state=42,
# )
# random_search.fit(X_train, y_train)

# # 4. Access the Best Estimator
# best_model_random = random_search.best_estimator_
# print("Best Hyperparameters found by RandomizedSearchCV:", random_search.best_params_)
grid_search.fit(X_train, y_train)

# 4. Access the Best Estimator
best_model_grid = grid_search.best_estimator_
print("Best Hyperparameters found by GridSearchCV:", grid_search.best_params_)

# 5. Make Predictions on Test Data
y_pred_grid = best_model_grid.predict(X_test)

# 6. Evaluate on Test Data
accuracy_grid = precision_score(y_test, y_pred_grid)
print(f"Test Accuracy of the Best GridSearchCV Model: {accuracy_grid:.4f}")

print_confusion_matrix(best_model_grid, y_test, X_test)

print("\nClassification Report of the Best GridSearchCV Model:")
print(classification_report(y_test, y_pred_grid))

# next steps:
# cut off data at 2023 and test if the 2024 inductees made it or not
# label the truth table better in the print out
# print out decision matrix
# label trained v test print outs
# still calc precision recall and f1
# what does 0.52 and 0.54 mean
# call functions from the notebook
# set up hot reloading correctly for the notebook
# programatize the models you're cross valiating with
