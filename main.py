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

from ml_utils import print_confusion_matrix, get_scores
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
import warnings
from sklearn.exceptions import ConvergenceWarning, UndefinedMetricWarning
import sys
from stat_utils import recursive_vif_selection

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
warnings.filterwarnings("ignore", category=UserWarning)  # General sklearn warnings

# 1. Define the save directory
SAVE_DIR = "trained_models_v2"
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

preprocess = Preprocess()
# df = preprocess.get_df_for_modeling_hitters()
# all_cols = list(df.columns.values)

# pprint.pprint(all_cols)

df = preprocess.get_df_for_modeling_pitchers()
all_cols = list(df.columns.values)


# pprint.pprint(df[["BAOpp"]])
# pprint.pprint(all_cols)

feature_cols = [
    x
    for x in all_cols
    if x
    in [
        "pitching_W",
        "pitching_L",
        "pitching_G",
        "pitching_GS",
        "pitching_CG",
        "pitching_SHO",
        "pitching_SV",
        "pitching_IPouts",
        "pitching_H",
        "pitching_ER",
        "pitching_HR",
        "pitching_BB",
        "pitching_SO",
        # "pitching_BAOpp",
        "pitching_IBB",
        "pitching_WP",
        "pitching_HBP",
        "pitching_BK",
        "pitching_BFP",
        "pitching_GF",
        "pitching_R",
        "pitching_SH",
        "pitching_SF",
        "pitching_GIDP",
        "pitching_ERA",
        "pitching_WHIP",
        "pitching_KPer9",
        "post_pitching_W",
        "post_pitching_L",
        "post_pitching_G",
        "post_pitching_GS",
        "post_pitching_CG",
        "post_pitching_SHO",
        "post_pitching_SV",
        "post_pitching_IPouts",
        "post_pitching_H",
        "post_pitching_ER",
        "post_pitching_HR",
        "post_pitching_BB",
        "post_pitching_SO",
        # "post_pitching_BAOpp",
        "post_pitching_IBB",
        "post_pitching_WP",
        "post_pitching_HBP",
        "post_pitching_BK",
        "post_pitching_BFP",
        "post_pitching_GF",
        "post_pitching_R",
        "post_pitching_SH",
        "post_pitching_SF",
        "post_pitching_GIDP",
        "post_pitching_ERA",
        "post_pitching_WHIP",
        "post_pitching_KPer9",
    ]
]

# feature_cols = [
#     x
#     for x in all_cols
#     if x
#     in [
#         "batting_PA",
#         "batting_AB",
#         "batting_AVG",
#         "batting_R",
#         "batting_H",
#         "batting_2B",
#         "batting_3B",
#         "batting_HR",
#         "batting_RBI",
#         "batting_SB",
#         "batting_CS",
#         "batting_BB",
#         "batting_SO",
#         "batting_IBB",
#         "batting_HBP",
#         "batting_SH",
#         "batting_SF",
#         "batting_GIDP",
#         "batting_1B",
#         "batting_TB",
#         "batting_OBP",
#         "batting_SLG",
#         "batting_OPS",
#         "batting_SBP",
#         "batting_BAbip",
#         "batting_ISO",
#         "batting_PowerSpeedNumber",
#         "batting_ABPerHR",
#         "batting_ABPerK",
#         "post_batting_PA",
#         "post_batting_AB",
#         "post_batting_AVG",
#         "post_batting_R",
#         "post_batting_H",
#         "post_batting_2B",
#         "post_batting_3B",
#         "post_batting_HR",
#         "post_batting_RBI",
#         "post_batting_SB",
#         "post_batting_CS",
#         "post_batting_BB",
#         "post_batting_SO",
#         "post_batting_IBB",
#         "post_batting_HBP",
#         "post_batting_SH",
#         "post_batting_SF",
#         "post_batting_GIDP",
#         "post_batting_1B",
#         "post_batting_TB",
#         "post_batting_OBP",
#         "post_batting_SLG",
#         "post_batting_OPS",
#         "post_batting_SBP",
#         "post_batting_BAbip",
#         "post_batting_ISO",
#         "post_batting_PowerSpeedNumber",
#         "post_batting_ABPerHR",
#         "post_batting_ABPerK",
#         "ASGs",
#         "TripleCrowns",
#         "MVPs",
#         "ROYs",
#         "WSMVPs",
#         "GoldGloves",
#         "ASGMVPs",
#         "NLCSMVPs",
#         "ALCSMVPs",
#         "SilverSluggers",
#         "PEDUser",
#     ]
# ]
label_col = "inducted"

X = df[feature_cols]
y = df[label_col]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

derived_features = recursive_vif_selection(
    X_train[feature_cols], vif_threshold=1.01
).columns.tolist()
# derived_features = [
#     "batting_BAbip",
#     "batting_ABPerK",
#     "TripleCrowns",
#     "ROYs",
#     "NLCSMVPs",
#     "ALCSMVPs",
#     "PEDUser",
# ]
pprint.pprint(derived_features)

ss_train = StandardScaler()
X_train_scaled = ss_train.fit_transform(X_train[derived_features])
X_test_scaled = ss_train.transform(X_test[derived_features])

# Save the global scaler used for this training session
scaler_path = os.path.join(SAVE_DIR, "global_standard_scaler.pkl")
with open(scaler_path, "wb") as f:
    pickle.dump(ss_train, f)

# # 4. Access the Best Estimator
# best_model_random = random_search.best_estimator_
# print("Best Hyperparameters found by RandomizedSearchCV:", random_search.best_params_)

print("new code running")
# 1. Define the Validators you want to test
validators = {
    "KFold": KFold(n_splits=5, shuffle=True, random_state=42),
    "StratifiedKFold": StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    "RepeatedStratifiedKFold": RepeatedStratifiedKFold(
        n_splits=5, n_repeats=2, random_state=42
    ),
    "ShuffleSplit": ShuffleSplit(n_splits=5, test_size=0.2, random_state=42),
}

# 2. Define the Models and their Hyperparameters
models_to_test = {
    "LogisticRegression": {
        "model": LogisticRegression(max_iter=1000, solver="liblinear", random_state=42),
        "params": {"C": [0.01, 0.1, 1, 10], "penalty": ["l1", "l2"]},
    },
    "RandomForest": {
        "model": RandomForestClassifier(random_state=42),
        "params": {
            "n_estimators": [100, 200],
            "max_depth": [None, 10, 20],
            "class_weight": ["balanced", None],
        },
    },
    "XGBoost": {
        "model": xgb.XGBClassifier(random_state=42, eval_metric="logloss"),
        "params": {
            "n_estimators": [100],
            "learning_rate": [0.05, 0.1],
            "max_depth": [3, 6],
            "scale_pos_weight": [1, 5],  # Useful for imbalanced Hall of Fame data
        },
    },
    "LightGBM": {
        "model": lgb.LGBMClassifier(random_state=42, verbose=-1),
        "params": {
            "learning_rate": [0.01, 0.1],
            "n_estimators": [100],
            "num_leaves": [31, 50],
        },
    },
    "GradientBoosting": {
        "model": GradientBoostingClassifier(random_state=42),
        "params": {
            "n_estimators": [100],
            "learning_rate": [0.05, 0.1],
            "max_depth": [3, 5],
        },
    },
    "KNN": {
        "model": KNeighborsClassifier(),
        "params": {"n_neighbors": [3, 5, 11], "weights": ["uniform", "distance"]},
    },
    "LinearSVC": {
        "model": LinearSVC(random_state=42, max_iter=2000),
        "params": [
            {"C": [0.1, 1, 10], "penalty": ["l2"], "dual": [True]},
            {
                "C": [0.1, 1, 10],
                "penalty": ["l1"],
                "dual": [False],
                "loss": ["squared_hinge"],
            },
        ],
    },
}

# 3. The Nested Loop
results = []

for v_name, v_obj in validators.items():
    pprint.pprint(f"\n--- Running with Validator: {v_name} ---")

    for m_name, m_config in models_to_test.items():
        pprint.pprint(f"\n--- Tuning {m_name}...")

        grid_search = GridSearchCV(
            estimator=m_config["model"],
            param_grid=m_config["params"],
            cv=v_obj,
            scoring="precision",
            n_jobs=-1,
            verbose=0,
            error_score=0,
        )

        # Fit using the scaled training data
        grid_search.fit(X_train_scaled, y_train)

        # Evaluate
        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(X_test_scaled)
        test_precision = precision_score(y_test, y_pred, zero_division=0)

        # --- NEW: SAVE MODEL LOGIC ---
        # Construct a unique filename for this specific validator/model combo
        model_filename = f"{v_name}_{m_name}_model.pkl"
        model_save_path = os.path.join(SAVE_DIR, model_filename)

        with open(model_save_path, "wb") as f:
            pickle.dump(best_model, f)
        # -----------------------------

        results.append(
            {
                "Validator": v_name,
                "Model": m_name,
                "Best_Params": grid_search.best_params_,
                "Test_Precision": test_precision,
                "Saved_At": model_save_path,
            }
        )

        print_confusion_matrix(best_model, y_test, X_test_scaled)
        pprint.pprint(f"Get Scores")
        pprint.pprint(
            get_scores(best_model, X_train_scaled, y_train, X_test_scaled, y_test)
        )

# 4. Display Results in a Table
results_df = pd.DataFrame(results)
print("\nFinal Comparison Table:")
print(results_df.sort_values(by="Test_Precision", ascending=False))


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
# basically need to test with a bunch of models

# consider TimeSeriesSplit validator, would need to chronologically order data
