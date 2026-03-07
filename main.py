import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
import warnings
import sys
import pprint

# Model Imports
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC, SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
import xgboost as xgb
import lightgbm as lgb

# Selection & Preprocessing
from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    RandomizedSearchCV,
    StratifiedKFold,
    KFold,
    ShuffleSplit,
    RepeatedStratifiedKFold,
)
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    precision_score,
    classification_report,
    average_precision_score,
    PrecisionRecallDisplay,
    confusion_matrix,
    matthews_corrcoef,
)
from sklearn.exceptions import ConvergenceWarning, UndefinedMetricWarning

# Custom Utils (Assuming these exist in your environment)
from ml_utils import print_confusion_matrix, get_scores
from preprocess import Preprocess
from stat_utils import recursive_vif_selection
from feature_cols import pitching_feature_cols, hitting_feature_cols, label_col

from sklearn.metrics import precision_recall_curve
from sklearn.calibration import CalibratedClassifierCV  # Crucial for SVM probs

SAVE_DIR = "trained_models_v2"
scalers_target_folder = os.path.join(SAVE_DIR, "scalers")
models_target_folder = os.path.join(SAVE_DIR, "models")

for folder in [SAVE_DIR, scalers_target_folder, models_target_folder]:
    if not os.path.exists(folder):
        os.makedirs(folder)

preprocess = Preprocess()

modeling_type_runs = [
    {
        "type": "hitter",
        "df": preprocess.get_df_for_modeling_hitters(),
        "feature_cols": hitting_feature_cols,
    },
    {
        "type": "pitcher",
        "df": preprocess.get_df_for_modeling_pitchers(),
        "feature_cols": pitching_feature_cols,
    },
]

for modeling_type_run in modeling_type_runs:
    df = modeling_type_run["df"]
    feature_cols = modeling_type_run["feature_cols"]
    m_type = modeling_type_run["type"]
    pprint.pprint(
        f"model type: {m_type}, HOF percent: {df[label_col].value_counts(normalize=True) * 100}"
    )
    # hitter 1.310553% 9.38
    # pitcher 1.310553% 8.66

    X = df[feature_cols]
    y = df[label_col]

    # Use stratify=y to maintain class balance in the split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Feature Selection via VIF
    derived_features = recursive_vif_selection(
        X_train[feature_cols], vif_threshold=5.01
    ).columns.tolist()

    pprint.pprint(f"model type: {m_type}")
    pprint.pprint(derived_features)

    # Scaling
    ss_train = StandardScaler()
    X_train_scaled = ss_train.fit_transform(X_train[derived_features])
    X_test_scaled = ss_train.transform(X_test[derived_features])

    # Save Scaler
    scaler_path = os.path.join(
        scalers_target_folder, f"{m_type}_global_standard_scaler.pkl"
    )
    with open(scaler_path, "wb") as f:
        pickle.dump(ss_train, f)

    print(f"\n--- Starting Grid Search for: {m_type} ---")

    validators = {
        "StratifiedKFold": StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        "RepeatedStratifiedKFold": RepeatedStratifiedKFold(
            n_splits=5, n_repeats=2, random_state=42
        ),
    }

    models_to_test = {
        "LogisticRegression": {
            "model": LogisticRegression(
                max_iter=1000, solver="liblinear", random_state=42
            ),
            "params": {
                "C": [0.1, 1, 10],
                "penalty": ["l1", "l2"],
                "class_weight": ["balanced", None],
            },
        },
        "RandomForest": {
            "model": RandomForestClassifier(random_state=42),
            "params": {
                "n_estimators": [100, 200],
                "max_depth": [10, 20],
                "class_weight": ["balanced", "balanced_subsample", None],
            },
        },
        "XGBoost": {
            "model": xgb.XGBClassifier(random_state=42, eval_metric="logloss"),
            "params": {
                "n_estimators": [100],
                "learning_rate": [0.05, 0.1],
                "scale_pos_weight": [1, 38, 75],
            },
        },
        "LinearSVC": {
            "model": CalibratedClassifierCV(
                LinearSVC(dual=False, random_state=42, max_iter=2000)
            ),
            "params": {
                "estimator__C": [0.1, 1, 10],  # Note the estimator__ prefix
                "estimator__class_weight": ["balanced", {0: 1, 1: 76}, None],
            },
        },
    }

    results = []

    for v_name, v_obj in validators.items():
        for m_name, m_config in models_to_test.items():

            grid_search = GridSearchCV(
                estimator=m_config["model"],
                param_grid=m_config["params"],
                cv=v_obj,
                scoring="average_precision",
                n_jobs=-1,
                error_score=0,
            )

            grid_search.fit(X_train_scaled, y_train)
            best_model = grid_search.best_estimator_

            # 1. Get the probabilities (or decision scores)
            if hasattr(best_model, "predict_proba"):
                y_scores = best_model.predict_proba(X_test_scaled)[:, 1]
            else:
                y_scores = best_model.decision_function(X_test_scaled)

            # 2. Get Standard predictions (usually 0.5 threshold)
            y_pred_standard = best_model.predict(X_test_scaled)

            # 3. THRESHOLD CALIBRATION LOGIC
            # Get all possible thresholds from the PR Curve
            precisions, recalls, thresholds = precision_recall_curve(y_test, y_scores)

            best_mcc = -1
            best_threshold = 0.5

            # Iterate through thresholds to maximize MCC
            for threshold in thresholds:
                y_pred_threshold = (y_scores >= threshold).astype(int)
                current_mcc = matthews_corrcoef(y_test, y_pred_threshold)

                if current_mcc > best_mcc:
                    best_mcc = current_mcc
                    best_threshold = threshold

            # 4. Final Metrics
            test_precision = precision_score(y_test, y_pred_standard, zero_division=0)
            test_auprc = average_precision_score(y_test, y_scores)
            standard_mcc = matthews_corrcoef(y_test, y_pred_standard)

            # Save Model (including the best_threshold metadata)
            # We wrap the model and threshold in a dict to use later in FastAPI
            model_payload = {
                "model": best_model,
                "threshold": best_threshold,
                "features": derived_features,
            }

            # might need to check for loop to see if it grabs the right threshold

            model_filename = f"{v_name}_{m_name}_model.pkl"
            model_save_path = os.path.join(
                models_target_folder, f"{m_type}_{model_filename}"
            )

            with open(model_save_path, "wb") as f:
                pickle.dump(model_payload, f)

            results.append(
                {
                    "Model": m_name,
                    "Validator": v_name,
                    "Test_AUPRC": test_auprc,
                    "Std_MCC": standard_mcc,
                    "Opt_MCC": best_mcc,
                    "Best_Threshold": best_threshold,
                    "Best_Params": grid_search.best_params_,
                }
            )

    # Display Results
    results_df = pd.DataFrame(results)
    print(f"\nFinal Comparison Table for {m_type}:")
    # Sorting by Optimized MCC shows which model is best after we "fix" the threshold
    print(
        results_df.sort_values(by="Opt_MCC", ascending=False)[
            ["Model", "Validator", "Test_AUPRC", "Std_MCC", "Opt_MCC", "Best_Threshold"]
        ]
    )

# Visualize the last run's PR Curve
# PrecisionRecallDisplay.from_predictions(y_test, y_scores, name=m_name)
# plt.title(f"Precision-Recall Curve: {m_name}")
# plt.show()
