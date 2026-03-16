import numpy as np
import pandas as pd
import pickle
import os
import pprint
import warnings
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    StratifiedKFold,
    RepeatedStratifiedKFold,
)
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFECV
from sklearn.metrics import (
    precision_recall_curve,
    matthews_corrcoef,
    average_precision_score,
    precision_score,
    make_scorer,
    brier_score_loss,
    fbeta_score,
    recall_score,
)
from sklearn.calibration import CalibratedClassifierCV
from skrebate import ReliefF

# Custom Imports (Ensure these files are in your directory)
from preprocess import Preprocess
from stat_utils import recursive_vif_selection
from feature_cols import (
    get_pitching_feature_cols,
    get_hitting_feature_cols,
    get_label_col,
)
from interpret.glassbox import ExplainableBoostingClassifier
from imblearn.ensemble import BalancedRandomForestClassifier

# 1. SETUP & CONFIGURATION
VIF_THRESHOLDS = [5.01, 10.01]
BASE_SAVE_DIR = "vif_threshold_study"
os.makedirs(BASE_SAVE_DIR, exist_ok=True)

preprocess = Preprocess()
study_summary = []

modeling_type_runs = [
    {
        "type": "hitter",
        "df": preprocess.get_df_for_modeling_hitters(),
        "feature_cols": get_hitting_feature_cols,
    },
    {
        "type": "pitcher",
        "df": preprocess.get_df_for_modeling_pitchers(),
        "feature_cols": get_pitching_feature_cols,
    },
]

# 2. MAIN EXECUTION LOOP
for modeling_type_run in modeling_type_runs:
    df = modeling_type_run["df"]
    m_type = modeling_type_run["type"]
    feature_cols = modeling_type_run["feature_cols"](df)
    label_col = get_label_col()

    X = df[feature_cols]
    y = df[label_col]

    y = y.astype(int)

    # Split once per player type to keep comparisons fair
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    for threshold in VIF_THRESHOLDS:
        print(f"\n{'='*40}")
        print(f"RUNNING: {m_type.upper()} | VIF THRESHOLD: {threshold}")
        print(f"{'='*40}")

        # Create unique folders for this specific VIF run
        threshold_dir = os.path.join(BASE_SAVE_DIR, f"vif_{threshold}")
        scalers_dir = os.path.join(threshold_dir, "scalers")
        models_dir = os.path.join(threshold_dir, "models")
        for d in [scalers_dir, models_dir]:
            os.makedirs(d, exist_ok=True)

        # STEP A: VIF Redundancy Filter
        vif_features = recursive_vif_selection(
            X_train[feature_cols], vif_threshold=threshold
        ).columns.tolist()

        # STEP B: ReliefF Interaction Ranking
        # Distance-based models require scaling
        tmp_scaler = StandardScaler()
        X_train_vif_scaled = tmp_scaler.fit_transform(X_train[vif_features])

        relief = ReliefF(
            n_features_to_select=len(vif_features), n_neighbors=100, n_jobs=-1
        )
        relief.fit(X_train_vif_scaled, y_train.values)
        relief_ranked = (
            pd.Series(relief.feature_importances_, index=vif_features)
            .sort_values(ascending=False)
            .index.tolist()
        )

        # STEP C: RFECV Subset Optimization
        f2_scorer = make_scorer(fbeta_score, beta=2)
        selector_model = LinearSVC(dual=False, random_state=42, max_iter=5000)
        rfecv = RFECV(
            estimator=selector_model,
            step=1,
            cv=StratifiedKFold(5),
            scoring=f2_scorer,
            n_jobs=-1,
        )

        # Pass ranked features to RFECV
        X_rfecv_input = X_train_vif_scaled[
            :, [vif_features.index(f) for f in relief_ranked]
        ]
        rfecv.fit(X_rfecv_input, y_train)
        golden_features = np.array(relief_ranked)[rfecv.support_].tolist()

        # STEP D: Final Scaling (On Golden Features Only)
        ss_train = StandardScaler()
        X_train_final = ss_train.fit_transform(X_train[golden_features])
        X_test_final = ss_train.transform(X_test[golden_features])

        # Save the scaler for this specific VIF path
        with open(os.path.join(scalers_dir, f"{m_type}_scaler.pkl"), "wb") as f:
            pickle.dump(ss_train, f)

        validators = {
            "StratifiedKFold": StratifiedKFold(
                n_splits=5, shuffle=True, random_state=42
            ),
            "RepeatedStratifiedKFold": RepeatedStratifiedKFold(
                n_splits=5, n_repeats=2, random_state=42
            ),
        }
        # STEP E: Model Grid Search & Evaluation
        models_to_test = {
            "LogisticRegression": {
                "model": LogisticRegression(
                    max_iter=1000, solver="liblinear", random_state=42
                ),
                "params": {
                    "C": [0.1, 1, 10],
                    "penalty": ["l1", "l2"],
                    "class_weight": ["balanced", None, {0: 1, 1: 76}],
                },
            },
            "RandomForest": {
                "model": RandomForestClassifier(random_state=42),
                "params": {
                    "n_estimators": [100, 200],
                    "max_depth": [10, 20],
                    "class_weight": [
                        "balanced",
                        "balanced_subsample",
                        None,
                        {0: 1, 1: 76},
                    ],
                },
            },
            "XGBoost": {
                "model": xgb.XGBClassifier(random_state=42, eval_metric="logloss"),
                "params": {
                    "n_estimators": [100],
                    "learning_rate": [0.05, 0.1],
                    "scale_pos_weight": [60, 70, 80],
                },
            },
            "LinearSVC": {
                "model": CalibratedClassifierCV(
                    LinearSVC(dual=False, random_state=42, max_iter=2000),
                    method="isotonic",
                    cv=5,
                ),
                "params": {
                    "estimator__C": [0.1, 1, 10],  # Note the estimator__ prefix
                    "estimator__class_weight": ["balanced", {0: 1, 1: 76}, None],
                },
            },
            "EBM": {
                "model": ExplainableBoostingClassifier(random_state=42),
                "params": {
                    "learning_rate": [0.01, 0.05],
                    "max_bins": [256],
                    "interactions": [10, 15],  # EBM searches for stat combinations
                },
            },
            "BalancedRandomForest": {
                "model": BalancedRandomForestClassifier(
                    random_state=42, sampling_strategy="auto"
                ),
                "params": {
                    "n_estimators": [100, 200],
                    "max_depth": [10, 20],
                    "replacement": [True, False],
                    "class_weight": ["balanced", "balanced_subsample", {0: 1, 1: 76}],
                },
            },
        }

        for v_name, v_obj in validators.items():
            for m_name, m_config in models_to_test.items():
                # Create the scorer
                f2_scorer = make_scorer(fbeta_score, beta=2)

                grid = GridSearchCV(
                    m_config["model"],
                    m_config["params"],
                    cv=v_obj,
                    scoring=f2_scorer,
                    n_jobs=-1,
                )
                grid.fit(X_train_final, y_train)
                best_model = grid.best_estimator_

                # Calibration & Metrics
                if hasattr(best_model, "predict_proba"):
                    y_scores = best_model.predict_proba(X_test_final)[:, 1]
                else:
                    y_scores = best_model.decision_function(X_test_final)

                precisions, recalls, thresholds = precision_recall_curve(
                    y_test, y_scores
                )

                # Brier Score (Lower is better)
                brier = brier_score_loss(y_test, y_scores)

                # Find Threshold that maximizes MCC
                target_recall = 0.9
                best_mcc = -1
                opt_t = 0.5
                for t in thresholds:
                    preds = (y_scores >= t).astype(int)
                    current_recall = recall_score(y_test, preds)
                    current_mcc = matthews_corrcoef(y_test, preds)

                    if current_recall >= target_recall:
                        if current_mcc > best_mcc:
                            best_mcc, opt_t = current_mcc, t

                # If no threshold hit x% recall, default to the highest recall found
                # if best_mcc == -1:
                #     opt_t = thresholds[np.argmax(recalls)]

                # # Calculate Final Metrics at Optimal Threshold
                # final_preds = (y_scores >= opt_t).astype(int)

                # Fallback: If no threshold hits your target, just take the max recall possible
                if best_mcc == -1:
                    opt_t = thresholds[np.argmax(recalls)]
                    final_preds = (y_scores >= opt_t).astype(int)
                    best_mcc = matthews_corrcoef(y_test, final_preds)
                else:
                    final_preds = (y_scores >= opt_t).astype(int)

                opt_precision = precision_score(y_test, final_preds, zero_division=0)
                opt_recall = recall_score(y_test, final_preds, zero_division=0)
                auprc = average_precision_score(y_test, y_scores)

                false_negatives = np.sum((y_test == 1) & (final_preds == 0))
                total_hof_in_test = np.sum(y_test == 1)

                # 3. SAVE RESULTS
                study_summary.append(
                    {
                        "Player_Type": m_type,
                        "VIF_Threshold": threshold,
                        "Model": m_name,
                        "Validator": v_name,
                        "Num_Features": len(golden_features),
                        "Opt_MCC": best_mcc,
                        "Brier_Score": brier,
                        "Opt_Recall": opt_recall,
                        "Opt_Precision": opt_precision,
                        "AUPRC": auprc,
                        "Opt_Threshold": opt_t,
                        "Golden_Features": ", ".join(golden_features),
                        "Missed_HOFers": false_negatives,
                        "Total_HOF_in_Test": total_hof_in_test,
                    }
                )

                # Save model payload
                payload = {
                    "model": best_model,
                    "threshold": opt_t,
                    "features": golden_features,
                }
                model_path = os.path.join(
                    models_dir, f"{m_type}_{v_name}_{m_name}_model.pkl"
                )
                with open(model_path, "wb") as f:
                    pickle.dump(payload, f)

# 4. FINAL EXPORT
results_df = pd.DataFrame(study_summary)
results_df.to_csv("vif_study_results.csv", index=False)

print("\n" + "=" * 40)
print("COMPLETED: Results saved to 'vif_study_results.csv'")
print("=" * 40)
print(results_df)
