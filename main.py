import numpy as np
import pandas as pd
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb

from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
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
    recall_score,
    confusion_matrix,
    fbeta_score,
    PrecisionRecallDisplay,
)
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from skrebate import ReliefF

# Custom Imports (Ensure these files exist in your directory)
from preprocess import Preprocess
from stat_utils import recursive_vif_selection
from feature_cols import (
    get_pitching_feature_cols,
    get_hitting_feature_cols,
    get_label_col,
)
from interpret.glassbox import ExplainableBoostingClassifier
from imblearn.ensemble import BalancedRandomForestClassifier


# ==========================================
# 1. DIAGNOSTIC PLOTTING FUNCTION
# ==========================================
def save_model_diagnostics(
    y_test, y_scores, y_preds, model, features, path_prefix, metric_name
):
    """Saves a 4-panel diagnostic figure (PR Curve, Calibration, Confusion Matrix, Feature Importance)."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Panel 1: Precision-Recall Curve
    PrecisionRecallDisplay.from_predictions(y_test, y_scores, ax=axes[0, 0])
    axes[0, 0].set_title(f"PR Curve (Goal: {metric_name})")

    # Panel 2: Calibration Curve
    prob_true, prob_pred = calibration_curve(y_test, y_scores, n_bins=10)
    axes[0, 1].plot(prob_pred, prob_true, marker="o", label="Model")
    axes[0, 1].plot([0, 1], [0, 1], linestyle="--", color="gray", label="Ideal")
    axes[0, 1].set_title("Calibration / Reliability")
    axes[0, 1].set_xlabel("Predicted Probability")
    axes[0, 1].set_ylabel("Actual Proportion")
    axes[0, 1].legend()

    # Panel 3: Confusion Matrix
    cm = confusion_matrix(y_test, y_preds)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[1, 0], cbar=False)
    axes[1, 0].set_title("Confusion Matrix (Raw Counts)")
    axes[1, 0].set_xlabel("Predicted Label")
    axes[1, 0].set_ylabel("True Label")

    # Panel 4: Feature Importance
    importances = None
    title = "Top 15 Features"
    if hasattr(model, "feature_importances_"):
        importances = pd.Series(model.feature_importances_, index=features)
    elif hasattr(model, "coef_"):
        importances = pd.Series(model.coef_[0], index=features).abs()
        title = "Absolute Coefficients"
    elif hasattr(model, "calibrated_classifiers_"):
        coefs = [clf.estimator.coef_[0] for clf in model.calibrated_classifiers_]
        importances = pd.Series(np.mean(coefs, axis=0), index=features).abs()
        title = "Mean Absolute Weights (SVC)"

    if importances is not None:
        importances.sort_values(ascending=True).tail(15).plot(
            kind="barh", ax=axes[1, 1], color="skyblue"
        )
        axes[1, 1].set_title(title)
    else:
        axes[1, 1].text(0.5, 0.5, "Importance not available", ha="center")

    plt.suptitle(f"Diagnostics: {os.path.basename(path_prefix)}", fontsize=20)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f"{path_prefix}_diagnostics.png")
    plt.close()


# ==========================================
# 2. SETUP & CONFIGURATION
# ==========================================
VIF_THRESHOLDS = [5.01, 10.01]
BASE_SAVE_DIR = "vif_threshold_study"
CACHE_DIR = os.path.join(BASE_SAVE_DIR, "cache")

# Define separate optimization goals
SCORING_METRICS = {
    "mcc": make_scorer(matthews_corrcoef),
    "precision": make_scorer(precision_score, zero_division=0),
    "recall": make_scorer(recall_score),
    "f2": make_scorer(fbeta_score, beta=2),
    "fPoint5": make_scorer(fbeta_score, beta=0.5),
    "auprc": "average_precision",
}

os.makedirs(BASE_SAVE_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

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

# ==========================================
# 3. MAIN EXECUTION LOOP
# ==========================================
for metric_name, scorer_obj in SCORING_METRICS.items():
    print(f"\n\n{'#'*60}\nGLOBAL OPTIMIZATION: {metric_name.upper()}\n{'#'*60}")

    for run in modeling_type_runs:
        df, m_type = run["df"], run["type"]
        feature_cols = run["feature_cols"](df)
        label_col = get_label_col()

        X = df[feature_cols]
        y = df[label_col].astype(int)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        for threshold in VIF_THRESHOLDS:
            print(f"\n--- {m_type.upper()} | VIF: {threshold} ---")

            # Result directory structure
            metric_dir = os.path.join(BASE_SAVE_DIR, metric_name, f"vif_{threshold}")
            scalers_dir = os.path.join(metric_dir, "scalers")
            models_dir = os.path.join(metric_dir, "models")
            plots_dir = os.path.join(metric_dir, "plots")
            for d in [scalers_dir, models_dir, plots_dir]:
                os.makedirs(d, exist_ok=True)

            # Caching Logic
            vif_path = os.path.join(CACHE_DIR, f"vif_{m_type}_{threshold}.pkl")
            relief_path = os.path.join(CACHE_DIR, f"relief_{m_type}_{threshold}.pkl")
            rfecv_path = os.path.join(
                CACHE_DIR, f"rfecv_{m_type}_{threshold}_{metric_name}.pkl"
            )

            # STEP A: VIF
            if os.path.exists(vif_path):
                with open(vif_path, "rb") as f:
                    vif_features = pickle.load(f)
            else:
                vif_features = recursive_vif_selection(
                    X_train[feature_cols], vif_threshold=threshold
                ).columns.tolist()
                with open(vif_path, "wb") as f:
                    pickle.dump(vif_features, f)

            # STEP B: ReliefF
            tmp_scaler = StandardScaler()
            X_train_vif_scaled = tmp_scaler.fit_transform(X_train[vif_features])
            if os.path.exists(relief_path):
                with open(relief_path, "rb") as f:
                    relief_ranked = pickle.load(f)
            else:
                relief = ReliefF(
                    n_features_to_select=len(vif_features), n_neighbors=100, n_jobs=-1
                )
                relief.fit(X_train_vif_scaled, y_train.values)
                relief_ranked = (
                    pd.Series(relief.feature_importances_, index=vif_features)
                    .sort_values(ascending=False)
                    .index.tolist()
                )
                with open(relief_path, "wb") as f:
                    pickle.dump(relief_ranked, f)

            # STEP C: RFECV (Optimized for Current Metric)
            if os.path.exists(rfecv_path):
                with open(rfecv_path, "rb") as f:
                    golden_features = pickle.load(f)
            else:
                selector = LinearSVC(dual=False, random_state=42, max_iter=5000)
                rfecv = RFECV(
                    estimator=selector,
                    step=1,
                    cv=StratifiedKFold(5),
                    scoring=scorer_obj,
                    n_jobs=-1,
                )
                X_rfecv_input = X_train_vif_scaled[
                    :, [vif_features.index(f) for f in relief_ranked]
                ]
                rfecv.fit(X_rfecv_input, y_train)
                golden_features = np.array(relief_ranked)[rfecv.support_].tolist()
                with open(rfecv_path, "wb") as f:
                    pickle.dump(golden_features, f)

            # STEP D: Final Scaling
            ss = StandardScaler()
            X_train_final = ss.fit_transform(X_train[golden_features])
            X_test_final = ss.transform(X_test[golden_features])
            with open(os.path.join(scalers_dir, f"{m_type}_scaler.pkl"), "wb") as f:
                pickle.dump(ss, f)

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
                    "model": xgb.XGBClassifier(
                        random_state=42,
                        eval_metric="logloss",
                        objective="binary:logistic",
                    ),
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
                        "class_weight": [
                            "balanced",
                            "balanced_subsample",
                            {0: 1, 1: 76},
                        ],
                    },
                },
            }

            for v_name, v_obj in validators.items():
                for m_name, m_config in models_to_test.items():
                    grid = GridSearchCV(
                        m_config["model"],
                        m_config["params"],
                        cv=v_obj,
                        scoring=scorer_obj,
                        n_jobs=-1,
                    )
                    grid.fit(X_train_final, y_train)
                    best_model = grid.best_estimator_

                    y_scores = (
                        best_model.predict_proba(X_test_final)[:, 1]
                        if hasattr(best_model, "predict_proba")
                        else best_model.decision_function(X_test_final)
                    )

                    # Threshold Selection
                    precisions, recalls, thresholds = precision_recall_curve(
                        y_test, y_scores
                    )

                    brier = brier_score_loss(y_test, y_scores)

                    n_t = len(thresholds)

                    if metric_name == "recall":
                        valid_idx = np.where(recalls[:n_t] >= 0.90)[0]
                        opt_t = thresholds[valid_idx[-1]] if len(valid_idx) > 0 else 0.5
                    elif metric_name == "precision":
                        opt_t = thresholds[np.argmax(precisions[:n_t])]
                    else:
                        opt_t = 0.5

                    final_preds = (y_scores >= opt_t).astype(int)

                    # Save Diagnostics
                    diag_path = os.path.join(plots_dir, f"{m_type}_{m_name}")
                    save_model_diagnostics(
                        y_test,
                        y_scores,
                        final_preds,
                        best_model,
                        golden_features,
                        diag_path,
                        metric_name,
                    )

                    study_summary.append(
                        {
                            "Opt_Goal": metric_name,
                            "Player_Type": m_type,
                            "VIF_Threshold": threshold,
                            "Model": m_name,
                            "Validator": v_name,
                            "Result_MCC": matthews_corrcoef(y_test, final_preds),
                            "Result_Precision": precision_score(
                                y_test, final_preds, zero_division=0
                            ),
                            "Result_Recall": recall_score(y_test, final_preds),
                            "Result_AUPRC": average_precision_score(y_test, y_scores),
                            "Brier_Score": brier,
                            "Opt_Threshold": opt_t,
                            "Num_Features": len(golden_features),
                            "Golden_Features": ", ".join(golden_features),
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
pd.DataFrame(study_summary).to_csv("hall_of_fame_study_results.csv", index=False)
print("\nExecution Complete. All models, plots, and results saved.")
