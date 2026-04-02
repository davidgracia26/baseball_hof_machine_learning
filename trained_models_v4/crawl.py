import pandas as pd
import numpy as np
import pickle
import os
import glob
from collections import Counter
from trained_models_v4.preprocess import Preprocess

# ==========================================
# 1. INITIALIZATION & CONFIGURATION
# ==========================================
try:
    training_results = pd.read_csv("hall_of_fame_study_results.csv")
except FileNotFoundError:
    print("Training results CSV not found. Ensure it's in the root directory.")
    training_results = pd.DataFrame()

STUDY_BASE_DIR = "vif_threshold_study"

# Validation Lists (The "Ground Truth" legends to test against)
VALIDATION_LISTS = {
    "hitter": [
        "beltrca01",
        "jonesan01",
        "kentje01",
        "allendi01",
        "parkeda01",
        "suzukic01",
        "beltrad01",
        "heltoto01",
        "mauerjo01",
        "mcgrifr01",
        "rolensc01",
        "hodgegi01",
        "minosmi01",
        "olivato01",
        "ortizda01",
        "jeterde01",
        "simmote01",
        "walkela01",
        "baineha01",
        "martied01",
        "guerrvl01",
        "jonesch06",
        "thomeji01",
        "trammal01",
        "bagweje01",
        "raineti01",
        "rodriiv01",
    ],
    "pitcher": [
        "kaatji01",
        "sabatcc01",
        "wagnebi02",
        "hallaro01",
        "smithle02",
        "mussimi01",
        "riverma01",
        "hoffmtr01",
        "morrija02",
    ],
}


def load_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)


# --- Registry Setup ---
all_target_ids = [
    pid.lower() for sublist in VALIDATION_LISTS.values() for pid in sublist
]
hit_registry = {pid: False for pid in all_target_ids}
hit_counts = {pid: 0 for pid in all_target_ids}

# New: Feature Tracking
feature_registry = {"hitter": [], "pitcher": []}

total_models_run = 0
summary_results = []
pp = Preprocess()

# ==========================================
# 2. VALIDATION LOOP
# ==========================================
metrics = [
    d
    for d in os.listdir(STUDY_BASE_DIR)
    if os.path.isdir(os.path.join(STUDY_BASE_DIR, d)) and d != "cache"
]

for metric in metrics:
    metric_path = os.path.join(STUDY_BASE_DIR, metric)
    vif_folders = [
        d
        for d in os.listdir(metric_path)
        if os.path.isdir(os.path.join(metric_path, d))
    ]

    for vif_f in vif_folders:
        current_run_path = os.path.join(metric_path, vif_f)
        vif_val = float(vif_f.replace("vif_", ""))

        for p_type, target_ids in VALIDATION_LISTS.items():
            if p_type == "hitter":
                df = pp.get_df_for_modeling_hitters()
            else:
                df = pp.get_df_for_modeling_pitchers()

            val_df = df[
                df["playerID"].str.lower().isin([pid.lower() for pid in target_ids])
            ]
            if val_df.empty:
                continue

            scaler_path = os.path.join(
                current_run_path, "scalers", f"{p_type}_scaler.pkl"
            )
            if not os.path.exists(scaler_path):
                continue
            scaler = load_pkl(scaler_path)

            model_files = glob.glob(
                os.path.join(current_run_path, "models", f"{p_type}_*_model.pkl")
            )

            for m_file in model_files:
                payload = load_pkl(m_file)
                model = payload["model"]
                threshold = payload["threshold"]
                features = payload["features"]

                # --- Track Features for Analysis ---
                feature_registry[p_type].append(set(features))

                m_filename = (
                    os.path.basename(m_file)
                    .replace(f"{p_type}_", "")
                    .replace("_model.pkl", "")
                )
                parts = m_filename.split("_")
                validator_name = parts[0]
                model_name = "_".join(parts[1:])

                # --- PREDICTION ---
                X_scaled = scaler.transform(val_df[features])
                if hasattr(model, "predict_proba"):
                    probs = model.predict_proba(X_scaled)[:, 1]
                else:
                    probs = model.decision_function(X_scaled)

                preds = (probs >= threshold).astype(int)

                # --- UPDATE HIT REGISTRY ---
                total_models_run += 1
                for pid, is_pred_hof in zip(val_df["playerID"].str.lower(), preds):
                    if is_pred_hof == 1:
                        hit_registry[pid] = True
                        hit_counts[pid] += 1

                # --- CALCULATE VALIDATION STATS ---
                correct_count = sum(preds)
                total_count = len(target_ids)
                vhr = correct_count / total_count

                # --- FETCH HISTORICAL TRAINING METRICS ---
                match = training_results[
                    (training_results["Model"] == model_name)
                    & (training_results["Validator"] == validator_name)
                    & (training_results["Opt_Goal"] == metric)
                    & (training_results["Player_Type"] == p_type)
                    & (training_results["VIF_Threshold"] == vif_val)
                ]

                mcc = match["Result_MCC"].values[0] if not match.empty else 0
                auprc = match["Result_AUPRC"].values[0] if not match.empty else 0
                precision = (
                    match["Result_Precision"].values[0] if not match.empty else 0
                )
                recall = match["Result_Recall"].values[0] if not match.empty else 0
                brier = match["Brier_Score"].values[0] if not match.empty else 0

                winner_score = (0.4 * mcc) + (0.4 * auprc) + (0.2 * vhr)

                summary_results.append(
                    {
                        "Goal": metric,
                        "VIF": vif_f,
                        "Type": p_type,
                        "Model": m_filename,
                        "Threshold": round(threshold, 4),
                        "Correct_Count": f"{correct_count}/{total_count}",
                        "VHR": f"{vhr:.2%}",
                        "Winner_Score": f"{winner_score:.2%}",
                        "Train_Prec": f"{precision:.2%}",
                        "Train_Rec": f"{recall:.2%}",
                        "Brier": round(brier, 4),
                        "MCC": f"{mcc:.2%}",
                        "AUPRC": f"{auprc:.2%}",
                    }
                )

# ==========================================
# 3. REPORTING & EXPORT
# ==========================================
results_df = pd.DataFrame(summary_results)
results_df = results_df.sort_values(
    by=["Type", "Winner_Score"], ascending=[True, False]
)

print("\n" + "=" * 80)
print("BATCH VALIDATION REPORT")
print("=" * 80)
print(results_df.to_string(index=False))
results_df.to_csv("batch_validation_summary.csv", index=False)

# ==========================================
# 4. PLAYER MISS ANALYSIS
# ==========================================
unanimous_misses = [pid for pid, was_hit in hit_registry.items() if not was_hit]
print("\n" + "=" * 80)
print("PLAYER MISS ANALYSIS")
print("=" * 80)

if not unanimous_misses:
    print("Success! Every player was caught by at least one configuration.")
else:
    print(f"{len(unanimous_misses)} players were missed by EVERY configuration:")
    for pid in unanimous_misses:
        print(f" - {pid}")

# ==========================================
# 5. FEATURE COMMONALITY ANALYSIS
# ==========================================
print("\n" + "=" * 80)
print("FEATURE PERSISTENCE ANALYSIS")
print("=" * 80)

for p_type, sets in feature_registry.items():
    if not sets:
        continue

    # 1. Absolute Commonality (Intersection)
    common_all = set.intersection(*sets)

    # 2. Frequent Features (Appearing in > 50% of models)
    all_features_flat = [f for s in sets for f in s]
    feature_counts = Counter(all_features_flat)
    freq_threshold = len(sets) * 0.5

    print(f"\n--- {p_type.upper()}S ({len(sets)} models analyzed) ---")

    print(f"Features common to 100% of models ({len(common_all)}):")
    if common_all:
        print(", ".join(sorted(list(common_all))))
    else:
        print("None (VIF/Feature selection varied across all runs)")

    print(f"\nCore Features (Appearing in > 50% of models):")
    frequent_list = [
        f"{feat} ({count}/{len(sets)})"
        for feat, count in feature_counts.items()
        if count >= freq_threshold
    ]
    for item in sorted(frequent_list):
        print(f" - {item}")

print("\nAnalysis complete.")
