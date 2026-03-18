import pandas as pd
import pickle
import os
import glob
from preprocess import Preprocess

try:
    training_results = pd.read_csv("hall_of_fame_study_results.csv")
except FileNotFoundError:
    print("Training results CSV not found. Ensure it's in the root directory.")
    training_results = pd.DataFrame()
# ==========================================
# 1. CONFIGURATION & PLAYER LISTS
# ==========================================
STUDY_BASE_DIR = "vif_threshold_study"

# The lists you provided in your API comments
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


# ==========================================
# 2. VALIDATION LOGIC
# ==========================================
summary_results = []

# Initialize Preprocessor
pp = Preprocess()

# 1. Loop through each Metric Goal (recall, precision, mcc, auprc)
metrics = [
    d
    for d in os.listdir(STUDY_BASE_DIR)
    if os.path.isdir(os.path.join(STUDY_BASE_DIR, d)) and d != "cache"
]

for metric in metrics:
    metric_path = os.path.join(STUDY_BASE_DIR, metric)

    # 2. Loop through VIF Thresholds (vif_5.01, vif_10.01)
    vif_folders = os.listdir(metric_path)
    for vif_f in vif_folders:
        current_run_path = os.path.join(metric_path, vif_f)

        # 3. Process Hitters and Pitchers separately
        for p_type, target_ids in VALIDATION_LISTS.items():
            # Get clean data for this type
            if p_type == "hitter":
                df = pp.get_df_for_modeling_hitters()
            else:
                df = pp.get_df_for_modeling_pitchers()

            # Filter for our specific validation list
            val_df = df[
                df["playerID"].str.lower().isin([pid.lower() for pid in target_ids])
            ]

            if val_df.empty:
                continue

            # Load Scaler for this specific VIF/Type run
            scaler_path = os.path.join(
                current_run_path, "scalers", f"{p_type}_scaler.pkl"
            )
            if not os.path.exists(scaler_path):
                continue
            scaler = load_pkl(scaler_path)

            # 4. Loop through every Model (XGBoost, EBM, LogReg)
            model_files = glob.glob(
                os.path.join(current_run_path, "models", f"{p_type}_*_model.pkl")
            )

            for m_file in model_files:
                payload = load_pkl(m_file)
                model = payload["model"]
                threshold = payload["threshold"]
                features = payload["features"]

                # Identify model name from filename
                m_name = (
                    os.path.basename(m_file)
                    .replace(f"{p_type}_", "")
                    .replace("_model.pkl", "")
                )

                # --- PREDICTION ---
                X_scaled = scaler.transform(val_df[features])

                if hasattr(model, "predict_proba"):
                    probs = model.predict_proba(X_scaled)[:, 1]
                else:
                    probs = model.decision_function(X_scaled)

                preds = (probs >= threshold).astype(int)

                vhr = sum(probs >= float(payload["threshold"])) / len(target_ids)

                # --- FETCH TRAINING METRICS ---
                # Match this specific model in your training CSV to get MCC and AUPRC
                validator_name, model_name = m_name.split("_")
                print(f"m_name: {m_name}")
                match = training_results[
                    (training_results["Model"] == model_name)
                    & (training_results["Validator"] == validator_name)
                    & (training_results["Opt_Goal"] == metric)
                    & (training_results["Player_Type"] == p_type)
                    & (
                        training_results["VIF_Threshold"]
                        == float(vif_f.replace("vif_", ""))
                    )
                ]

                mcc = match["Result_MCC"].values[0] if not match.empty else 0
                auprc = match["Result_AUPRC"].values[0] if not match.empty else 0
                precision = (
                    match["Result_Precision"].values[0] if not match.empty else 0
                )
                recall = match["Result_Recall"].values[0] if not match.empty else 0
                brier = match["Brier_Score"].values[0] if not match.empty else 0

                # --- THE ULTIMATE WINNER SCORE EQUATION ---
                # Weights: 40% MCC, 40% AUPRC, 20% Unseen Validation
                winner_score = (0.4 * mcc) + (0.4 * auprc) + (0.2 * vhr)

                # --- CALC ACCURACY ---
                # Since these are lists of "should be HOFers", correct = pred is 1
                correct_count = sum(preds)
                total_count = len(target_ids)
                accuracy = correct_count / total_count

                summary_results.append(
                    {
                        "Goal": metric,
                        "VIF": vif_f,
                        "Type": p_type,
                        "Model": m_name,
                        "Correct_Count": f"{correct_count}/{total_count}",
                        "Accuracy_Pct": f"{accuracy:.2%}",
                        "MCC": f"{mcc:.2%}",
                        "AUPRC": f"{auprc:.2%}",
                        "Precision": f"{precision:.2%}",  # Added Precision
                        "Recall": f"{recall:.2%}",  # Added Recall
                        "Brier": f"{brier:.4f}",
                        "VHR": f"{vhr:.2%}",
                        "Winner_Score": f"{winner_score:.2%}",
                    }
                )

# ==========================================
# 3. DISPLAY RESULTS
# ==========================================
results_df = pd.DataFrame(summary_results)
# Sort by Accuracy to see the winner
results_df = results_df.sort_values(
    by=["Type", "Winner_Score"], ascending=[True, False]
)

print("\n" + "=" * 80)
print("BATCH VALIDATION REPORT")
print("=" * 80)
print(results_df.to_string(index=False))

# Export to CSV for analysis
results_df.to_csv("batch_validation_summary.csv", index=False)
print("\nResults exported to batch_validation_summary.csv")
