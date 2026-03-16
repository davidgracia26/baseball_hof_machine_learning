from fastapi import FastAPI
import pandas as pd
from pydantic import BaseModel
import pickle
from preprocess import Preprocess
import os
import pprint


class SearchPlayersRequest(BaseModel):
    firstName: str
    lastName: str


class PredictHallOfFameRequest(BaseModel):
    # This now expects { "playerIds": ["id1", "id2"] }
    playerIds: list[str]


def create_players(request: SearchPlayersRequest):
    df = pd.read_csv("source_data/master.csv")

    cols = [
        "playerID",
        "nameFirst",
        "nameLast",
        "nameGiven",
        "birthYear",
        "birthMonth",
        "birthDay",
        "weight",
        "height",
        "bats",
        "throws",
    ]

    df = df[cols]
    df = df.fillna("")

    filtered_df = df[
        (df["nameFirst"].str.lower() == request.firstName.lower())
        & (df["nameLast"].str.lower() == request.lastName.lower())
    ]

    players = []
    for _, row in filtered_df.iterrows():
        player = {
            "playerID": row["playerID"],
            "nameFirst": row["nameFirst"],
            "nameLast": row["nameLast"],
            "birthYear": row["birthYear"],
            "birthMonth": row["birthMonth"],
            "birthDay": row["birthDay"],
            # "weight": row["weight"],
            # "height": row["height"],
            "bats": row["bats"],
            "throws": row["throws"],
        }
        players.append(player)

    return players


app = FastAPI()


def get_best_model(m_type: str):
    if m_type == "hitter":
        return f"vif_threshold_study/vif_10.01/models/{m_type}_StratifiedKFold_EBM_model.pkl"
    elif m_type == "pitcher":
        return f"vif_threshold_study/vif_10.01/models/{m_type}_StratifiedKFold_EBM_model.pkl"
    else:
        raise ValueError(f"{m_type} is an invalid model type")


def load_model_assets(m_type: str, version: str):
    """Helper to load the dictionary payload and scaler"""

    model_dir = ""

    if version == "v2":
        model_dir = "trained_models_v2"
    elif version == "v3":
        model_dir = ""
    scalers_path = os.path.join(model_dir, "vif_threshold_study/vif_10.01/scalers")
    models_path = os.path.join(model_dir, "")

    # Load Scaler
    scaler_file = ""
    if version == "v2":
        scaler_file = f"{m_type}_global_standard_scaler.pkl"
    elif version == "v3":
        scaler_file = f"{m_type}_scaler.pkl"

    with open(os.path.join(scalers_path, scaler_file), "rb") as f:
        scaler = pickle.load(f)

    # Load Model Payload (The dict containing 'model', 'threshold', and 'features')
    # Update the filename here to match your best performing model from the logs
    model_file = get_best_model(m_type)
    with open(os.path.join(models_path, model_file), "rb") as f:
        payload = pickle.load(f)

    return scaler, payload


@app.post("/")
def search_players(request: SearchPlayersRequest):

    players = create_players(request)

    return {"players": players}


@app.post("/predict_hof_batch_hitter")
def predict_hof_batch(request: PredictHallOfFameRequest):
    # 1. Load Assets
    scaler, payload = load_model_assets("hitter", "v3")
    model = payload["model"]
    threshold = payload["threshold"]
    feature_cols = payload["features"]  # Use the features determined by Recursive VIF

    pprint.pprint(f"model: {model}")
    pprint.pprint(f"threshold: {threshold}")
    pprint.pprint(f"feature_cols: {feature_cols}")

    # 2. Get Data
    df = Preprocess().get_df_for_modeling_hitters()
    filtered_df = df[
        df["playerID"].str.lower().isin([pid.lower() for pid in request.playerIds])
    ]

    if filtered_df.empty:
        return {"error": "No players found"}

    # 3. Predict with Calibrated Threshold
    X_scaled = scaler.transform(filtered_df[feature_cols])

    # Get probabilities
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_scaled)[:, 1]
    else:
        probs = model.decision_function(X_scaled)

    # 4. Map Results
    results = []
    # Ensure threshold is a standard float
    native_threshold = float(threshold)

    for i, (_, row) in enumerate(filtered_df.iterrows()):
        current_prob = float(probs[i])  # Ensure standard float
        is_hof = bool(current_prob >= native_threshold)

        results.append(
            {
                "playerId": str(row["playerID"]),
                "player_name": " ".join(
                    df.loc[
                        df["playerID"] == row["playerID"], ["nameFirst", "nameLast"]
                    ].values[0]
                ),
                # Change "Missed" to "Below Threshold" for a more professional tone
                "result": "Hall of Fame Caliber" if is_hof else "Below Threshold",
                "probability": f"{current_prob:.2%}",
                "met_threshold": is_hof,
                # Added: A label to explain WHY the threshold is lower
                "model_logic": "Optimized for High Recall (Legend Inclusion)",
                "model_threshold_used": f"{native_threshold:.2%}",
            }
        )

    print(
        f"{len(results)} hitters, {len([r for r in results if r['met_threshold'] == True])} correct {len([r for r in results if r['met_threshold'] == True])/len(results)}"
    )
    return {"predictions": results}


@app.post("/predict_hof_batch_pitcher")
def predict_hof_batch(request: PredictHallOfFameRequest):
    # 1. Load Assets
    scaler, payload = load_model_assets("pitcher", "v3")
    model = payload["model"]
    threshold = payload["threshold"]
    feature_cols = payload["features"]

    pprint.pprint(f"model: {model}")
    pprint.pprint(f"threshold: {threshold}")
    pprint.pprint(f"feature_cols: {feature_cols}")

    # 2. Get Data
    df = Preprocess().get_df_for_modeling_pitchers()
    filtered_df = df[
        df["playerID"].str.lower().isin([pid.lower() for pid in request.playerIds])
    ]

    if filtered_df.empty:
        return {"error": "No players found"}

    # 3. Predict
    X_scaled = scaler.transform(filtered_df[feature_cols])

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_scaled)[:, 1]
    else:
        probs = model.decision_function(X_scaled)

    # 4. Map Results
    results = []
    native_threshold = float(threshold)

    for i, (_, row) in enumerate(filtered_df.iterrows()):
        current_prob = float(probs[i])
        is_hof = bool(current_prob >= native_threshold)

        results.append(
            {
                "playerId": str(row["playerID"]),
                "player_name": " ".join(
                    df.loc[
                        df["playerID"] == row["playerID"], ["nameFirst", "nameLast"]
                    ].values[0]
                ),
                # Change "Missed" to "Below Threshold" for a more professional tone
                "result": "Hall of Fame Caliber" if is_hof else "Below Threshold",
                "probability": f"{current_prob:.2%}",
                "met_threshold": is_hof,
                # Added: A label to explain WHY the threshold is lower
                "model_logic": "Optimized for High Recall (Legend Inclusion)",
                "model_threshold_used": f"{native_threshold:.2%}",
            }
        )

    print(
        f"{len(results)} pitchers, {len([r for r in results if r['met_threshold'] == True])} correct {len([r for r in results if r['met_threshold'] == True])/len(results)}"
    )

    return {"predictions": results}


# {
#   "playerIds":
# [
# "beltrca01",
# "jonesan01",
# "kentje01",
# "allendi01",
# "parkeda01",
# "suzukic01",
# "beltrad01",
# "heltoto01",
# "mauerjo01",
# "mcgrifr01",
# "rolensc01",
# "hodgegi01",
# "minosmi01",
# "olivato01",
# "ortizda01",
# "jeterde01",
# "simmote01",
# "walkela01",
# "baineha01",
# "martied01",
# "guerrvl01",
# "jonesch06",
# "thomeji01",
# "trammal01",
# "bagweje01",
# "raineti01",
# "rodriiv01"
#   ]
# }


# pitchers
# {
#   "playerIds": [
# "kaatji01",
# "sabatc.01",
# "wagnebi02",
# "hallaro01",
# "smithle02",
# "mussimi01",
# "riverma01",
# "hoffmtr01",
# "morrija02"
# ]
# }
