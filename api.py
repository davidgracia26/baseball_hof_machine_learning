from fastapi import FastAPI
import pandas as pd
from pydantic import BaseModel
import pickle
from preprocess import Preprocess
import os


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


@app.post("/")
def search_players(request: SearchPlayersRequest):

    players = create_players(request)

    return {"players": players}


@app.post("/predict_hof_batch_hitter")
def predict_hof_batch(request: PredictHallOfFameRequest):
    # 1. Load Data and Models (as we did before)
    df = Preprocess().get_df_for_modeling_hitters()

    MODEL_DIR = "trained_models_v2"
    scalers_path = os.path.join(MODEL_DIR, "scalers")
    models_path = os.path.join(MODEL_DIR, "models")
    with open(
        os.path.join(scalers_path, "hitter_global_standard_scaler.pkl"), "rb"
    ) as f:
        scaler = pickle.load(f)
    with open(
        # os.path.join(models_path, "hitter_StratifiedKFold_LinearSVC_model.pkl"), "rb"
        os.path.join(models_path, "hitter_StratifiedKFold_XGBoost_model.pkl"),
        "rb",
    ) as f:
        model = pickle.load(f)

    # feature_cols = [
    #     "batting_BAbip",
    #     "batting_ABPerK",
    #     "TripleCrowns",
    #     "ROYs",
    #     "NLCSMVPs",
    #     "ALCSMVPs",
    #     "PEDUser",
    # ]

    # feature_cols = [
    #     "batting_3B",
    #     "batting_SB",
    #     "batting_CS",
    #     "batting_IBB",
    #     "batting_HBP",
    #     "batting_SH",
    #     "batting_GIDP",
    #     "batting_OBP",
    #     "batting_SBP",
    #     "batting_BAbip",
    #     "batting_ISO",
    #     "batting_ABPerHR",
    #     "batting_ABPerK",
    #     "post_batting_3B",
    #     "post_batting_HR",
    #     "post_batting_SB",
    #     "post_batting_CS",
    #     "post_batting_IBB",
    #     "post_batting_HBP",
    #     "post_batting_SH",
    #     "post_batting_SF",
    #     "post_batting_GIDP",
    #     "post_batting_SBP",
    #     "post_batting_BAbip",
    #     "post_batting_ISO",
    #     "post_batting_PowerSpeedNumber",
    #     "post_batting_ABPerHR",
    #     "post_batting_ABPerK",
    #     "ASGs",
    #     "TripleCrowns",
    #     "MVPs",
    #     "ROYs",
    #     "WSMVPs",
    #     "GoldGloves",
    #     "ASGMVPs",
    #     "NLCSMVPs",
    #     "ALCSMVPs",
    #     "SilverSluggers",
    #     "PEDUser",
    # ]

    feature_cols = [
        "batting_3B",
        "batting_SB",
        "batting_CS",
        "batting_IBB",
        "batting_HBP",
        "batting_SH",
        "batting_GIDP",
        "batting_OBP",
        "batting_SBP",
        "batting_BAbip",
        "batting_ISO",
        "batting_ABPerHR",
        "batting_ABPerK",
        "post_batting_3B",
        "post_batting_HR",
        "post_batting_SB",
        "post_batting_CS",
        "post_batting_IBB",
        "post_batting_HBP",
        "post_batting_SH",
        "post_batting_SF",
        "post_batting_GIDP",
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

    # 2. Filter DF for all IDs in the list
    # We use .isin() to get everyone at once
    filtered_df = df[
        df["playerID"].str.lower().isin([pid.lower() for pid in request.playerIds])
    ]

    if filtered_df.empty:
        return {"error": "No players found"}

    # 3. Scale and Predict
    X = filtered_df[feature_cols]
    X_scaled = scaler.transform(X)

    predictions = model.predict(X_scaled)
    # Get probabilities if the model supports it
    probs = (
        model.predict_proba(X_scaled)[:, 1]
        if hasattr(model, "predict_proba")
        else [None] * len(predictions)
    )

    # 4. Map results back to IDs
    results = []
    for idx, row in filtered_df.iterrows():
        results.append(
            {
                "playerId": row["playerID"],
                "result": (
                    "Hall of Famer" if predictions[len(results)] == 1 else "Missed"
                ),
                "probability": (
                    f"{probs[len(results)]:.2%}" if probs[0] is not None else "N/A"
                ),
            }
        )

    return {"predictions": results}


@app.post("/predict_hof_batch_pitcher")
def predict_hof_batch(request: PredictHallOfFameRequest):
    # 1. Load Data and Models (as we did before)
    df = Preprocess().get_df_for_modeling_pitchers()

    MODEL_DIR = "trained_models_v2"
    scalers_path = os.path.join(MODEL_DIR, "scalers")
    models_path = os.path.join(MODEL_DIR, "models")
    with open(
        os.path.join(scalers_path, "pitcher_global_standard_scaler.pkl"), "rb"
    ) as f:
        scaler = pickle.load(f)
    with open(
        # os.path.join(models_path, "pitcher_StratifiedKFold_XGBoost_model.pkl"), "rb"
        os.path.join(models_path, "pitcher_StratifiedKFold_XGBoost_model.pkl"),
        "rb",
    ) as f:
        model = pickle.load(f)

    # feature_cols = [
    #     "pitching_GIDP",
    #     "pitching_ERA",
    #     "post_pitching_SHO",
    #     "post_pitching_SV",
    # ]

    # feature_cols = [
    #     "pitching_CG",
    #     "pitching_SV",
    #     "pitching_IBB",
    #     "pitching_WP",
    #     "pitching_HBP",
    #     "pitching_BK",
    #     "pitching_SH",
    #     "pitching_GIDP",
    #     "pitching_ERA",
    #     "pitching_KPer9",
    #     "post_pitching_W",
    #     "post_pitching_L",
    #     "post_pitching_G",
    #     "post_pitching_CG",
    #     "post_pitching_SHO",
    #     "post_pitching_SV",
    #     "post_pitching_HR",
    #     "post_pitching_IBB",
    #     "post_pitching_WP",
    #     "post_pitching_HBP",
    #     "post_pitching_BK",
    #     "post_pitching_SH",
    #     "post_pitching_SF",
    #     "post_pitching_GIDP",
    #     "post_pitching_ERA",
    #     "post_pitching_KPer9",
    # ]

    feature_cols = [
        "pitching_CG",
        "pitching_SV",
        "pitching_IBB",
        "pitching_WP",
        "pitching_HBP",
        "pitching_BK",
        "pitching_SH",
        "pitching_GIDP",
        "pitching_ERA",
        "pitching_KPer9",
        "post_pitching_W",
        "post_pitching_L",
        "post_pitching_G",
        "post_pitching_CG",
        "post_pitching_SHO",
        "post_pitching_SV",
        "post_pitching_HR",
        "post_pitching_IBB",
        "post_pitching_WP",
        "post_pitching_HBP",
        "post_pitching_BK",
        "post_pitching_SH",
        "post_pitching_SF",
        "post_pitching_GIDP",
        "post_pitching_ERA",
        "post_pitching_KPer9",
    ]

    # 2. Filter DF for all IDs in the list
    # We use .isin() to get everyone at once
    filtered_df = df[
        df["playerID"].str.lower().isin([pid.lower() for pid in request.playerIds])
    ]

    if filtered_df.empty:
        return {"error": "No players found"}

    # 3. Scale and Predict
    X = filtered_df[feature_cols]
    X_scaled = scaler.transform(X)

    predictions = model.predict(X_scaled)
    # Get probabilities if the model supports it
    probs = (
        model.predict_proba(X_scaled)[:, 1]
        if hasattr(model, "predict_proba")
        else [None] * len(predictions)
    )

    # 4. Map results back to IDs
    results = []
    for idx, row in filtered_df.iterrows():
        results.append(
            {
                "playerId": row["playerID"],
                "result": (
                    "Hall of Famer" if predictions[len(results)] == 1 else "Missed"
                ),
                "probability": (
                    f"{probs[len(results)]:.2%}" if probs[0] is not None else "N/A"
                ),
            }
        )

    return {"predictions": results}


# {
#   "playerIds": [
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
