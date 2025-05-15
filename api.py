from fastapi import FastAPI
import pandas as pd
from pydantic import BaseModel
import pickle
from preprocess import Preprocess


class SearchPlayersRequest(BaseModel):
    firstName: str
    lastName: str


class PredictHallOfFameRequest(BaseModel):
    playerId: str


def create_players(request: SearchPlayersRequest):
    df = pd.read_csv("master.csv")

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


# need to be able to pass the player name after selected
# need to be able to pass playerID
@app.post("/predict_hof")
def predict_hof(request: PredictHallOfFameRequest):
    df = Preprocess().get_df_for_modeling()

    filtered_df = df[(df["playerID"].str.lower() == request.playerId.lower())]

    if len(filtered_df) == 0:
        return "Player not in data"

    feature_cols = [
        "batting_3B",
        "MVPs",
        "pitching_SHO",
        "post_batting_1B",
        "post_pitching_CG",
    ]

    X = filtered_df[feature_cols]

    with open("Linear SVC.pkl", "rb") as f:
        clf2 = pickle.load(f)

    prediction = clf2.predict(X).tolist()
    is_hall_of_famer = prediction[0]
    return "Hall of Famer" if is_hall_of_famer else "Missed"
