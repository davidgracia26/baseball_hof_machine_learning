# https://en.wikipedia.org/wiki/List_of_Major_League_Baseball_players_suspended_for_performance-enhancing_drugs
# https://en.wikipedia.org/wiki/List_of_Major_League_Baseball_players_named_in_the_Mitchell_Report
# https://www.espn.com/mlb/news/story?id=4366683

import pandas as pd


def create_grouped_ped_users_df(df):
    unique_players = df["playerID"].unique()

    data = {"playerID": unique_players, "PEDUser": 1}

    return pd.DataFrame(data)
