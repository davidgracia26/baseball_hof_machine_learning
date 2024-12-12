import pandas as pd
import numpy as np
from batting_utils import create_grouped_batting_df
from pitching_utils import create_grouped_pitching_df
from master_utils import create_years_played_data
from awards_utils import create_grouped_award_df
from allstar_utility import create_grouped_all_star_df
from merge_dfs import merge_dfs
from ped_users_utils import create_grouped_ped_users_df


def get_df_for_modeling():
    batting_df = pd.read_csv("batting.csv")
    grouped_reg_batting_df = create_grouped_batting_df(batting_df)

    batting_post_df = pd.read_csv("battingpost.csv")
    grouped_post_batting_df = create_grouped_batting_df(batting_post_df, True)

    hof_df = pd.read_csv("halloffame.csv")
    grouped_hof_df = hof_df.loc[
        (hof_df["category"] == "Player") & (hof_df["inducted"] == "Y")
    ][["playerID", "inducted"]]
    grouped_hof_df["inducted"] = grouped_hof_df["inducted"].map({"Y": 1, "N": 0})
    rows_grouped_hof_df = len(grouped_hof_df)

    master_df = pd.read_csv("master.csv")[["playerID", "nameFirst", "nameLast"]]
    rows_master_df = len(master_df)

    minimal_necessary_accuracy = 1 - (rows_grouped_hof_df / rows_master_df)

    print(minimal_necessary_accuracy)

    pitching_df = pd.read_csv("pitching.csv")
    grouped_reg_pitching_df = create_grouped_pitching_df(pitching_df)

    pitching_post_df = pd.read_csv("pitchingpost.csv")
    grouped_post_pitching_df = create_grouped_pitching_df(pitching_post_df, True)

    allstar_df = pd.read_csv("allstarfull.csv")
    grouped_allstar_df = create_grouped_all_star_df(allstar_df)

    awards_players_df = pd.read_csv("awardsplayers.csv")
    grouped_awards_players_df = create_grouped_award_df(awards_players_df)

    ped_users_df = pd.read_csv("steroidusers.csv")
    grouped_ped_users = create_grouped_ped_users_df(ped_users_df)

    print(grouped_ped_users)

    # start joining dfs

    first_merge_df = pd.merge(
        grouped_allstar_df, grouped_reg_batting_df, on="playerID", how="outer"
    )
    second_merge_df = pd.merge(
        first_merge_df, grouped_post_batting_df, on="playerID", how="outer"
    )
    third_merge_df = pd.merge(
        second_merge_df, grouped_hof_df, on="playerID", how="outer"
    )
    fourth_merge_df = pd.merge(third_merge_df, master_df, on="playerID", how="outer")
    fifth_merge_df = pd.merge(
        fourth_merge_df, grouped_reg_pitching_df, on="playerID", how="outer"
    )
    sixth_merge_df = pd.merge(
        fifth_merge_df, grouped_post_pitching_df, on="playerID", how="outer"
    )
    seventh_merge_df = pd.merge(
        sixth_merge_df, grouped_awards_players_df, on="playerID", how="outer"
    )
    eigth_merge_df = pd.merge(
        seventh_merge_df, grouped_ped_users, on="playerID", how="outer"
    )

    eigth_merge_df = eigth_merge_df.fillna(0)
    eigth_merge_df = eigth_merge_df.replace([np.inf, -np.inf], 0)

    # best precision = 0.8
    # cols = ['ASGs', 'inducted', 'PitchingTripleCrowns', 'TripleCrowns', 'MVPs', 'CyYoungs', 'GoldGloves']
    cols = [
        "ASGs",
        "inducted",
        "PitchingTripleCrowns",
        "TripleCrowns",
        "MVPs",
        "CyYoungs",
        "GoldGloves",
        "PEDUser",
    ]
    # cols = ['ASGs', 'inducted', 'PitchingTripleCrowns', 'TripleCrowns', 'MVPs', 'CyYoungs', 'GoldGloves']

    modeling_df = eigth_merge_df[cols]

    return modeling_df
