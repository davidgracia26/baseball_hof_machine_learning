import pandas as pd
import numpy as np
import pprint
from trained_models_v4.merge_dfs import merge_dfs

master_df = pd.read_csv("source_data/master.csv")
df = pd.read_csv("source_data/pitching.csv")

pitching_career = df.groupby("playerID").agg(
    {
        "yearID": "nunique",
        "G": "sum",
        "GS": "sum",
        "SV": "sum",
        "GF": "sum",
        "IPouts": "sum",
    }
)

pitching_career = pitching_career.rename(columns={"yearID": "years_played"})

pitching_career["IP"] = np.floor(pitching_career["IPouts"] / 3 * 100) / 100

pitching_career["GS_Ratio"] = np.where(
    pitching_career["G"] > 0, pitching_career["GS"] / pitching_career["G"], 0
)

pitching_career["GF_Ratio"] = np.where(
    pitching_career["G"] > 0, pitching_career["GF"] / pitching_career["G"], 0
)

pprint.pprint(pitching_career)

ordered_pitchers = pitching_career.sort_values(
    by=["GS_Ratio", "IPouts"], ascending=[False, False]
)


dfs = [master_df, ordered_pitchers]

merged_df = merge_dfs(dfs)

merged_df.to_csv("ordered_pitchers.csv")
