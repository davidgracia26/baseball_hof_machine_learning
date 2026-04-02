import numpy as np
import pandas as pd

from trained_models_v4.merge_dfs import merge_dfs


class BattingUtils:
    def __init__(self):
        pass

    # 2. Calculate Z-Scores per Season (Era Normalization)
    # We group by yearID and apply a lambda to calculate: (x - mean) / std
    def calculate_zscore(self, x):
        # Handle cases where std is 0 to avoid division by zero
        if x.std() == 0:
            return 0
        return (x - x.mean()) / x.std()

    def reg_season_batting_z_score_df(self, df):
        # 1. Identify the stats you want to analyze
        # stat_cols = ["G", "AB", "R", "H", "2B", "3B", "HR", "RBI", "SB", "BB", "SO"]

        df.fillna(0, inplace=True)

        df["1B"] = df["H"] - (df["HR"] + df["3B"] + df["2B"])
        df["TB"] = df["1B"] + 2 * df["2B"] + 3 * df["3B"] + 4 * df["HR"]
        df["PA"] = df["AB"] + df["BB"] + df["HBP"] + df["SF"]

        df["OBP"] = (df["H"] + df["BB"] + df["HBP"]) / (df["PA"])

        print(df.columns)

        df["AVG"] = df["H"] / df["AB"]
        df["SLG"] = df["TB"] / df["AB"]
        df["OPS"] = df["OBP"] + df["SLG"]
        df["SBP"] = df["SB"] / (df["SB"] + df["CS"])
        df["BAbip"] = (df["H"] - df["HR"]) / (df["AB"] - df["SO"] - df["HR"] + df["SF"])
        df["ISO"] = (df["TB"] - df["H"]) / df["AB"]
        df["PowerSpeedNumber"] = 2 * (df["HR"] * df["SB"]) / (df["HR"] + df["SB"])
        df["ABPerHR"] = df["AB"] / df["HR"]
        df["ABPerK"] = df["AB"] / df["SO"]

        excluded_cols = ["playerID", "yearID", "stint", "teamID", "lgID", "round"]
        stat_cols = [col for col in df.columns if col not in excluded_cols]

        df = df.replace([np.inf, -np.inf], np.nan).fillna(0)

        # Create a copy to store z-scores
        df_zscores = df[["playerID", "yearID"] + stat_cols].copy()

        for col in stat_cols:
            df_zscores[f"{col}_z"] = df.groupby("yearID")[col].transform(
                self.calculate_zscore
            )

        # 3. Accumulate Career Z-Scores
        # Group by playerID and sum the seasonal z-scores
        # career_z_columns = [f"{col}_z" for col in stat_cols]
        career_z_scores = df_zscores.groupby("playerID")[stat_cols].sum().reset_index()

        # 4. (Optional) Rename for clarity
        career_z_scores.columns = ["playerID"] + [f"batting_{col}" for col in stat_cols]

        # print(career_z_scores.head())

        career_z_scores.to_csv("hitter_z_scores.csv", index=False)

        return career_z_scores

    def create_reg_season_grouped_batting_df(self):
        df = pd.read_csv("source_data/batting.csv")
        return self.create_grouped_batting_df(df)

    def create_post_season_grouped_batting_df(self):
        df = pd.read_csv("source_data/battingpost.csv")

        grouped_df = self.create_grouped_batting_df(df)
        grouped_df = grouped_df.add_prefix("post_")
        grouped_df = grouped_df.rename(columns={"post_playerID": "playerID"})

        return grouped_df

    def create_grouped_batting_df(self, df):
        return self.reg_season_batting_z_score_df(df)

        # grouped_df = (
        #     df[
        # [
        #     "playerID",
        #     "AB",
        #     "R",
        #     "H",
        #     "2B",
        #     "3B",
        #     "HR",
        #     "RBI",
        #     "SB",
        #     "CS",
        #     "BB",
        #     "SO",
        #     "IBB",
        #     "HBP",
        #     "SH",
        #     "SF",
        #     "GIDP",
        # ]
        #     ]
        #     .groupby(["playerID"], dropna=False, as_index=False)
        #     .sum()
        # )
        # grouped_df["1B"] = grouped_df["H"] - (
        #     grouped_df["HR"] + grouped_df["3B"] + grouped_df["2B"]
        # )
        # grouped_df["TB"] = (
        #     grouped_df["1B"]
        #     + 2 * grouped_df["2B"]
        #     + 3 * grouped_df["3B"]
        #     + 4 * grouped_df["HR"]
        # )
        # grouped_df["PA"] = (
        #     grouped_df["AB"] + grouped_df["BB"] + grouped_df["HBP"] + grouped_df["SF"]
        # )
        # grouped_df["OBP"] = (grouped_df["H"] + grouped_df["BB"] + grouped_df["HBP"]) / (
        #     grouped_df["PA"]
        # )
        # grouped_df["AVG"] = grouped_df["H"] / grouped_df["AB"]
        # grouped_df["SLG"] = grouped_df["TB"] / grouped_df["AB"]
        # grouped_df["OPS"] = grouped_df["OBP"] + grouped_df["SLG"]
        # grouped_df["SBP"] = grouped_df["SB"] / (grouped_df["SB"] + grouped_df["CS"])
        # grouped_df["Batter"] = 1
        # grouped_df["BAbip"] = (grouped_df["H"] - grouped_df["HR"]) / (
        #     grouped_df["AB"] - grouped_df["SO"] - grouped_df["HR"] + grouped_df["SF"]
        # )
        # grouped_df["ISO"] = (grouped_df["TB"] - grouped_df["H"]) / grouped_df["AB"]
        # grouped_df["PowerSpeedNumber"] = (
        #     2
        #     * (grouped_df["HR"] * grouped_df["SB"])
        #     / (grouped_df["HR"] + grouped_df["SB"])
        # )
        # grouped_df["ABPerHR"] = grouped_df["AB"] / grouped_df["HR"]
        # grouped_df["ABPerK"] = grouped_df["AB"] / grouped_df["SO"]
        # grouped_df = grouped_df.fillna(0)

        # grouped_df = grouped_df.add_prefix("batting_")
        # grouped_df = grouped_df.rename(columns={"batting_playerID": "playerID"})

        # return grouped_df
