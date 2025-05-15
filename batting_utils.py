import pandas as pd


class BattingUtils:
    def __init__(self):
        pass

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
        grouped_df = (
            df[
                [
                    "playerID",
                    "AB",
                    "R",
                    "H",
                    "2B",
                    "3B",
                    "HR",
                    "RBI",
                    "SB",
                    "CS",
                    "BB",
                    "SO",
                    "IBB",
                    "HBP",
                    "SH",
                    "SF",
                    "GIDP",
                ]
            ]
            .groupby(["playerID"], dropna=False, as_index=False)
            .sum()
        )
        grouped_df["1B"] = grouped_df["H"] - (
            grouped_df["HR"] + grouped_df["3B"] + grouped_df["2B"]
        )
        grouped_df["TB"] = (
            grouped_df["1B"]
            + 2 * grouped_df["2B"]
            + 3 * grouped_df["3B"]
            + 4 * grouped_df["HR"]
        )
        grouped_df["OBP"] = (grouped_df["H"] + grouped_df["BB"] + grouped_df["HBP"]) / (
            grouped_df["AB"] + grouped_df["BB"] + grouped_df["HBP"] + grouped_df["SF"]
        )
        grouped_df["SLG"] = grouped_df["TB"] / grouped_df["AB"]
        grouped_df["OPS"] = grouped_df["OBP"] + grouped_df["SLG"]
        grouped_df["SBP"] = grouped_df["SB"] / (grouped_df["SB"] + grouped_df["CS"])
        grouped_df["Batter"] = 1
        grouped_df["BAbip"] = (grouped_df["H"] - grouped_df["HR"]) / (
            grouped_df["AB"] - grouped_df["SO"] - grouped_df["HR"] + grouped_df["SF"]
        )
        grouped_df["ISO"] = (grouped_df["TB"] - grouped_df["H"]) / grouped_df["AB"]
        grouped_df["PowerSpeedNumber"] = (
            2
            * (grouped_df["HR"] * grouped_df["SB"])
            / (grouped_df["HR"] + grouped_df["SB"])
        )
        grouped_df["ABPerHR"] = grouped_df["AB"] / grouped_df["HR"]
        grouped_df["ABPerK"] = grouped_df["AB"] / grouped_df["SO"]
        grouped_df = grouped_df.fillna(0)

        grouped_df = grouped_df.add_prefix("batting_")
        grouped_df = grouped_df.rename(columns={"batting_playerID": "playerID"})

        return grouped_df
