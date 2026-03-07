import pandas as pd


class PitchingUtils:
    def __init__(self):
        pass

    def create_reg_season_grouped_pitching_df(self):
        df = pd.read_csv("source_data/pitching.csv")

        return self.create_grouped_pitching_df(df)

    def create_post_season_grouped_pitching_df(self):
        df = pd.read_csv("source_data/pitchingpost.csv")

        grouped_df = self.create_grouped_pitching_df(df)
        grouped_df = grouped_df.add_prefix("post_")
        grouped_df = grouped_df.rename(columns={"post_playerID": "playerID"})

        return grouped_df

    def create_grouped_pitching_df(self, df):
        grouped_df = (
            df[
                [
                    "playerID",
                    "W",
                    "L",
                    "G",
                    "GS",
                    "CG",
                    "SHO",
                    "SV",
                    "IPouts",
                    "H",
                    "ER",
                    "HR",
                    "BB",
                    "SO",
                    "BAOpp",
                    "IBB",
                    "WP",
                    "HBP",
                    "BK",
                    "BFP",
                    "GF",
                    "R",
                    "SH",
                    "SF",
                    "GIDP",
                ]
            ]
            .groupby(["playerID"], dropna=False, as_index=False)
            .sum()
        )
        grouped_df["ERA"] = 27 * (grouped_df["ER"] / grouped_df["IPouts"])
        grouped_df["WHIP"] = (
            3 * (grouped_df["BB"] + grouped_df["H"]) / grouped_df["IPouts"]
        )
        grouped_df["KPer9"] = 27 * (grouped_df["SO"] / grouped_df["IPouts"])
        grouped_df["KPerBB"] = grouped_df["SO"] / (grouped_df["BB"] + 1)
        grouped_df["HPer9"] = 27 * (grouped_df["H"] / grouped_df["IPouts"])
        grouped_df["BBPer9"] = 27 * (grouped_df["BB"] / grouped_df["IPouts"])
        grouped_df["FIP"] = (
            13 * grouped_df["HR"]
            + 3 * (grouped_df["BB"] + grouped_df["HBP"])
            - 2 * grouped_df["SO"]
        ) / (grouped_df["IPouts"] / 3) + 3.1

        # hitting stats against
        grouped_df["AB"] = (
            grouped_df["BFP"]
            - grouped_df["BB"]
            - grouped_df["HBP"]
            - grouped_df["SH"]
            - grouped_df["SF"]
        )

        grouped_df["OBP"] = (grouped_df["H"] + grouped_df["BB"] + grouped_df["HBP"]) / (
            grouped_df["BFP"]
        )
        grouped_df["AVG"] = grouped_df["H"] / grouped_df["AB"]
        grouped_df["BAbip"] = (grouped_df["H"] - grouped_df["HR"]) / (
            grouped_df["AB"] - grouped_df["SO"] - grouped_df["HR"] + grouped_df["SF"]
        )
        grouped_df["Pitcher"] = 1
        grouped_df = grouped_df.fillna(0)

        grouped_df = grouped_df.add_prefix("pitching_")
        grouped_df = grouped_df.rename(columns={"pitching_playerID": "playerID"})

        return grouped_df
