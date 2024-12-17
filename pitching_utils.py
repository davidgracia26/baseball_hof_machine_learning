import pandas as pd


def create_reg_season_grouped_pitching_df():
    df = pd.read_csv("pitching.csv")

    return create_grouped_pitching_df(df)


def create_post_season_grouped_pitching_df():
    df = pd.read_csv("pitchingpost.csv")

    grouped_df = create_grouped_pitching_df(df)
    grouped_df = grouped_df.add_prefix("post_")
    grouped_df = grouped_df.rename(columns={"post_playerID": "playerID"})

    return grouped_df


def create_grouped_pitching_df(df):
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
    grouped_df["WHIP"] = 3 * (grouped_df["BB"] + grouped_df["H"]) / grouped_df["IPouts"]
    grouped_df["KPer9"] = 27 * (grouped_df["SO"] / grouped_df["IPouts"])
    grouped_df = grouped_df.fillna(0)

    grouped_df = grouped_df.add_prefix("pitching_")
    grouped_df = grouped_df.rename(columns={"pitching_playerID": "playerID"})

    return grouped_df
