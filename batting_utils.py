import pandas as pd


def create_reg_season_grouped_batting_df():
    df = pd.read_csv("batting.csv")
    return create_grouped_batting_df(df)


def create_post_season_grouped_batting_df():
    df = pd.read_csv("battingpost.csv")

    grouped_df = create_grouped_batting_df(df)
    grouped_df = grouped_df.add_prefix("post_")
    grouped_df = grouped_df.rename(columns={"post_playerID": "playerID"})

    return grouped_df


def create_grouped_batting_df(df):

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
    grouped_df = grouped_df.fillna(0)

    grouped_df = grouped_df.add_prefix("batting_")
    grouped_df = grouped_df.rename(columns={"batting_playerID": "playerID"})

    return grouped_df
