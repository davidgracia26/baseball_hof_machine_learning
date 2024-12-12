def create_grouped_pitching_df(df, isPostSeason=False):
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

    feature_cols = ["playerID", "H", "SO"]
    grouped_df = grouped_df[feature_cols]

    if isPostSeason:
        grouped_df = grouped_df.add_prefix("post_")
        grouped_df = grouped_df.rename(columns={"post_playerID": "playerID"})

    return grouped_df
