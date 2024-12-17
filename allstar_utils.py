import pandas as pd


def create_grouped_all_star_df():
    df = pd.read_csv("allstarfull.csv")

    grouped_all_star_df = (
        df[["playerID", "gameNum"]]
        .groupby(["playerID"], dropna=False, as_index=False)
        .sum()
    )

    grouped_all_star_df = grouped_all_star_df.rename(columns={"gameNum": "ASGs"})

    return grouped_all_star_df
