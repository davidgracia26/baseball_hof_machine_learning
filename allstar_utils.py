import pandas as pd


class AllStarUtils:
    def __init__(self):
        pass

    def create_grouped_all_star_df(self):
        df = pd.read_csv("source_data/allstarfull.csv")

        grouped_all_star_df = (
            df[["playerID", "gameNum"]]
            .groupby(["playerID"], dropna=False, as_index=False)
            .sum()
        )

        grouped_all_star_df = grouped_all_star_df.rename(columns={"gameNum": "ASGs"})

        return grouped_all_star_df
