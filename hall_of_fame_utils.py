import pandas as pd


class HallOfFameUtils:
    def __init__(self):
        pass

    def create_grouped_hall_of_fame_df(self):
        df = pd.read_csv("halloffame.csv")
        grouped_hof_df = df.loc[(df["category"] == "Player") & (df["inducted"] == "Y")][
            ["playerID", "inducted"]
        ]
        grouped_hof_df["inducted"] = grouped_hof_df["inducted"].map({"Y": 1, "N": 0})

        return grouped_hof_df
