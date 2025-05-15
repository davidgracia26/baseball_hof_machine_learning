import pandas as pd
import numpy as np
from merge_dfs import merge_dfs


class AwardsUtils:
    def __init__(self):
        pass

    def create_grouped_award_category_df(self, df, filter_column, award_column):
        filter = df["awardID"] == filter_column
        filtered_df = df[filter].copy()
        filtered_df.loc[:, award_column] = 1
        grouped_filtered_df = (
            filtered_df[["playerID", award_column]]
            .groupby(["playerID"], dropna=False, as_index=False)
            .sum()
        )

        return grouped_filtered_df

    def create_grouped_award_df(self):
        df = pd.read_csv("source_data/awardsplayers.csv")

        grouped_pitching_triple_crown_df = self.create_grouped_award_category_df(
            df, "Pitching Triple Crown", "PitchingTripleCrowns"
        )

        grouped_triple_crown_df = self.create_grouped_award_category_df(
            df, "Triple Crown", "TripleCrowns"
        )

        grouped_mvp_df = self.create_grouped_award_category_df(
            df, "Most Valuable Player", "MVPs"
        )

        grouped_roy_df = self.create_grouped_award_category_df(
            df, "Rookie of the Year", "ROYs"
        )

        grouped_ws_mvp_df = self.create_grouped_award_category_df(
            df, "World Series MVP", "WSMVPs"
        )

        grouped_cy_young_df = self.create_grouped_award_category_df(
            df, "Cy Young Award", "CyYoungs"
        )

        grouped_gold_glove_df = self.create_grouped_award_category_df(
            df, "Gold Glove", "GoldGloves"
        )

        grouped_all_star_mvp_df = self.create_grouped_award_category_df(
            df, "All-Star Game MVP", "ASGMVPs"
        )

        grouped_rolaids_relief_df = self.create_grouped_award_category_df(
            df, "Rolaids Relief Man Award", "RolaidsReliefManAwards"
        )

        grouped_nlcs_mvp_df = self.create_grouped_award_category_df(
            df, "NLCS MVP", "NLCSMVPs"
        )

        grouped_alcs_mvp_df = self.create_grouped_award_category_df(
            df, "ALCS MVP", "ALCSMVPs"
        )

        grouped_silver_slugger_df = self.create_grouped_award_category_df(
            df, "Silver Slugger", "SilverSluggers"
        )

        # merge these

        dfs_to_merge = [
            grouped_pitching_triple_crown_df,
            grouped_triple_crown_df,
            grouped_mvp_df,
            grouped_roy_df,
            grouped_ws_mvp_df,
            grouped_cy_young_df,
            grouped_gold_glove_df,
            grouped_all_star_mvp_df,
            grouped_rolaids_relief_df,
            grouped_nlcs_mvp_df,
            grouped_alcs_mvp_df,
            grouped_silver_slugger_df,
        ]

        merged_df = merge_dfs(dfs_to_merge)

        merged_df = merged_df.fillna(0)
        merged_df = merged_df.replace([np.inf, -np.inf], 0)

        return merged_df
