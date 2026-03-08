import pandas as pd
import numpy as np
from batting_utils import BattingUtils
from pitching_utils import PitchingUtils
from master_utils import MasterUtils
from awards_utils import AwardsUtils
from allstar_utils import AllStarUtils
from merge_dfs import merge_dfs
from ped_users_utils import PedUsersUtils
from hall_of_fame_utils import HallOfFameUtils
from merge_dfs import merge_dfs


def reg_season_batting_z_score_df():
    df = pd.read_csv("source_data/batting.csv")
    # 1. Identify the stats you want to analyze
    stat_cols = ["G", "AB", "R", "H", "2B", "3B", "HR", "RBI", "SB", "BB", "SO"]

    # 2. Calculate Z-Scores per Season (Era Normalization)
    # We group by yearID and apply a lambda to calculate: (x - mean) / std
    def calculate_zscore(x):
        # Handle cases where std is 0 to avoid division by zero
        if x.std() == 0:
            return 0
        return (x - x.mean()) / x.std()

    # Create a copy to store z-scores
    df_zscores = df[["playerID", "yearID"] + stat_cols].copy()

    for col in stat_cols:
        df_zscores[f"{col}_z"] = df.groupby("yearID")[col].transform(calculate_zscore)

    # 3. Accumulate Career Z-Scores
    # Group by playerID and sum the seasonal z-scores
    career_z_columns = [f"{col}_z" for col in stat_cols]
    career_z_scores = (
        df_zscores.groupby("playerID")[career_z_columns].sum().reset_index()
    )

    # 4. (Optional) Rename for clarity
    career_z_scores.columns = ["playerID"] + [f"career_{col}_zsum" for col in stat_cols]

    print(career_z_scores.head())

    career_z_scores.to_csv("hitter_z_scores.csv", index=False)


t_method()


class Preprocess:
    def __init__(self):
        self.hall_of_fame_utils = HallOfFameUtils()
        self.batting_utils = BattingUtils()
        self.pitching_utils = PitchingUtils()
        self.master_utils = MasterUtils()
        self.allstar_utils = AllStarUtils()
        self.awards_utils = AwardsUtils()
        self.ped_users_utils = PedUsersUtils()

    def get_df_for_modeling_hitters(self):
        grouped_reg_season_batting_df = (
            self.batting_utils.create_reg_season_grouped_batting_df()
        )
        grouped_post_season_batting_df = (
            self.batting_utils.create_post_season_grouped_batting_df()
        )

        grouped_hall_of_fame_df = (
            self.hall_of_fame_utils.create_grouped_hall_of_fame_df()
        )

        master_df = self.master_utils.create_master_data()

        grouped_allstar_df = self.allstar_utils.create_grouped_all_star_df()

        grouped_awards_players_df = self.awards_utils.create_grouped_award_df()

        grouped_ped_users = self.ped_users_utils.create_grouped_ped_users_df()

        dfs_to_merge = [
            grouped_reg_season_batting_df,
            grouped_post_season_batting_df,
            grouped_hall_of_fame_df,
            master_df,
            grouped_allstar_df,
            grouped_awards_players_df,
            grouped_ped_users,
        ]

        merged_df = merge_dfs(dfs_to_merge)

        merged_df = merged_df.fillna(0)
        merged_df = merged_df.replace([np.inf, -np.inf], 0)

        return merged_df

    def get_df_for_modeling_pitchers(self):
        grouped_hall_of_fame_df = (
            self.hall_of_fame_utils.create_grouped_hall_of_fame_df()
        )

        master_df = self.master_utils.create_master_data()

        grouped_reg_season_pitching_df = (
            self.pitching_utils.create_reg_season_grouped_pitching_df()
        )
        grouped_post_season_pitching_df = (
            self.pitching_utils.create_post_season_grouped_pitching_df()
        )

        grouped_allstar_df = self.allstar_utils.create_grouped_all_star_df()

        grouped_awards_players_df = self.awards_utils.create_grouped_award_df()

        grouped_ped_users = self.ped_users_utils.create_grouped_ped_users_df()

        dfs_to_merge = [
            grouped_hall_of_fame_df,
            master_df,
            grouped_reg_season_pitching_df,
            grouped_post_season_pitching_df,
            grouped_allstar_df,
            grouped_awards_players_df,
            grouped_ped_users,
        ]

        merged_df = merge_dfs(dfs_to_merge)

        merged_df = merged_df.fillna(0)
        merged_df = merged_df.replace([np.inf, -np.inf], 0)

        return merged_df
