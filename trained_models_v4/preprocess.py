import pandas as pd
import numpy as np
from trained_models_v4.batting_utils import BattingUtils
from trained_models_v4.pitching_utils import PitchingUtils
from trained_models_v4.master_utils import MasterUtils
from trained_models_v4.awards_utils import AwardsUtils
from trained_models_v4.allstar_utils import AllStarUtils
from trained_models_v4.merge_dfs import merge_dfs
from trained_models_v4.ped_users_utils import PedUsersUtils
from trained_models_v4.hall_of_fame_utils import HallOfFameUtils
from trained_models_v4.merge_dfs import merge_dfs


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
