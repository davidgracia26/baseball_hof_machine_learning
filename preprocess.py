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


class Preprocess:
    def __init__(self):
        self.hall_of_fame_utils = HallOfFameUtils()
        self.batting_utils = BattingUtils()
        self.pitching_utils = PitchingUtils()
        self.master_utils = MasterUtils()
        self.allstar_utils = AllStarUtils()
        self.awards_utils = AwardsUtils()
        self.ped_users_utils = PedUsersUtils()

    def get_df_for_modeling(self):
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
            grouped_reg_season_batting_df,
            grouped_post_season_batting_df,
            grouped_hall_of_fame_df,
            master_df,
            # grouped_reg_season_pitching_df,
            # grouped_post_season_pitching_df,
            grouped_allstar_df,
            grouped_awards_players_df,
            grouped_ped_users,
        ]

        merged_df = merge_dfs(dfs_to_merge)

        merged_df = merged_df.fillna(0)
        merged_df = merged_df.replace([np.inf, -np.inf], 0)

        return merged_df
