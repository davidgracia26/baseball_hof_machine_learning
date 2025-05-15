import pandas as pd
import numpy as np
from batting_utils import (
    create_reg_season_grouped_batting_df,
    create_post_season_grouped_batting_df,
)
from pitching_utils import (
    create_reg_season_grouped_pitching_df,
    create_post_season_grouped_pitching_df,
)
from master_utils import create_master_data
from awards_utils import create_grouped_award_df
from allstar_utils import create_grouped_all_star_df
from merge_dfs import merge_dfs
from ped_users_utils import create_grouped_ped_users_df
from hall_of_fame_utils import create_grouped_hall_of_fame_df
from merge_dfs import merge_dfs


def get_df_for_modeling():
    grouped_reg_season_batting_df = create_reg_season_grouped_batting_df()
    grouped_post_season_batting_df = create_post_season_grouped_batting_df()

    grouped_hall_of_fame_df = create_grouped_hall_of_fame_df()

    master_df = create_master_data()

    grouped_reg_season_pitching_df = create_reg_season_grouped_pitching_df()
    grouped_post_season_pitching_df = create_post_season_grouped_pitching_df()

    grouped_allstar_df = create_grouped_all_star_df()

    grouped_awards_players_df = create_grouped_award_df()

    grouped_ped_users = create_grouped_ped_users_df()

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
