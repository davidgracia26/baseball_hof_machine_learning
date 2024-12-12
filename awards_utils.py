import pandas as pd
import numpy as np


def create_grouped_award_category_df(df, filter_column, award_column):
    filter = df["awardID"] == filter_column
    filtered_df = df[filter].copy()
    filtered_df.loc[:, award_column] = 1
    grouped_filtered_df = (
        filtered_df[["playerID", award_column]]
        .groupby(["playerID"], dropna=False, as_index=False)
        .sum()
    )

    return grouped_filtered_df


def create_grouped_award_df(df):
    grouped_pitching_triple_crown_df = create_grouped_award_category_df(
        df, "Pitching Triple Crown", "PitchingTripleCrowns"
    )

    grouped_triple_crown_df = create_grouped_award_category_df(
        df, "Triple Crown", "TripleCrowns"
    )

    grouped_mvp_df = create_grouped_award_category_df(
        df, "Most Valuable Player", "MVPs"
    )

    grouped_roy_df = create_grouped_award_category_df(df, "Rookie of the Year", "ROYs")

    grouped_ws_mvp_df = create_grouped_award_category_df(
        df, "World Series MVP", "WSMVPs"
    )

    grouped_cy_young_df = create_grouped_award_category_df(
        df, "Cy Young Award", "CyYoungs"
    )

    grouped_gold_glove_df = create_grouped_award_category_df(
        df, "Gold Glove", "GoldGloves"
    )

    grouped_all_star_mvp_df = create_grouped_award_category_df(
        df, "All-Star Game MVP", "ASGMVPs"
    )

    grouped_rolaids_relief_df = create_grouped_award_category_df(
        df, "Rolaids Relief Man Award", "RolaidsReliefManAwards"
    )

    grouped_nlcs_mvp_df = create_grouped_award_category_df(df, "NLCS MVP", "NLCSMVPs")

    grouped_alcs_mvp_df = create_grouped_award_category_df(df, "ALCS MVP", "ALCSMVPs")

    grouped_silver_slugger_df = create_grouped_award_category_df(
        df, "Silver Slugger", "SilverSluggers"
    )

    # merge these

    int1_df = pd.merge(
        grouped_pitching_triple_crown_df,
        grouped_triple_crown_df,
        on="playerID",
        how="outer",
    )
    int2_df = pd.merge(int1_df, grouped_mvp_df, on="playerID", how="outer")
    int3_df = pd.merge(int2_df, grouped_roy_df, on="playerID", how="outer")
    int4_df = pd.merge(int3_df, grouped_ws_mvp_df, on="playerID", how="outer")
    int5_df = pd.merge(int4_df, grouped_cy_young_df, on="playerID", how="outer")
    int6_df = pd.merge(int5_df, grouped_gold_glove_df, on="playerID", how="outer")
    int7_df = pd.merge(int6_df, grouped_all_star_mvp_df, on="playerID", how="outer")
    int8_df = pd.merge(int7_df, grouped_rolaids_relief_df, on="playerID", how="outer")
    int9_df = pd.merge(int8_df, grouped_nlcs_mvp_df, on="playerID", how="outer")
    int10_df = pd.merge(int9_df, grouped_alcs_mvp_df, on="playerID", how="outer")
    int11_df = pd.merge(int10_df, grouped_silver_slugger_df, on="playerID", how="outer")

    int11_df = int11_df.fillna(0)
    int11_df = int11_df.replace([np.inf, -np.inf], 0)

    return int11_df
