import pandas as pd

def create_years_played_data(batting_df, pitching_df):
    batting_test_df = batting_df[['playerID', 'yearID']].sort_values(by=['playerID', 'yearID'])
    
    pitching_test_df = pitching_df[['playerID', 'yearID']].sort_values(by=['playerID', 'yearID'])
    
    merged_df = pd.concat([batting_test_df, pitching_test_df]).sort_values(by=['playerID', 'yearID'])
    
    merged_df = merged_df.groupby(['playerID', 'yearID'], dropna=False, as_index=False).size()
    
    merged_df = merged_df.groupby(['playerID', 'yearID'], dropna=False, as_index=False).size()
    
    merged_df = merged_df.groupby(['playerID'], dropna=False, as_index=False).size()
    
    return merged_df.rename(columns={"size": "seasons"})