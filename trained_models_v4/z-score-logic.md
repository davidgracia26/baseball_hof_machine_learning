```python
import pandas as pd
import numpy as np

# 1. Identify the stats you want to analyze
stat_cols = ['G', 'AB', 'R', 'H', '2B', '3B', 'HR', 'RBI', 'SB', 'BB', 'SO']

# 2. Calculate Z-Scores per Season (Era Normalization)
# We group by yearID and apply a lambda to calculate: (x - mean) / std
def calculate_zscore(x):
    # Handle cases where std is 0 to avoid division by zero
    if x.std() == 0:
        return 0
    return (x - x.mean()) / x.std()

# Create a copy to store z-scores
df_zscores = df[['playerID', 'yearID'] + stat_cols].copy()

for col in stat_cols:
    df_zscores[f'{col}_z'] = df.groupby('yearID')[col].transform(calculate_zscore)

# 3. Accumulate Career Z-Scores
# Group by playerID and sum the seasonal z-scores
career_z_columns = [f'{col}_z' for col in stat_cols]
career_z_scores = df_zscores.groupby('playerID')[career_z_columns].sum().reset_index()

# 4. (Optional) Rename for clarity
career_z_scores.columns = ['playerID'] + [f'career_{col}_zsum' for col in stat_cols]

print(career_z_scores.head())
```