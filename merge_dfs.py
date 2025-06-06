import pandas as pd
from functools import reduce


def merge_dfs(dfs):
    return reduce(
        lambda left, right: pd.merge(left, right, on=["playerID"], how="outer"), dfs
    )
