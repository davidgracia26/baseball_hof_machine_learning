from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import OneClassSVM
from sklearn.model_selection import train_test_split
from sklearn import metrics
from preprocess import Preprocess
from sklearn.preprocessing import StandardScaler
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import mutual_info_classif
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFE, RFECV
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import Lasso
import numpy as np
import pprint

# pitching_Pitcher
# post_pitching_Pitcher
# batting_Batter
# post_batting_Batter

preprocess = Preprocess()
df = preprocess.get_df_for_modeling()
# df = df.loc[(df["pitching_Pitcher"] == 1) | (df["post_pitching_Pitcher"] == 1)]
# df = df.loc[(df["batting_Batter"] == 1) | (df["post_batting_Batter"] == 1)]

# print("df.corr()")
# print(df.corr())

# print(df)

# best precision = 0.8
# cols = ['ASGs', 'inducted', 'PitchingTripleCrowns', 'TripleCrowns', 'MVPs', 'CyYoungs', 'GoldGloves']
all_cols = list(df.columns.values)

pprint.pprint(all_cols)


features_for_heatmap = [
    x
    for x in all_cols
    if x
    in [
        "batting_R",
        "batting_OPS",
        "batting_PowerSpeedNumber",
        "post_batting_OPS",
        "post_batting_PowerSpeedNumber",
        "ASGs",
        "TripleCrowns",
        "MVPs",
        "ROYs",
        "WSMVPs",
        "GoldGloves",
        "ASGMVPs",
        "NLCSMVPs",
        "ALCSMVPs",
        "SilverSluggers",
        "PEDUser",
    ]
    # in [
    #     # "batting_OBP",
    #     # "batting_SLG",
    #     "batting_OPS",
    #     "batting_SBP",
    #     "batting_BAbip",
    #     "batting_ISO",
    #     "batting_PowerSpeedNumber",
    #     "batting_ABPerHR",
    #     "batting_ABPerK",
    #     # "post_batting_OBP",
    #     # "post_batting_SLG",
    #     "post_batting_OPS",
    #     "post_batting_SBP",
    #     "post_batting_BAbip",
    #     "post_batting_ISO",
    #     "post_batting_PowerSpeedNumber",
    #     "post_batting_ABPerHR",
    #     "post_batting_ABPerK",
    #     "ASGs",
    #     "TripleCrowns",
    #     "MVPs",
    #     "ROYs",
    #     "WSMVPs",
    #     "GoldGloves",
    #     "ASGMVPs",
    #     "NLCSMVPs",
    #     "ALCSMVPs",
    #     "SilverSluggers",
    #     "PEDUser",
    # ]
    # not in [
    #     "playerID",
    #     "nameFirst",
    #     "nameLast",
    #     "pitching_BAOpp",
    #     "post_pitching_BAOpp",
    #     "inducted",
    #     "pitching_Pitcher",
    #     "post_pitching_Pitcher",
    #     "batting_Batter",
    #     "post_batting_Batter",
    #     "PitchingTripleCrowns",
    #     "CyYoungs",
    #     "RolaidsReliefManAwards",
    #     #
    #     # "batting_AB",
    #     # "post_batting_AB",
    # ]
]


# for x in all_cols:
#     result = df[[x]].isin(["0.160.2500.52"]).sum()
#     print(x, result)
# abbotpa01
# -----------------------------------------------------------------------------
# corr = df[features_for_heatmap].corr()["inducted"].sort_values(ascending=False)
# print(corr.to_string())
# -----------------------------------------------------------------------------

vt = VarianceThreshold(threshold=0.2)
X_train_vt = vt.fit_transform(df[features_for_heatmap])
pprint.pprint("vt:")
pprint.pprint(np.array(features_for_heatmap)[vt.get_support()])

# derived_features = list(np.array(features_for_heatmap)[vt.get_support()])
# -----------------------------------------------------------------------------
# pprint.pprint(df[features_for_heatmap])
# -----------------------------------------------------------------------------

# new_features = SelectKBest(f_classif, k=10).fit(
#     df[features_for_heatmap], df["inducted"]
# )

# pprint.pprint(new_features.get_feature_names_out())
# pprint.pprint(new_features.get_support())

skb_mi = SelectKBest(score_func=mutual_info_classif, k=10)
X_train_mi = skb_mi.fit_transform(df[features_for_heatmap], df["inducted"])
pprint.pprint("skb_mi:")
pprint.pprint(np.array(features_for_heatmap)[skb_mi.get_support()])

# derived_features = list(np.array(features_for_heatmap)[skb_mi.get_support()])

skb_anova = SelectKBest(score_func=f_classif, k=10)
X_train_anova = skb_anova.fit_transform(df[features_for_heatmap], df["inducted"])
pprint.pprint("skb_anova:")
pprint.pprint(np.array(features_for_heatmap)[skb_anova.get_support()])

# derived_features = list(np.array(features_for_heatmap)[skb_anova.get_support()])

# should i bet using Logistic Regression or GradientBoostClassifier
# model = LogisticRegression(max_iter=1000)
# selector = RFE(model, n_features_to_select=10)
# X_new = selector.fit(
#     StandardScaler().fit_transform(df[features_for_heatmap]), df["inducted"]
# ).get_feature_names_out()

# pprint.pprint(X_new)

# model = GradientBoostingClassifier(n_estimators=25)
# selector = RFE(model, n_features_to_select=10)
# selector.fit(StandardScaler().fit_transform(df[features_for_heatmap]), df["inducted"])

# model = xgb.XGBClassifier(n_estimators=25)
# cv = StratifiedKFold(n_splits=5)
# rfecv = RFECV(estimator=model, step=1, cv=cv, scoring="precision")
# rfecv.fit_transform(df[features_for_heatmap], df["inducted"])

# pprint.pprint("rfecv:")

# derived_features = list(np.array(features_for_heatmap)[rfecv.get_support()])

model = xgb.XGBClassifier(n_estimators=25)
cv = StratifiedKFold(n_splits=5)
rfe = RFE(model, n_features_to_select=10)
rfe.fit_transform(df[features_for_heatmap], df["inducted"])

pprint.pprint("rfe:")

derived_features = list(np.array(features_for_heatmap)[rfe.get_support()])

# feature_ranking = rfecv.ranking_
# ranked_features = pd.DataFrame(
#     {"Feature": df[features_for_heatmap].columns, "Ranking": feature_ranking}
# ).sort_values(by="Ranking")
# pprint.pprint(ranked_features.to_string())

# importances = rfecv.estimator_.feature_importances_
# feature_importance_df = pd.DataFrame(
#     {
#         "Feature": df[features_for_heatmap].columns[rfecv.support_],
#         "Importance": importances,
#     }
# )
# feature_importance_df = feature_importance_df.sort_values(
#     by="Importance", ascending=False
# )
# pprint.pprint("\nFeature Importances (from the final estimator):")
# pprint.pprint(feature_importance_df.to_string())

# [
#     pprint.pprint(val)
#     for idx, val in enumerate(features_for_heatmap)
#     if idx in [0, 1, 6, 13, 18, 20, 21, 22, 23, 64]
# ]

# model = LogisticRegression(max_iter=1000)
# selector = RFECV(model)
# X_new = selector.fit(
#     StandardScaler().fit_transform(df[features_for_heatmap]), df["inducted"]
# ).get_feature_names_out()

# pprint.pprint(X_new)


# plt.figure(figsize=(10, 6))
# sns.heatmap(corr, annot=True)
# plt.show()

# feature_cols = [
#     "ASGs",
#     "PitchingTripleCrowns",
#     "TripleCrowns",
#     "MVPs",
#     "CyYoungs",
#     "GoldGloves",
#     "PEDUser",
# ]

# gives 0.923077 on linear svc
# [
#     "batting_3B",
#     "MVPs",
#     "pitching_SHO",
#     "post_batting_1B",
#     "post_pitching_CG",
# ]

# -----------------------------------------------------------------------------
# feature_cols = [
#     "batting_3B",
#     "MVPs",
#     "pitching_SHO",
#     "post_batting_1B",
#     "post_pitching_CG",
# ]
feature_cols = derived_features
# -----------------------------------------------------------------------------
# best pitching specific features (really good. like > 0.95 precision)
# feature_cols = [
#     "pitching_IPouts",
#     "PitchingTripleCrowns",
#     "MVPs",
#     "post_pitching_IPouts",
#     "CyYoungs",
#     "ASGs",
#     "GoldGloves",
# ]
# best hitting specific features (really bad. like ~ 0.70 precision)
# feature_cols = [
#     "batting_3B",
#     "MVPs",
#     "post_batting_1B",
#     "ASGs",
#     "TripleCrowns",
#     "GoldGloves",
# ]

# ------------------------------------
label_col = ["inducted"]

# print(feature_cols)
# print(label_col)

X = df[feature_cols]
y = df[label_col]
# -----------------------------------

# print(y.value_counts())

# minimal_necessary_accuracy = 1 - (rows_grouped_hof_df / rows_master_df)

# print(minimal_necessary_accuracy)

# correlation_matrix = X.corr()

# print(correlation_matrix)

# print(X)
# print(y)

# -----------------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

ss_train = StandardScaler()
X_train = ss_train.fit_transform(X_train)

ss_test = StandardScaler()
X_test = ss_test.fit_transform(X_test)

models = {}

models["Decision Tree Classifier"] = DecisionTreeClassifier()
models["Random Forest Classifier"] = RandomForestClassifier()
models["Gradient Boosting Classifier"] = GradientBoostingClassifier()
models["XGBoost"] = xgb.XGBClassifier()
# models["LightGBM"] = lgb.LGBMClassifier()
models["Logistic Regression"] = LogisticRegression()
models["Linear SVC"] = LinearSVC()
models["Gaussian NB"] = GaussianNB()
models["K Neighbors Classifier"] = KNeighborsClassifier()
models["MLP Classifier"] = MLPClassifier()

accuracy, precision, recall = {}, {}, {}

for key in models.keys():
    models[key].fit(X_train, y_train.values.ravel())

    # save
    with open(f"{key}.pkl", "wb") as f:
        pickle.dump(models[key], f)

    predictions = models[key].predict(X_test)

    accuracy[key] = metrics.accuracy_score(y_test, predictions)
    precision[key] = metrics.precision_score(y_test, predictions)
    recall[key] = metrics.recall_score(y_test, predictions)


df_model = pd.DataFrame(
    index=models.keys(), columns=["Accuracy", "Precision", "Recall"]
)
df_model["Accuracy"] = accuracy.values()
df_model["Precision"] = precision.values()
df_model["Recall"] = recall.values()

df_sorted = df_model.sort_values(by="Precision", ascending=False)

print(df_sorted)
# -----------------------------------------------------------------------------

# get rid of the steroid users with a column
# find playerID and if they played during the steroid era and won awards

# A system with high recall but low precision returns most of the relevant items, but the proportion of returned results that are incorrectly labeled is high.
# A system with high precision but low recall is just the opposite, returning very few of the relevant items, but most of its predicted labels are correct when compared to the actual labels.
# An ideal system with high precision and high recall will return most of the relevant items, with most results labeled correctly.

# trying to maximize precision with this model
# stingy gatekeeping of the HOF
