from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import OneClassSVM
from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    RandomizedSearchCV,
    StratifiedKFold,
)
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
from sklearn.svm import SVC
from sklearn.linear_model import Lasso
import numpy as np
import pprint
import os
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
from eval_utils import EvalUtils
from sklearn.datasets import make_classification
from sklearn.metrics import precision_score, classification_report


def recursive_vif_selection(
    data: pd.DataFrame, vif_threshold: float = 5.0
) -> pd.DataFrame:
    """
    Recursively selects features based on Variance Inflation Factor (VIF).

    VIF = 1: No multicollinearity
    1 < VIF < 5 Low to moderate multicollinearity. May not be a major concern.
    5 < VIF < 10: High multicollinearity. Consider investigating and potentially removing variables
    VIF >= 10: Very high multicollinearity. Likely needs to be addressed to stabilize the model

    Args:
        data (pd.DataFrame): DataFrame containing the features.
        vif_threshold (float): Threshold for VIF. Features with VIF above this
                               value will be removed iteratively.

    Returns:
        pd.DataFrame: DataFrame containing the selected features (those with VIF
                      below the threshold).
    """
    if data.empty:
        return data

    # Add a constant for VIF calculation
    data_with_const = add_constant(data)

    vif = pd.DataFrame()
    vif["Feature"] = data_with_const.columns
    vif["VIF"] = [
        variance_inflation_factor(data_with_const.values, i)
        for i in range(data_with_const.shape[1])
    ]

    # Remove the constant row
    vif = vif[vif["Feature"] != "const"]

    # Find the feature with the highest VIF
    highest_vif_feature = vif.loc[vif["VIF"].idxmax()]

    if highest_vif_feature["VIF"] > vif_threshold:
        print(
            f"Removing '{highest_vif_feature['Feature']}' with VIF = {highest_vif_feature['VIF']:.2f}"
        )
        data = data.drop(columns=[highest_vif_feature["Feature"]])
        return recursive_vif_selection(data, vif_threshold)
    else:
        print("All remaining features have VIF below the threshold.")
        return data


# pitching_Pitcher
# post_pitching_Pitcher
# batting_Batter
# post_batting_Batter

eval_utils = EvalUtils()

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
        "batting_AB",
        "batting_R",
        "batting_H",
        "batting_2B",
        "batting_3B",
        "batting_HR",
        "batting_RBI",
        "batting_SB",
        "batting_CS",
        "batting_BB",
        "batting_SO",
        "batting_IBB",
        "batting_HBP",
        "batting_SH",
        "batting_SF",
        "batting_GIDP",
        "batting_1B",
        "batting_TB",
        "batting_OBP",
        "batting_SLG",
        "batting_OPS",
        "batting_SBP",
        "batting_BAbip",
        "batting_ISO",
        "batting_PowerSpeedNumber",
        "batting_ABPerHR",
        "batting_ABPerK",
        "post_batting_AB",
        "post_batting_R",
        "post_batting_H",
        "post_batting_2B",
        "post_batting_3B",
        "post_batting_HR",
        "post_batting_RBI",
        "post_batting_SB",
        "post_batting_CS",
        "post_batting_BB",
        "post_batting_SO",
        "post_batting_IBB",
        "post_batting_HBP",
        "post_batting_SH",
        "post_batting_SF",
        "post_batting_GIDP",
        "post_batting_1B",
        "post_batting_TB",
        "post_batting_OBP",
        "post_batting_SLG",
        "post_batting_OPS",
        "post_batting_SBP",
        "post_batting_BAbip",
        "post_batting_ISO",
        "post_batting_PowerSpeedNumber",
        "post_batting_ABPerHR",
        "post_batting_ABPerK",
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
        # "inducted",
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
# -----------------------------------------------------------------------------
# derived_features = recursive_vif_selection(
#     df[features_for_heatmap], vif_threshold=1.99
# ).columns.tolist()
# pprint.pprint(derived_features)
# -----------------------------------------------------------------------------
# vt = VarianceThreshold(threshold=0.2)
# X_train_vt = vt.fit_transform(df[features_for_heatmap])
# pprint.pprint("vt:")
# pprint.pprint(np.array(features_for_heatmap)[vt.get_support()])

# derived_features = list(np.array(features_for_heatmap)[vt.get_support()])
# -----------------------------------------------------------------------------
# pprint.pprint(df[features_for_heatmap])
# -----------------------------------------------------------------------------

# new_features = SelectKBest(f_classif, k=10).fit(
#     df[features_for_heatmap], df["inducted"]
# )

# pprint.pprint(new_features.get_feature_names_out())
# pprint.pprint(new_features.get_support())

# skb_mi = SelectKBest(score_func=mutual_info_classif, k=10)
# X_train_mi = skb_mi.fit_transform(df[features_for_heatmap], df["inducted"])
# pprint.pprint("skb_mi:")
# pprint.pprint(np.array(features_for_heatmap)[skb_mi.get_support()])

# derived_features = list(np.array(features_for_heatmap)[skb_mi.get_support()])

# skb_anova = SelectKBest(score_func=f_classif, k=10)
# X_train_anova = skb_anova.fit_transform(df[features_for_heatmap], df["inducted"])
# pprint.pprint("skb_anova:")
# pprint.pprint(np.array(features_for_heatmap)[skb_anova.get_support()])

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

# model = xgb.XGBClassifier(n_estimators=25)
# cv = StratifiedKFold(n_splits=5)
# rfe = RFE(model, n_features_to_select=10)
# rfe.fit_transform(df[features_for_heatmap], df["inducted"])

# pprint.pprint("rfe:")

# derived_features = list(np.array(features_for_heatmap)[rfe.get_support()])

# model = LinearSVC()
# rfe = RFE(model, n_features_to_select=10)
# rfe.fit_transform(df[features_for_heatmap], df["inducted"])

# pprint.pprint("rfe:")

# derived_features = list(np.array(features_for_heatmap)[rfe.get_support()])

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
# feature_cols = derived_features
feature_cols = features_for_heatmap
# feature_cols = [
#     "batting_CS",
#     "batting_SH",
#     "batting_SBP",
#     "batting_BAbip",
#     "batting_ISO",
#     "batting_ABPerHR",
#     "batting_ABPerK",
#     "post_batting_3B",
#     "post_batting_CS",
#     "post_batting_IBB",
#     "post_batting_HBP",
#     "post_batting_SH",
#     "post_batting_SF",
#     "post_batting_SBP",
#     "post_batting_BAbip",
#     "post_batting_ISO",
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
# feature_cols = ["batting_3B", "MVPs", "post_batting_ABPerK", "ASGs", "GoldGloves"]
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

# models = {}

# models["Decision Tree Classifier"] = DecisionTreeClassifier()
# models["Random Forest Classifier"] = RandomForestClassifier()
# models["Gradient Boosting Classifier"] = GradientBoostingClassifier()
# models["XGBoost"] = xgb.XGBClassifier()
# # models["LightGBM"] = lgb.LGBMClassifier()
# models["Logistic Regression"] = LogisticRegression()
# models["Linear SVC"] = LinearSVC()
# models["Gaussian NB"] = GaussianNB()
# models["K Neighbors Classifier"] = KNeighborsClassifier()
# models["MLP Classifier"] = MLPClassifier()
# models["SVC"] = SVC()

# accuracy, precision, recall = {}, {}, {}

# folder_path = "trained_models/"
# os.makedirs(folder_path, exist_ok=True)

# for key in models.keys():
#     models[key].fit(X_train, y_train.values.ravel())

#     file_name = f"{key}.pkl"
#     full_path = f"{folder_path}{file_name}"
#     # save
#     with open(full_path, "wb") as f:
#         pickle.dump(models[key], f)

#     predictions = models[key].predict(X_test)

#     accuracy[key] = metrics.accuracy_score(y_test, predictions)
#     precision[key] = metrics.precision_score(y_test, predictions)
#     recall[key] = metrics.recall_score(y_test, predictions)

#     metrics.confusion_matrix(y_test, predictions)


# df_model = pd.DataFrame(
#     index=models.keys(), columns=["Accuracy", "Precision", "Recall"]
# )
# df_model["Accuracy"] = accuracy.values()
# df_model["Precision"] = precision.values()
# df_model["Recall"] = recall.values()

# df_sorted = df_model.sort_values(by="Precision", ascending=False)

# print(df_sorted)
# -----------------------------------------------------------------------------
# tree_basic = DecisionTreeClassifier(criterion="gini", random_state=1)
# tree_basic.fit(X_train, y_train)

# eval_utils.print_confusion_matrix(tree_basic, X_test, y_test)
# eval_utils.get_scores(tree_basic, X_train, X_test, y_train, y_test)
# eval_utils.print_tree(tree_basic, X_train)
# eval_utils.get_importances(tree_basic, X_train)

# parameters = {
#     "max_depth": [6, 8, 9, 12, 15, None],
#     "criterion": ["entropy", "gini"],
#     "splitter": ["best", "random"],
#     "max_leaf_nodes": [2, 3, 5, 6, 10, None],
#     "min_samples_leaf": [1, 2, 3, None],
# }

# scorer = metrics.make_scorer(metrics.precision_score)

# # Run the grid search
# grid_obj = GridSearchCV(tree_basic, parameters, scoring=scorer, cv=5)
# grid_obj = grid_obj.fit(X, y)

# # Set the clf to the best combination of parameters
# tree_preprune = grid_obj.best_estimator_

# print("Best Hyper Parameters", grid_obj.best_params_)

# # Fit the best algorithm to the data.
# tree_preprune.fit(X_test, y_test)

# eval_utils.print_confusion_matrix(tree_preprune, X_test, y_test)
# eval_utils.get_scores(tree_preprune, X_train, X_test, y_train, y_test)

# pd.concat(
#     [
#         eval_utils.get_scores(tree_basic, X_train, X_test, y_train, y_test),
#         eval_utils.get_scores(tree_preprune, X_train, X_test, y_train, y_test),
#     ]
# )

# eval_utils.print_tree(tree_preprune, X_train)
# eval_utils.get_importances(tree_preprune, X_train)

# tree_postprune = DecisionTreeClassifier(random_state=1)
# path = tree_postprune.cost_complexity_pruning_path(X_train, y_train)
# ccp_alphas, impurities = path.ccp_alphas, path.impurities
# # print the alphas and impurities
# pd.DataFrame(path)

# trees_postprune = []
# for ccp_alpha in ccp_alphas:
#     tree_alpha = DecisionTreeClassifier(random_state=1, ccp_alpha=ccp_alpha)
#     # Fit each tree to the training data
#     tree_alpha.fit(X_train, y_train)
#     trees_postprune.append(tree_alpha)

# prune_score_train = []
# prune_score_test = []
# for trp in trees_postprune:
#     prune_testp = trp.predict(X_test)
#     prune_trainp = trp.predict(X_train)
#     prune_score_test.append(metrics.precision_score(y_test, prune_testp))
#     prune_score_train.append(metrics.precision_score(y_train, prune_trainp))

# Plot F1 score vs each alpha
# fig, ax = plt.subplots(figsize=(15, 5))
# ax.set_xlabel("alpha")
# ax.set_ylabel("Score")
# ax.set_title("F1 vs alpha for training and testing sets")
# ax.plot(
#     ccp_alphas, prune_score_train, marker="o", label="train", drawstyle="steps-post"
# )
# ax.plot(ccp_alphas, prune_score_test, marker="o", label="test", drawstyle="steps-post")
# ax.legend()
# plt.show()

# tree_best_postprune = trees_postprune[np.argmax(prune_score_test)]

# eval_utils.print_tree(tree_best_postprune, X_train)

# eval_utils.print_confusion_matrix(tree_best_postprune, X_test, y_test)
# pd.concat(
#     [
#         eval_utils.get_scores(tree_preprune, X_train, X_test, y_train, y_test),
#         eval_utils.get_scores(tree_best_postprune, X_train, X_test, y_train, y_test),
#     ]
# )

# pprint.pprint(
#     metrics.precision_score(
#         y_test,
#         DecisionTreeClassifier(
#             criterion="entropy",
#             max_depth=6,
#             max_leaf_nodes=10,
#             min_samples_leaf=1,
#             splitter="random",
#             random_state=1,
#         )
#         .fit(
#             X_train,
#             y_train,
#         )
#         .predict(X_test),
#     )
# )

# eval_utils.get_importances(tree_best_postprune, X_train)


# get rid of the steroid users with a column
# find playerID and if they played during the steroid era and won awards

# A system with high recall but low precision returns most of the relevant items, but the proportion of returned results that are incorrectly labeled is high.
# A system with high precision but low recall is just the opposite, returning very few of the relevant items, but most of its predicted labels are correct when compared to the actual labels.
# An ideal system with high precision and high recall will return most of the relevant items, with most results labeled correctly.

# trying to maximize precision with this model
# stingy gatekeeping of the HOF

# pruning got to 0.875 with VIF Selection using all batter numericals
# pruning got to 0.846 with using all batter numericals

param_grid_xgb = {
    "n_estimators": [100, 200, 300],
    "learning_rate": [0.01, 0.05, 0.1],
    "max_depth": [3, 4, 5],
    "min_child_weight": [1, 3, 5],
    "gamma": [0, 0.1, 0.2],
    "subsample": [0.8, 1.0],
    "colsample_bytree": [0.8, 1.0],
    "reg_alpha": [0, 0.1],
    "reg_lambda": [1, 1.5],
    "objective": ["binary:logistic"],  # Or your specific objective
    "scale_pos_weight": [1],  # Adjust if you have imbalanced data
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Initialize GridSearchCV
# grid_search_xgb = GridSearchCV(
#     estimator=xgb.XGBClassifier(
#         random_state=42, use_label_encoder=False, eval_metric="logloss"
#     ),
#     param_grid=param_grid_xgb,
#     scoring="precision",  # Or your desired metric
#     cv=cv,
#     n_jobs=-1,  # Use all available cores
#     verbose=2,
# )

# grid_search_xgb.fit(X_train, y_train)

random_search_xgb = RandomizedSearchCV(
    estimator=xgb.XGBClassifier(
        random_state=42, use_label_encoder=False, eval_metric="logloss"
    ),
    param_distributions=param_grid_xgb,
    n_iter=10,  # Number of parameter settings that are sampled
    scoring="precision",
    cv=cv,
    n_jobs=-1,
    verbose=2,
    random_state=42,
)

random_search_xgb.fit(X_train, y_train)

# if isinstance(grid_search_xgb, GridSearchCV):
#     print("Best Hyperparameters (GridSearchCV):", grid_search_xgb.best_params_)
#     print("Best Score (GridSearchCV):", grid_search_xgb.best_score_)
#     best_xgb_model = grid_search_xgb.best_estimator_
if isinstance(
    random_search_xgb, RandomizedSearchCV
):  # isinstance(random_search_xgb, RandomizedSearchCV)
    print("Best Hyperparameters (RandomizedSearchCV):", random_search_xgb.best_params_)
    print("Best Score (RandomizedSearchCV):", random_search_xgb.best_score_)
    best_xgb_model = random_search_xgb.best_estimator_

y_pred_xgb = best_xgb_model.predict(X_test)
precision_xgb = precision_score(y_test, y_pred_xgb)
print("Test Precision of the Best XGBoost Model:", precision_xgb)

print("\nClassification Report of the Best XGBoost Model:")
print(classification_report(y_test, y_pred_xgb))
