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
from preprocess import get_df_for_modeling
from sklearn.preprocessing import StandardScaler
import pandas as pd
import xgboost as xgb
import lightgbm as lgb

df = get_df_for_modeling()

# print("df.corr()")
# print(df.corr())

label_name = "inducted"
all_cols = list(df.columns.values)

feature_cols = [x for x in all_cols if x != label_name]
label_col = [label_name]

print(feature_cols)
print(label_col)

X = df[feature_cols]
y = df[label_col]

# print(y.value_counts())

# correlation_matrix = X.corr()

# print(correlation_matrix)

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
models["LightGBM"] = lgb.LGBMClassifier()
models["Logistic Regression"] = LogisticRegression()
models["Linear SVC"] = LinearSVC()
models["Gaussian NB"] = GaussianNB()
models["K Neighbors Classifier"] = KNeighborsClassifier()
models["MLP Classifier"] = MLPClassifier()

accuracy, precision, recall = {}, {}, {}

for key in models.keys():
    models[key].fit(X_train, y_train.values.ravel())

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

# get rid of the steroid users with a column
# find playerID and if they played during the steroid era and won awards

# A system with high recall but low precision returns most of the relevant items, but the proportion of returned results that are incorrectly labeled is high.
# A system with high precision but low recall is just the opposite, returning very few of the relevant items, but most of its predicted labels are correct when compared to the actual labels.
# An ideal system with high precision and high recall will return most of the relevant items, with most results labeled correctly.

# trying to maximize precision with this model
# stingy gatekeeping of the HOF
