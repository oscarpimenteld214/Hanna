# Libraries
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from pandas.api.types import CategoricalDtype
import seaborn as sns
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.linear_model import LogisticRegression
from scipy.stats import ks_2samp, norm
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
import xgboost as xgb
import os
import pickle
import sklearn

import functions as func

pd.options.mode.copy_on_write = True
pd.set_option("mode.copy_on_write", False)

# Load data
in_path = "data/univariate/"

train = func.get_data(in_path + "Train.csv", in_path + "dtypes_dict.pkl")
test = func.get_data(in_path + "Test.csv", in_path + "dtypes_dict.pkl")
with open(in_path + "x_shortlisted.pkl", "rb") as fp:
    x_shortlisted = pickle.load(fp)

# Create output folder
out_path = "data/modeling/"
try:
    os.mkdir(out_path)
except:
    pass

# Useless variables for modelling
vars_to_remove = [
    "GI_Application_Date",
    "GI_Client_ID",
    "GI_DOB",
    "GI_Application_YYYYMM",
]
train = train.drop(vars_to_remove, axis=1)
test = test.drop(vars_to_remove, axis=1)

# ------------------------------------------------------
# Model 1: Logistic Regression (x_shortlisted)
# ------------------------------------------------------
X_train, y_train, X_test, y_test = func.prepare_for_modeling(
    train, test, x_shortlisted, "GB"
)

model = LogisticRegression(solver="liblinear").fit(X_train, y_train)

metrics_to_print = ["Model 1: Logistic Regression (x_shortlisted)"]
metrics_to_print.append(func.auroc(X_train, y_train, model, "train"))
metrics_to_print.append(func.auroc(X_test, y_test, model, "test"))
metrics_to_print.append("\n")

scores_table = func.get_scores_table(
    train[x_shortlisted], model, ref_odds=100, ref_score=500, pdo=40
)
scores_table.to_csv(out_path + "scores_table.csv", index=False)

score_bands = func.get_scores_distribution(
    train[x_shortlisted], y_train, scores_table, bins=8, path=out_path + "scores_dist.png"
)

# ------------------------------------------------------
# Model 2: Random forest (x_shortlisted)
# ------------------------------------------------------

X_train, y_train, X_test, y_test = func.prepare_for_modeling(
    train, test, x_shortlisted, "GB"
)

rnd_clf = RandomForestClassifier(n_estimators=200, max_depth=4, n_jobs=-1)
rnd_clf.fit(X_train, y_train)

metrics_to_print.append("Model 2: Random Forest (x_shortlisted)")
metrics_to_print.append(func.auroc(X_train, y_train, rnd_clf, "train"))
metrics_to_print.append(func.auroc(X_test, y_test, rnd_clf, "test"))
metrics_to_print.append("\n")

# ------------------------------------------------------
# Model 3 Random Forest (with all variables)
# ------------------------------------------------------

X_train, y_train, X_test, y_test = func.prepare_for_modeling(
    train, test, train.columns.to_list(), "GB"
)

# Remove columns with missing values
X_train = X_train.dropna(axis=1)
X_test = X_test[X_train.columns]

rnd_clf_all = RandomForestClassifier(n_estimators=200, max_depth=5, n_jobs=-1)
rnd_clf_all.fit(X_train, y_train)

metrics_to_print.append("Model 3: Random Forest (all variables)")
metrics_to_print.append(func.auroc(X_train, y_train, rnd_clf_all, "train"))
metrics_to_print.append(func.auroc(X_test, y_test, rnd_clf_all, "test"))
metrics_to_print.append("\n")

# ------------------------------------------------------
# Model 4 AdaBoost (with all variables)
# ------------------------------------------------------

X_train, y_train, X_test, y_test = func.prepare_for_modeling(
    train, test, train.columns.to_list(), "GB"
)

# Remove columns with missing values
X_train = X_train.dropna(axis=1)
X_test = X_test[X_train.columns]

ada_clf = AdaBoostClassifier(
    DecisionTreeClassifier(max_depth=1),
    n_estimators=200,
    algorithm="SAMME.R",
    learning_rate=0.7,
)
ada_clf.fit(X_train, y_train)

metrics_to_print.append("Model 4: AdaBoost (all variables)")
metrics_to_print.append(func.auroc(X_train, y_train, ada_clf, "train"))
metrics_to_print.append(func.auroc(X_test, y_test, ada_clf, "test"))
metrics_to_print.append("\n")

# ------------------------------------------------------
# Model 5 XGBoost (with all variables)
# ------------------------------------------------------

X_train, y_train, X_test, y_test = func.prepare_for_modeling(
    train, test, train.columns.to_list(), "GB"
)

X_train = X_train.loc[:, ~X_train.columns.str.contains("gpd")]
X_test = X_test.loc[:, ~X_test.columns.str.contains("gpd")]

xgb_clf = xgb.XGBClassifier(n_estimators=200, max_depth=1, learning_rate=0.035)
xgb_clf.fit(X_train, y_train)

metrics_to_print.append("Model 5: XGBoost (all variables)")
metrics_to_print.append(func.auroc(X_train, y_train, xgb_clf, "train"))
metrics_to_print.append(func.auroc(X_test, y_test, xgb_clf, "test"))
metrics_to_print.append("\n")

# save metrics results
f = open(out_path + "metrics.txt", "w")
f.write("\n".join(metrics_to_print))
