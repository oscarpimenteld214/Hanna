from typing import Tuple, List
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from pandas.api.types import CategoricalDtype
import seaborn as sns
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from scipy.stats import ks_2samp, norm
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
import xgboost as xgb
import pickle
import os

import functions as func

pd.options.mode.copy_on_write = True
pd.set_option("mode.copy_on_write", False)

# ------------------------------
# 3. Univariate Analysis
# ------------------------------

# load data from prepare stage
with open("data\prepared\dtypes_dict.pkl", "rb") as fp:
    dtypes_dict = pickle.load(fp)
date_dtypes = ["datetime64[ns]", "datetime64", "datetime"]
parse_dates = []
for key, value in dtypes_dict.items():
    if value in date_dtypes:
        dtypes_dict[key] = "object"
        parse_dates.append(key)

train = pd.read_csv(
    "data/portfolio/Train.csv",
    dtype=dtypes_dict,
    parse_dates=parse_dates,
    encoding="latin",
)

test = pd.read_csv(
    "data/portfolio/Test.csv",
    dtype=dtypes_dict,
    parse_dates=parse_dates,
    encoding="latin",
)

# Create output folder
path = "data/univariate/"
os.mkdir(path)

# 1. Long list of variables

woe, iv = func.woe_iv(train, "GB", bins=5)
woe.to_csv(path + "woe.csv")
iv.to_csv(path + "iv.csv")

top20_variables = iv.sort_values(ascending=False).head(20).index

# 2. Visualize the WOE

for var in top20_variables:
    func.woe_plot(woe, var, path)

# 3. Coerce classing


def add_coerse_variable(var_name):
    grp_var = var_name + "_gpd"
    train[grp_var] = func.coerse_classing(train[var_name], left_bounds, right_bounds)
    test[grp_var] = func.coerse_classing(test[var_name], left_bounds, right_bounds)


x_shortlisted = []

# DL_Total_Terms
# --------------------------------
var_tocoarse = "DL_Total_Terms"
left_bounds = [0.0, 12.0]
right_bounds = [12.0, 30.0]
func.coerse_classing_results(train, var_tocoarse, left_bounds, right_bounds, "GB", path)

# Add grouped column to train-test sets
add_coerse_variable(var_tocoarse)

# Add to shortlist of variables
x_shortlisted.append(var_tocoarse + "_gpd")
# --------------------------------

# DL_Actual_Disbursed_Amt
# --------------------------------
var_tocoarse = "DL_Actual_Disbursed_Amt"
left_bounds = [1e4, 4e5, 5e5, 10e5]
right_bounds = [4e5, 5e5, 10e5, 60e5]
func.coerse_classing_results(train, var_tocoarse, left_bounds, right_bounds, "GB", path)

# Add grouped column to train-test sets
add_coerse_variable(var_tocoarse)

# Add to shortlist of variables
x_shortlisted.append(var_tocoarse + "_gpd")
# --------------------------------

# FC_Total_Business_Expense
# --------------------------------
variable = "FC_Total_Business_Expense"

train[variable + '_gpd'] = pd.qcut(train[variable], q=5, duplicates="drop") 
test[variable + '_gpd'] = pd.qcut(test[variable], q=5, duplicates="drop")

# Add to shortlist of variables
x_shortlisted.append(variable + '_gpd')
# --------------------------------

# FC_House_area_ftsquare
# --------------------------------
var_tocoarse = "FC_House_area_ftsquare"
left_bounds = [89.0, 732.8, 1200.0, 1970.0]
right_bounds = [732.8, 1200.0, 1970.0, 75000]
func.coerse_classing_results(train, var_tocoarse, left_bounds, right_bounds, "GB", path)

# Add grouped column to train-test sets
add_coerse_variable(var_tocoarse)

# Add to shortlist of variables
# x_shortlisted.append(var_tocoarse + "_gpd")
# --------------------------------

# FC_House_width.ft.
# --------------------------------
var_tocoarse = "FC_House_width.ft."
left_bounds = [8.0, 20.0, 23.0, 40.0]
right_bounds = [20.0, 23.0, 40.0, np.inf]
func.coerse_classing_results(train, var_tocoarse, left_bounds, right_bounds, "GB", path)

# Add grouped column to train-test sets
add_coerse_variable(var_tocoarse)

# Add to shortlist of variables
x_shortlisted.append(var_tocoarse + "_gpd")
# --------------------------------

# BI_Biz_Main_Type
# --------------------------------
# Organize categories based on the WOE
var_to_organize = "BI_Biz_Main_Type"
categories = (
    woe[woe["Variable"] == var_to_organize].sort_values(by="WoE").Categories.values
)
train[var_to_organize] = train[var_to_organize].cat.reorder_categories(
    categories, ordered=True
)
test[var_to_organize] = test[var_to_organize].cat.reorder_categories(
    categories, ordered=True
)

# Add to shortlist of variables
x_shortlisted.append(var_to_organize)

# FC_Total_Business_Expense
# --------------------------------
variable = "FC_Total_Personal_Expense"

train[variable + '_gpd'] = pd.qcut(train[variable], q=5, duplicates="drop") 
test[variable + '_gpd'] = pd.qcut(test[variable], q=5, duplicates="drop")

# Add to shortlist of variables
x_shortlisted.append(variable + '_gpd')
# --------------------------------

# FI_No_Depend
# --------------------------------
var_tocoarse = "FI_No_Depend"
left_bounds = [-0.001, 1.0, 3.0]
right_bounds = [1.0, 3.0, 8.0]
func.coerse_classing_results(train, var_tocoarse, left_bounds, right_bounds, "GB", path)

# Add grouped column to train-test sets
add_coerse_variable(var_tocoarse)

# Add to shortlist of variables
x_shortlisted.append(var_tocoarse + "_gpd")
# --------------------------------

# FC_Income_Expense_Ratio_woe
# --------------------------------
var_tocoarse = "FC_Income_Expense_Ratio"
left_bounds = [-0.001, 1.379, 2.114, 2.764]
right_bounds = [1.379, 2.114, 2.764, np.inf]
func.coerse_classing_results(train, var_tocoarse, left_bounds, right_bounds, "GB", path)

# Add grouped column to train-test sets
add_coerse_variable(var_tocoarse)

# Add to shortlist of variables
x_shortlisted.append(var_tocoarse + "_gpd")
# --------------------------------

# FI_No_Depend
# --------------------------------
var_tocoarse = "FI_No_Support"
left_bounds = [-0.001, 1.0, 2.0]
right_bounds = [1.0, 2.0, 7.0]
func.coerse_classing_results(train, var_tocoarse, left_bounds, right_bounds, "GB", path)

# Add grouped column to train-test sets
add_coerse_variable(var_tocoarse)

# Add to shortlist of variables
x_shortlisted.append(var_tocoarse + "_gpd")
# --------------------------------

train.to_csv(path + "Train.csv", index=False)
test.to_csv(path + "Test.csv", index=False)