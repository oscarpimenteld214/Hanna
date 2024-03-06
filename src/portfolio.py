# 0. Libraries
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

import functions as func

pd.options.mode.copy_on_write = True
pd.set_option("mode.copy_on_write", False)

# ------------------------------
# 2. Portfolio Analysis
# ------------------------------

# 1 Portfolio overview
# Create age at day application column
raw["GI_Age_at_app_date"] = (raw["GI_Application_Date"] - raw["GI_DOB"]).dt.days // 365

# date of application by year and month
raw["GI_Application_YYYYMM"] = raw["GI_Application_Date"].dt.to_period("M")

GI_list = ["GI_Application_YYYYMM", "GI_Age_at_app_date"]
BI_list = ["BI_Biz_Main_Type", "BI_Length_of_Business", "OL_Outstanding_Loan"]
FC_list = ["FC_Total_Cash_Income", "FC_Total_Business_Expense", "FC_Net_Income.FO."]
DL_list = raw.filter(regex="DL").columns.tolist()

func.feature_plot(raw, GI_list, "GI.png")
func.feature_plot(raw, BI_list, "BI.png")
func.feature_plot(raw, FC_list, "FC.png")
func.feature_plot(raw, DL_list, "DL.png")

# 2. Exclusions
policy_age = (raw["GI_Age_at_app_date"] >= 18) & (raw["GI_Age_at_app_date"] <= 60)
app_date_no_missing = ~raw["GI_Application_Date"].isna()
non_duplicate_records = ~raw["GI_Client_ID"].duplicated(keep=False)
# Getting index of accounts that satisfy the conditions
policy_age_index = raw[policy_age].index
app_date_missing_index = raw[app_date_no_missing].index
duplicate_records_index = raw[non_duplicate_records | raw["GI_Client_ID"].isna()].index
# Filter the dataset to get the accounts that satisfy all the conditions
good_index = list(
    set(duplicate_records_index) & set(policy_age_index) & set(app_date_missing_index)
)

raw_filter = raw.filter(items=good_index, axis=0)

# 3. Good-Bad Definition
perf_window = 12

Max_application_date = raw_filter["GI_Application_Date"].max()
raw_filter["MOB"] = (
    Max_application_date - raw_filter["GI_Application_Date"]
).dt.days // 30


def GB_tag(x):
    if x["DL_Total_Late_Day"] >= 90:
        return 1  # Bad
    elif x["DL_Total_Late_Day"] < 90 and x["MOB"] >= perf_window:
        return 0  # Good
    else:
        return -1  # Undefined


raw_filter["GB"] = raw_filter.apply(GB_tag, axis=1)

# Remove variable used to define the GB tagging and performance variables
raw_filter = raw_filter.drop(columns=["DL_Total_Late_Day", "MOB"])
raw_filter = raw_filter.drop(columns=["DL_x_Times_Late", "DL_1st_Late_Term"])


# 4. Variable Generation
# In the new variable V3_App_State_eq_NRC_State, 0 means false and 1 means true
raw_filter["R5_House_Value_to_Req_Loan_Amt"] = (
    raw_filter["FC_House_Value"] / raw_filter["PL_Requested_Loan_Amount"]
)
raw_filter["R8_Farm_Value_to_Farm_Size"] = (
    raw_filter["FC_Farm_Value"] / raw_filter["FC_Farm_Size"]
)
raw_filter["V3_App_State_eq_NRC_State"] = (
    raw_filter["GI_State.Division"].astype(object)
    == raw_filter["GI_NRC_State"].astype(object)
).astype(int)
raw_filter['FC_House_area_ftsquare'] = raw_filter['FC_House_width.ft.'] * raw_filter['FC_House_length.ft.']

raw_filter = raw_filter[raw_filter["GB"] != -1].reset_index()
raw_filter = raw_filter.drop(columns="index")

# Remove useless categories from the categorical variables as a result of exceptions and filters
for variable in raw_filter.columns:
    try:
        raw_filter[variable] = raw_filter[variable].cat.remove_unused_categories()
    except:
        pass

# 5. Missing analysis
# Variables with more than 80% missing values
vars_high_missing = (
    (100 * raw_filter.isna().sum() / raw_filter.shape[0])
    .sort_values(ascending=False)
    .loc[lambda x: x > 80]
    .index.to_list()
)
# Remove previous variables from the dataset to check performance of the model
raw_filter = raw_filter.drop(columns=vars_high_missing)

# 6. Sample Selection
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=1234)
for train_index, test_index in split.split(raw_filter, raw_filter["GB"]):
    train = raw_filter.loc[train_index]
    test = raw_filter.loc[test_index]