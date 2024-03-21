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
# 1. Data Preparation
# ------------------------------

# load data
raw = pd.read_csv("data/raw.csv", encoding="latin")
raw = raw.drop(columns=["Unnamed: 0"])
raw = raw.replace({"-": np.nan, "- ": np.nan, " -": np.nan, " - ": np.nan})

# Check data types
cont = [
    "BI_Length_of_Business",
    "BI_No_of_employees",
    "BI_Salary",
    "OL_Amount",
    "OL_Term",
    "OL_Interest_Rate",
    "OL_Repayment_Amount",
    "SVI_Rental_Monthly_Fee",
    "FC_Total_Cash_Income",
    "FC_Total_Business_Expense",
    "FC_Total_Personal_Expense",
    "FC_Net_Income.client.",
    "FC_Net_Income.FO.",
    "FC_House_width.ft.",
    "FC_House_length.ft.",
    "FC_House_Value",
    "FC_Farm_Size",
    "FC_Farm_Value",
    "FC_Business.Site.Value",
    "FC_Livestock_Count",
    "FC_Livestock_Value",
    "FC_Vehicle_Count",
    "FC_Vehicle_Value",
    "FI_No_Support",
    "FI_No_Depend",
    "PL_Requested_Loan_Amount",
    "PL_Requested_Loan_Term",
    "DL_Actual_Disbursed_Amt",
    "DL_Total_Terms",
    "DL_x_Times_Late",
    "DL_1st_Late_Term",
    "DL_Total_Late_Day",
]

categ = [
    "GI_Branch_Code",
    "GI_Gender",
    "GI_State.Division",
    "GI_Township",
    "GI_VillageorWard_Name",
    "GI_VillageOr_Ward",
    "GI_NRC_State",
    "GI_NRC_Town",
    "BI_Biz_Main_Type",
    "BI_Biz_Sub_type",
    "BI_Govt_Registered",
    "BI_Business_Owner",
    "FI_Marital_Status",
    "FI_Education",
    "BI_Working_Experience_Before_Business",
    "OL_Outstanding_Loan",
    "SVI_Accessibility_Transpo",
    "SVI_Water",
    "SVI_Electricity",
    "SVI_Electricity_Strong.Weak",
    "SVI_Communication_Landline",
    "SVI_Property_Ownership",
    "SVI_OwnerName",
    "SVI_Rental_Period",
    "FC_Livestock_Type",
    "FC_Vehicle_Type",
    "PL_Loan_Purpose1",
    "PL_Loan_Purpose2",
]

# Categorical variables
for variable in categ:
    categories = raw[variable].drop_duplicates().dropna().tolist()
    raw[variable] = raw[variable].astype(pd.api.types.CategoricalDtype(categories))

# Continuous variables
for variable in cont:
    raw[variable] = pd.to_numeric(raw[variable], errors="coerce")

# Application date
raw["GI_Application_Date"] = pd.to_datetime(
    raw["GI_Application_Date"], format="%d/%m/%Y"
)
raw["GI_DOB"] = pd.to_datetime(raw["GI_DOB"], format="%d/%m/%Y")

# Inconsistent entries
buss_owner = raw["BI_Business_Owner"].value_counts().index
buss_owner_dict = {
    category: "Self" if "self" in category.lower() else category
    for category in buss_owner
}
raw["BI_Business_Owner"] = raw["BI_Business_Owner"].replace(buss_owner_dict)

raw["BI_Govt_Registered"] = raw["BI_Govt_Registered"].replace({"n": "N"})

raw["BI_Working_Experience_Before_Business"] = (
    raw["BI_Working_Experience_Before_Business"]
    .str.strip()
    .replace({"n": "N", "y": "Y"})
)

raw["FI_Marital_Status"] = raw["FI_Marital_Status"].replace({"SIngle": "Single"})

raw["OL_Outstanding_Loan"] = raw["OL_Outstanding_Loan"].replace({"n": "N"})

for column in raw.filter(regex="SVI").columns:
    raw[column] = raw[column].replace({"n": "N", "y": "Y"})

raw["GI_NRC_State"] = raw["GI_NRC_State"].replace({"Taninthayi": "Tanintharyi"})

# Remove useless variables for modeling purposes
useless_vars = [
    "No.",
    "GI_Township",
    "GI_VillageorWard_Name",
    "GI_NRC_Town",
    "BI_Biz_Sub_type",
    "SVI_OwnerName",
    "GI_Client_Name",
]
raw = raw.drop(columns=useless_vars)

categories_vehicle_type = ["car", "motorbike", "boat", "bicycle"]


def recat_buss_owner(x):
    self_buss = ["Self", "Seelf", "S"]
    if x in self_buss:
        return "Self"
    else:
        return "Other"


raw["BI_Business_Owner"] = raw["BI_Business_Owner"].apply(recat_buss_owner)

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

# Variables with more than 80% missing values
vars_high_missing = (
    (100 * raw.isna().sum() / raw.shape[0])
    .sort_values(ascending=False)
    .loc[lambda x: x > 80]
    .index.to_list()
)
# Remove previous variables from the dataset to check performance of the model
raw = raw.drop(columns=vars_high_missing)

# 6. Sample Selection
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=1234)
for train_index, test_index in split.split(raw_filter, raw_filter["GB"]):
    train = raw_filter.loc[train_index]
    test = raw_filter.loc[test_index]

# ------------------------------
# 3. Univariate Analysis
# ------------------------------

# 1. Long list of variables

woe, iv = func.woe_iv(train, "GB", bins=5)
woe.to_csv("tables/woe_test.csv")
iv.to_csv("tables/iv_test.csv")

top20_variables = iv.sort_values(ascending=False).head(20).index

# 2. Visualize the WOE

for var in top20_variables:
    func.woe_plot(woe, var)
