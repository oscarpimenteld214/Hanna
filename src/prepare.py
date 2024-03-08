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
import pickle
import os

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
    raw[variable] = raw[variable].astype(CategoricalDtype(categories))

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
# Variables with few accounts, such that in some categories we don't have bad accounts. This lead to a inf information value. We will remove these variables from the dataset
useless_vars += [
    "GI_NRC_State",
    "GI_Branch_Code",
    "PL_Loan_Purpose1",
    "FC_Vehicle_Type",
    "GI_State.Division",
]
raw = raw.drop(columns=useless_vars)

def recat_buss_owner(x):
    self_buss = ["Self", "Seelf", "S"]
    if x in self_buss:
        return "Self"
    else:
        return "Other"
raw["BI_Business_Owner"] = raw["BI_Business_Owner"].apply(recat_buss_owner)

categories = ["A", "B", "C"]
raw["FI_Education"] = raw["FI_Education"].cat.reorder_categories(
    categories, ordered=True
)

# Create age at day application column
raw["GI_Age_at_app_date"] = (raw["GI_Application_Date"] - raw["GI_DOB"]).dt.days // 365

# date of application by year and month
raw["GI_Application_YYYYMM"] = raw["GI_Application_Date"].dt.to_period("M")

#GI_list = ["GI_Application_YYYYMM", "GI_Age_at_app_date"]
#BI_list = ["BI_Biz_Main_Type", "BI_Length_of_Business", "OL_Outstanding_Loan"]
#FC_list = ["FC_Total_Cash_Income", "FC_Total_Business_Expense", "FC_Net_Income.FO."]
#DL_list = raw.filter(regex="DL").columns.tolist()

#func.feature_plot(raw, GI_list, "GI.png")
#func.feature_plot(raw, BI_list, "BI.png")
#func.feature_plot(raw, FC_list, "FC.png")
#func.feature_plot(raw, DL_list, "DL.png")

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

os.mkdir("data\prepared")

raw_filter.to_csv("data\prepared\Data_cleaned_filtered.csv", index=False)
# There useless categories as a result of exceptions that will be removed after the GB tagging in the following stage. In that stage we will filter only good and bad borrowers, excluding undefined ones.

dtypes_dict = {raw_filter[column].name: raw_filter[column].dtype.name for column in raw_filter.columns}
with open('data\prepared\dtypes_dict.pkl', 'wb') as fp:
    pickle.dump(dtypes_dict, fp)