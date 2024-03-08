# 0. Libraries
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
import os
import pickle
import numpy as np

import functions as func

pd.options.mode.copy_on_write = True
pd.set_option("mode.copy_on_write", False)

# ------------------------------
# 2. Portfolio Analysis
# ------------------------------

# load data from prepare stage
with open('data\prepared\dtypes_dict.pkl', 'rb') as fp:
    dtypes_dict = pickle.load(fp)
date_dtypes = ["datetime64[ns]", "datetime64", "datetime"]
parse_dates = []
for key, value in dtypes_dict.items():
    if value in date_dtypes:
        dtypes_dict[key] = "object"
        parse_dates.append(key)

raw_filter = pd.read_csv("data/prepared/Data_cleaned_filtered.csv", dtype=dtypes_dict, parse_dates=parse_dates, encoding="latin")

# Create output folder
os.mkdir("data\portfolio")

# 1 Portfolio overview
#"GI_Application_YYYYMM", 
GI_list = ["GI_Application_YYYYMM", "GI_Age_at_app_date"]
BI_list = ["BI_Biz_Main_Type", "BI_Length_of_Business", "OL_Outstanding_Loan"]
FC_list = ["FC_Total_Cash_Income", "FC_Total_Business_Expense", "FC_Net_Income.FO."]
DL_list = raw_filter.filter(regex="DL").columns.tolist()

func.feature_plot(raw_filter, GI_list, "data\portfolio\GeneralIinfo.png")
func.feature_plot(raw_filter, BI_list, "data\portfolio\BusinessInfo.png")
func.feature_plot(raw_filter, FC_list, "data\portfolio\FinancialCond.png")
func.feature_plot(raw_filter, DL_list, "data\portfolio\DisbursedLoan.png")

# 3. Good-Bad Definition
perf_window = 12
bad_definition = 90

Max_application_date = raw_filter["GI_Application_Date"].max()
raw_filter["MOB"] = (
    Max_application_date - raw_filter["GI_Application_Date"]
).dt.days // 30


def GB_tag(x):
    if x["DL_Total_Late_Day"] >= bad_definition:
        return 1  # Bad
    elif x["DL_Total_Late_Day"] < bad_definition and x["MOB"] >= perf_window:
        return 0  # Good
    else:
        return -1  # Undefined


raw_filter["GB"] = raw_filter.apply(GB_tag, axis=1)

# Remove variable used to define the GB tagging and performance variables
raw_filter = raw_filter.drop(columns=["DL_Total_Late_Day", "MOB"])
raw_filter = raw_filter.drop(columns=["DL_x_Times_Late", "DL_1st_Late_Term"])


# 4. Variable Generation
# In the new variable V3_App_State_eq_NRC_State, 0 means false and 1 means true
#raw_filter["R5_House_Value_to_Req_Loan_Amt"] = (
#    raw_filter["FC_House_Value"] / raw_filter["PL_Requested_Loan_Amount"]
#)
#raw_filter["R8_Farm_Value_to_Farm_Size"] = (
#    raw_filter["FC_Farm_Value"] / raw_filter["FC_Farm_Size"]
#)
raw_filter["FC_House_area_ftsquare"] = (
    raw_filter["FC_House_width.ft."] * raw_filter["FC_House_length.ft."]
)
raw_filter['FC_Income_Expense_Ratio'] = raw_filter['FC_Total_Cash_Income'] / (raw_filter['FC_Total_Business_Expense'] + raw_filter['FC_Total_Personal_Expense'])
raw_filter['FC_Income_Expense_Ratio'] = raw_filter['FC_Income_Expense_Ratio'].replace({np.inf: np.nan})

raw_filter = raw_filter[raw_filter["GB"] != -1].reset_index()
raw_filter = raw_filter.drop(columns="index")

# Remove useless categories from the categorical variables as a result of exceptions and filters
for variable in raw_filter.columns:
    try:
        raw_filter[variable] = raw_filter[variable].cat.remove_unused_categories()
    except:
        pass

# 6. Sample Selection
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=1234)
for train_index, test_index in split.split(raw_filter, raw_filter["GB"]):
    train = raw_filter.loc[train_index]
    test = raw_filter.loc[test_index]

train.to_csv("data\portfolio\Train.csv", index=False)
test.to_csv("data\portfolio\Test.csv", index=False)