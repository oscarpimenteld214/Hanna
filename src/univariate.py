# Libraries
import numpy as np
import pandas as pd
import pickle
import os

import functions as func

pd.options.mode.copy_on_write = True
pd.set_option("mode.copy_on_write", False)

in_path = "data/portfolio/"
out_path = "data/univariate/"
# ------------------------------
# 3. Univariate Analysis
# ------------------------------

train = func.get_data(in_path + "Train.csv", in_path + "dtypes_dict.pkl")
test = func.get_data(in_path + "Test.csv", in_path + "dtypes_dict.pkl")

# Create output folder
os.mkdir(out_path)

# 1. Long list of variables

woe, iv = func.woe_iv(train, "GB", bins=5)
woe.to_csv(out_path + "woe.csv")
iv.to_csv(out_path + "iv.csv")

top20_variables = iv.sort_values(ascending=False).head(20).index

# 2. Visualize the WOE

for var in top20_variables:
    func.woe_plot(woe, var, out_path)

# 3. Coerce classing


def add_coerse_variable(var_name):
    grp_var = var_name + "_gpd"
    train[grp_var] = func.coerse_classing(train[var_name], left_bounds, right_bounds)
    test[grp_var] = func.coerse_classing(test[var_name], left_bounds, right_bounds)


x_shortlisted = []

# FC_Total_Business_Expense
# --------------------------------
var_tocoarse = "FC_Total_Business_Expense"

left_bounds = [-0.001, 70540.0, 328060.0]
right_bounds = [70540.0, 328060.0, 19682000.0]
func.coerse_classing_results(train, var_tocoarse, left_bounds, right_bounds, "GB", out_path)

# Add grouped column to train-test sets
add_coerse_variable(var_tocoarse)

# Add to shortlist of variables
x_shortlisted.append(var_tocoarse + "_gpd")
# --------------------------------

# FC_House_area_ftsquare
# --------------------------------
var_tocoarse = "FC_House_area_ftsquare"
left_bounds = [89.0, 732.8, 1200.0, 1970.0]
right_bounds = [732.8, 1200.0, 1970.0, 75000]
func.coerse_classing_results(train, var_tocoarse, left_bounds, right_bounds, "GB", out_path)

# Add grouped column to train-test sets
add_coerse_variable(var_tocoarse)

# Add to shortlist of variables
x_shortlisted.append(var_tocoarse + "_gpd")
# --------------------------------

# FC_House_width.ft.
# --------------------------------
var_tocoarse = "FC_House_width.ft."

left_bounds = [8.0, 23.0, 40.0]
right_bounds = [23.0, 40.0, np.inf]
func.coerse_classing_results(train, var_tocoarse, left_bounds, right_bounds, "GB", out_path)

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

# FC_Total_Personal_Expense
# --------------------------------
var_tocoarse = "FC_Total_Personal_Expense"
left_bounds = [-0.001, 135400.0, 297360.0]
right_bounds = [135400.0, 297360.0, 4344000.0]
func.coerse_classing_results(train, var_tocoarse, left_bounds, right_bounds, "GB", out_path)

# Add grouped column to train-test sets
add_coerse_variable(var_tocoarse)

# Add to shortlist of variables
x_shortlisted.append(var_tocoarse + "_gpd")

# --------------------------------

# FI_No_Depend
# --------------------------------
var_tocoarse = "FI_No_Depend"
left_bounds = [-0.001, 3.0]
right_bounds = [3.0, 8.0]
func.coerse_classing_results(train, var_tocoarse, left_bounds, right_bounds, "GB", out_path)

# Add grouped column to train-test sets
add_coerse_variable(var_tocoarse)

# Add to shortlist of variables
x_shortlisted.append(var_tocoarse + "_gpd")
# --------------------------------

# FC_Income_Expense_Ratio_woe
# --------------------------------
var_tocoarse = "FC_Income_Expense_Ratio"
left_bounds = [-0.001, 1.379, 2.764]
right_bounds = [1.379, 2.764, np.inf]
func.coerse_classing_results(train, var_tocoarse, left_bounds, right_bounds, "GB", out_path)

# Add grouped column to train-test sets
add_coerse_variable(var_tocoarse)

# Add to shortlist of variables
x_shortlisted.append(var_tocoarse + "_gpd")
# --------------------------------

# FI_No_Depend
# --------------------------------
var_tocoarse = "FI_No_Support"

left_bounds = [0.0, 1.0, 1.0]
right_bounds = [0.0, 1.0, 7.0]
func.coerse_classing_results(train, var_tocoarse, left_bounds, right_bounds, "GB", out_path)

# Add grouped column to train-test sets
add_coerse_variable(var_tocoarse)

# Add to shortlist of variables
x_shortlisted.append(var_tocoarse + "_gpd")
# --------------------------------

# ------------------------------------------------------
# Correlation Analysis
# ------------------------------------------------------

shortlisted_df = train[x_shortlisted]
# Convert variables to codes
for var in x_shortlisted:
    # This if statement should be at the beginning
    if shortlisted_df[var].dtype.name == "object":
        shortlisted_df[var] = shortlisted_df[var].astype("category")
    shortlisted_df.loc[:,var] = shortlisted_df[var].cat.codes
# Compute the correlation matrix
corr_matrix = shortlisted_df.corr(method="spearman") * 100
# Print the variable pairs with a correlation coefficient > 75%
for i in corr_matrix.index:
    for col in corr_matrix.columns:
        if np.abs(corr_matrix.loc[i, col]) > 75 and i != col:
            print("the following variables are correlated: ", i, col)


# save train and test datasets with new grouped variables
train.to_csv(out_path + "Train.csv", index=False)
test.to_csv(out_path + "Test.csv", index=False)

# save shortlisted variables
with open(out_path + 'x_shortlisted.pkl', 'wb') as fp:
    pickle.dump(x_shortlisted, fp)

# copy dtypes_dict.pkl to the new folder
os.system(r"copy data\prepared\dtypes_dict.pkl data\univariate\dtypes_dict.pkl")